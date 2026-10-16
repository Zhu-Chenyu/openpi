"""Smoke / wiring checks of MobileBench v1.11 on real RMBench batches with a tiny ("dummy") Gemma.

    python scripts/mobilebench/check_v111.py <prep dir with phase_labels.npz> [--device cuda]

Checks: finite loss; every v1.11 module gets a gradient; the ACTION loss alone reaches the
memory input projections and the writers of earlier updates (i.e. through the VLM); the slow
group is written only on updates 0, K, 2K, ...; the action heads cannot see memory positions
directly; inference runs and resets the memory at a new episode.
"""

import argparse
import dataclasses
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rmbench_episodes as rme  # noqa: E402
import train_rmbench  # noqa: E402

from openpi.models import model as _model  # noqa: E402
from openpi.models import pi0_config  # noqa: E402
from openpi.models_pytorch.mobilebench.v111 import LossWeightsV111  # noqa: E402
from openpi.models_pytorch.mobilebench.v111 import MobileBenchV111  # noqa: E402
from openpi.models_pytorch.mobilebench.v111 import V111Config  # noqa: E402
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device(a.device)
    pi0 = PI0Pytorch(pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy",
                                          pytorch_compile_mode=None))  # fmt: skip
    cfg = V111Config(vlm_width=pi0.paligemma_with_expert.paligemma.config.text_config.hidden_size)
    model = MobileBenchV111(pi0, cfg, vlm_lora=True, expert_lora=True).to(dev)
    model.train()

    tf, _ = train_rmbench.frame_transform()
    lib = rme.Library.build(np.load(f"{a.data}/meta.npz")["state"])
    ds = rme.EpisodeSequences(a.data, tf, cfg, lib, stride=50, trace_lags=4)
    items = [ds[0], ds[1]]
    for it in items:  # keep it small: 5 updates per episode
        for k in ("obs",):
            it[k] = it[k][:5]
        it["step"] = {k: v[:5] for k, v in it["step"].items()}
        it["labels"] = {k: v[:5] for k, v in it["labels"].items()}
    batch = train_rmbench.to_device(rme.collate(items), dev)
    obs_dict = batch["observation"]
    batch["observation"] = _model.Observation.from_dict(obs_dict)
    n = batch["actions"].shape[0]
    am = torch.zeros(model.action_dim, dtype=torch.bool, device=dev)
    am[:14] = True
    batch["action_mask"] = am[None].expand(n, -1)
    phase_text = torch.as_tensor(ds.phase_text, device=dev)
    print("phase labels of episode 0:", batch["labels"]["phase_id"][0].tolist(),
          "present", batch["labels"]["phase_present"][0].tolist())  # fmt: skip

    loss, logs, mem = model.episode_loss(batch, LossWeightsV111(), phase_text)
    assert torch.isfinite(loss), logs
    print("logs:", {k: round(v, 4) for k, v in logs.items()})
    loss.backward()
    new_mods = ["in_fast", "in_slow", "in_type", "in_slot_fast", "motion_enc", "fast_init", "slow_init", "p_h",
                "fast_state", "writer_fast", "writer_slow", "enc_workspace", "enc_state", "act_fast", "act_slow",
                "cond_type", "aff_in", "aff_read", "aff_point", "aff_valid", "phase_read", "phase_out"]  # fmt: skip
    for name in new_mods:
        g = sum(p.grad.abs().sum().item() for p in getattr(model, name).parameters() if p.grad is not None) \
            if isinstance(getattr(model, name), torch.nn.Module) else getattr(model, name).grad.abs().sum().item()
        print(f"  grad |{name}| = {g:.3e}")
        assert g > 0 or name in ("in_fast", "in_slow", "motion_enc"), name  # zero-init outputs: weight grads only below
    # Zero-initialised output layers: their weights must still receive a gradient.
    for name, lin in (("in_fast", model.in_fast[1]), ("in_slow", model.in_slow[1]), ("motion_enc", model.motion_enc[2])):
        assert lin.weight.grad is not None and lin.weight.grad.abs().sum() > 0, name

    # Action loss only: must reach the slow writer (written at update 0, read by the VLM at update 1+).
    model.zero_grad(set_to_none=True)
    w = dataclasses.replace(LossWeightsV111(), aff=0.0, slow_text=0.0)
    with torch.no_grad():  # give the zero-initialised memory input projections a non-zero value
        for lin in (model.in_fast[1], model.in_slow[1]):
            lin.weight.normal_(0, 0.02)
    loss, _, _ = model.episode_loss(batch, w, phase_text)
    loss.backward()
    gs = model.writer_slow[0].weight.grad
    print("action-only grad |writer_slow| =", None if gs is None else f"{gs.abs().sum().item():.3e}")
    assert gs is not None and gs.abs().sum() > 0, "action loss does not reach the slow writer"

    # Slow writes on updates 0, K, 2K only.
    with torch.no_grad():
        mem = model.init_memory(1, dev)
        emb, pad = model.embed_observation(batch["observation"], train=False)
        written = []
        for k in range(5):
            st = {name: v[0:1, k] for name, v in batch["step"].items()}
            out = model.update(emb[k : k + 1], pad[k : k + 1], st, mem, torch.zeros(1, dtype=torch.long, device=dev))
            written.append(bool(out.slow_written[0]))
            mem = out.memory
        print("slow written per update:", written)
        assert written == [k % cfg.slow_write_every == 0 for k in range(5)]
        assert not out.head_pad[0, pad.shape[1] :].any(), "heads must not read memory positions"

        model.eval()
        first = lambda v: {kk: vv[:1] for kk, vv in v.items()} if isinstance(v, dict) else v[:1]  # noqa: E731
        obs1 = _model.Observation.from_dict({k: first(v) for k, v in obs_dict.items()})
        st = {name: v[0:1, 0] for name, v in batch["step"].items()}
        act, _, out = model.infer(obs1, st, mem, num_steps=2)
        assert act.shape == (1, model.action_horizon, model.action_dim) and torch.isfinite(act).all()
        assert int(out.memory.step[0]) == 1, "new_episode must reset the memory before the update"
    print("all v1.11 checks passed")


if __name__ == "__main__":
    main()
