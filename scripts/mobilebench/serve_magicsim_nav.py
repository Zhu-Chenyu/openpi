"""Websocket policy server for MobileBench v1.11 in MagicSim go2 navigation closed-loop eval.

MagicSim (Isaac Sim) runs in its own venv and talks to this process through openpi_client.
Like serve_rmbench.py the server keeps the memory across calls: one call = one policy update
= one memory write, and the client sets `reset=True` on the first call of every episode.

    python scripts/mobilebench/serve_magicsim_nav.py --ckpt <out>/<exp>/<step> --port 8792

Request: {"image": HxWx3 uint8 onboard camera, "state": [vx, vy, yaw rate] measured in the body
          frame, "pos": [x, y, z] and "yaw": base pose (odometry), "reset": bool}
Reply:   {"actions": [50, 3]} commanded (vx, vy, yaw rate) for the next 50 ticks (50 Hz), plus
          "slow_readout" (nearest slow-memory sentence) and "aff" (predicted NavTo point and
          target centre in the body frame) for logging.
"""

import argparse
import functools
import json
import logging
import os
import sys

import numpy as np
import safetensors.torch
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nav_episodes  # noqa: E402
import train_rmbench  # noqa: E402

from openpi.models import model as _model  # noqa: E402
from openpi.models_pytorch.mobilebench.v111 import V111Config  # noqa: E402
from openpi_client import base_policy  # noqa: E402
from openpi.serving import websocket_policy_server  # noqa: E402


class NavMemoryPolicy(base_policy.BasePolicy):
    def __init__(self, ckpt: str, weights: str, num_steps: int = 10, stride: int = 50, fps: float = 50.0):
        exp_dir = os.path.dirname(ckpt.rstrip("/"))
        self.cfg = V111Config(**json.load(open(f"{exp_dir}/{train_rmbench.CONFIG_FILES['v1.11']}")))
        pt = np.load(f"{exp_dir}/phase_texts.npz")
        self.phase_texts, self.phase_emb = [str(t) for t in pt["texts"]], torch.as_tensor(pt["text_emb"])
        self.desc = np.load(f"{exp_dir}/nav_base_desc.npy").astype(np.float32)
        targs = json.load(open(f"{exp_dir}/train_args.json"))
        self.dev = torch.device("cuda")
        self.model = train_rmbench.build_model(
            weights, self.cfg, self.dev, vlm_lora=targs.get("vlm_lora", False), expert_lora=targs.get("expert_lora", False)
        )
        state = safetensors.torch.load_file(f"{ckpt}/trainable.safetensors", device="cuda")
        _, unexpected = self.model.load_state_dict(state, strict=False)
        missing = sorted({k for k, p in self.model.named_parameters() if p.requires_grad} - set(state))
        if missing or unexpected:
            raise RuntimeError(f"checkpoint mismatch: missing {missing[:5]} unexpected {unexpected[:5]}")
        self.model.eval()
        self.tf, self.stats = nav_episodes.frame_transform(f"{exp_dir}/assets")
        self.prompt = "Navigate to the fridge."
        self.base_mask = torch.zeros(1, self.model.action_dim, dtype=torch.bool, device=self.dev)
        self.base_mask[:, : nav_episodes.BASE_DIMS] = True
        self.num_steps, self.dt = num_steps, stride / fps
        self.memory, self.prev, self.updates = None, None, 0

    def infer(self, obs: dict) -> dict:
        if obs.get("reset") or self.memory is None:
            self.memory, self.prev, self.updates = self.model.init_memory(1, self.dev), None, 0
        d = self.tf({"image": np.asarray(obs["image"])[..., :3], "state": np.asarray(obs["state"], np.float32),
                     "prompt": self.prompt})  # fmt: skip
        batch = {k: (v[None] if not isinstance(v, dict) else {kk: vv[None] for kk, vv in v.items()}) for k, v in d.items()}
        batch = train_rmbench.to_device(
            {k: (torch.from_numpy(np.asarray(v)) if not isinstance(v, dict)
                 else {kk: torch.from_numpy(np.asarray(vv)) for kk, vv in v.items()}) for k, v in batch.items()},
            self.dev,
        )  # fmt: skip
        observation = _model.Observation.from_dict(batch)
        pos, yaw = np.asarray(obs["pos"], np.float64), float(obs["yaw"])
        prev = self.prev or (None, None)
        st = nav_episodes.step_inputs(pos, yaw, *prev, self.dt, np.asarray(d["state"]), self.cfg, self.desc)
        st = {k: torch.as_tensor(np.asarray(v))[None].to(self.dev) for k, v in st.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, act_b, out = self.model.infer(observation, st, self.memory, num_steps=self.num_steps,
                                             base_dim_mask=self.base_mask)  # fmt: skip
        self.memory, self.prev = out.memory, (pos, yaw)
        self.updates += 1
        actions = nav_episodes.unnormalize_base(act_b[0].float().cpu().numpy(), self.stats)
        readout = self.phase_texts[int((out.phase_z[0].float().cpu() @ self.phase_emb.T).argmax())]
        aff = out.aff_points[0].float().cpu().numpy()
        logging.info("update %d: slow_written=%s nav=%s obj=%s cmd0=%s | %s", self.updates, bool(out.slow_written[0]),
                     np.round(aff[0], 2), np.round(aff[1], 2), np.round(actions[0], 2), readout)  # fmt: skip
        return {"actions": actions.astype(np.float32), "slow_readout": readout, "aff": aff.astype(np.float32)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--weights", required=True, help="pi05_base PyTorch model.safetensors")
    ap.add_argument("--port", type=int, default=8792)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    policy = NavMemoryPolicy(args.ckpt, args.weights)
    # Localhost only; the client blocks in Isaac Sim between calls, so no keepalive pings.
    websocket_policy_server._server.serve = functools.partial(websocket_policy_server._server.serve, ping_interval=None)
    server = websocket_policy_server.WebsocketPolicyServer(policy, host="127.0.0.1", port=args.port)
    logging.info("serving %s on port %d", args.ckpt, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
