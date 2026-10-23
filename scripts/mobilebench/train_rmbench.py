"""Train MobileBench pi0.5 (frozen VLM + memory/affordance/goal conditioner + dual action heads)
on RMBench put_back_block with whole-episode sequences.

Uses the SAME data transforms and norm stats as the pi0.5 LoRA baseline config
(`pi05_rmbench_put_back_block_lora`), so the two runs differ only in the model.

    python scripts/mobilebench/train_rmbench.py --exp mb_v1 --steps 8000

Throughput notes (single RTX 6000 Ada): the frozen VLM runs once per update frame under
no_grad in bf16, the all-masked right-wrist image is skipped, frames come from
pre-decoded memmaps (prep_rmbench.py) through a multi-worker loader, and the log line
reports the fraction of time spent waiting for data so a loader stall is visible.
"""

import argparse
import dataclasses
import json
import logging
import math
import os
import shutil
import sys
import time

import numpy as np
import safetensors.torch
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nav_episodes  # noqa: E402
import rmbench_episodes as rme  # noqa: E402

from openpi.models import model as _model  # noqa: E402
from openpi.models import pi0_config  # noqa: E402
from openpi.models_pytorch.mobilebench.config import MobileBenchConfig  # noqa: E402
from openpi.models_pytorch.mobilebench.model import LossWeights  # noqa: E402
from openpi.models_pytorch.mobilebench.model import MobileBenchPi05  # noqa: E402
from openpi.models_pytorch.mobilebench.model import trainable_parameter_groups  # noqa: E402
from openpi.models_pytorch.mobilebench.v111 import LossWeightsV111  # noqa: E402
from openpi.models_pytorch.mobilebench.v111 import MobileBenchV111  # noqa: E402
from openpi.models_pytorch.mobilebench.v111 import V111Config  # noqa: E402
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch  # noqa: E402
from openpi.training import config as _config  # noqa: E402
from openpi.training import task_mix  # noqa: E402
import openpi.transforms as _transforms  # noqa: E402

# The pi0.5 LoRA baseline config whose data transforms / norm stats both runs share.
BASELINE_CONFIG = "pi05_rmbench_put_back_block_lora"


# Per-architecture config file in the run dir (serve_rmbench.py picks the model from it).
CONFIG_FILES = {"v1.2": "mobilebench_config.json", "v1.11": "mobilebench_v111_config.json"}


def mobilebench_config(arch: str = "v1.2"):
    if arch == "v1.11":
        return V111Config()
    return MobileBenchConfig(
        num_end_effectors=2,  # aloha left + right
        action_workspace_tokens=16,  # pool the 128 library tokens before the action heads
        trace_lags=2 * 4,  # 4 lags x 2 arms
    )


def frame_transform(assets_dirs=None, config_name: str = BASELINE_CONFIG):
    cfg = _config.get_config(config_name)
    data = cfg.data.create(assets_dirs or cfg.assets_dirs, cfg.model)
    tfs = [
        *data.repack_transforms.inputs,
        *data.data_transforms.inputs,
        _transforms.Normalize(data.norm_stats, use_quantiles=data.use_quantile_norm),
        *data.model_transforms.inputs,
    ]
    return _transforms.compose(tfs), data


def build_model(weights: str, mb_cfg, device, *, vlm_lora=False, expert_lora=False):
    pcfg = pi0_config.Pi0Config(pi05=True, pytorch_compile_mode=None)
    pi0 = PI0Pytorch(pcfg)
    missing, unexpected = safetensors.torch.load_model(pi0, weights, strict=False)
    if missing or unexpected:
        logging.warning("pi05 weights: %d missing, %d unexpected (e.g. %s)", len(missing), len(unexpected),
                        (missing or unexpected)[:3])  # fmt: skip
    cls = MobileBenchV111 if isinstance(mb_cfg, V111Config) else MobileBenchPi05
    model = cls(pi0, mb_cfg, vlm_lora=vlm_lora, expert_lora=expert_lora)
    del pi0
    return model.to(device)


def to_device(batch: dict, device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            out[k] = to_device(v, device)
        elif torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def lr_at(step: int, warmup: int, total: int, floor: float = 0.1) -> float:
    if step < warmup:
        return (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument(
        "--dataset", choices=("rmbench", "magicsim_nav"), default="rmbench",
        help="rmbench: aloha arms, prep_rmbench*.py data, --baseline_config transforms; magicsim_nav: go2 base "
        "commands to the base head, prep_magicsim_nav.py data with its own norm stats (v1.11 only)",
    )  # fmt: skip
    ap.add_argument(
        "--arch", choices=sorted(CONFIG_FILES), default="v1.2",
        help="v1.2: conditioner after a frozen-input VLM; v1.11: memory tokens into the VLM, MLP writers, "
        "slow-text supervision (needs phase_labels.npz from annotate_phases.py in every data dir)",
    )  # fmt: skip
    ap.add_argument(
        "--data", default="/data/users/dfv1344/mobilebench_data/put_back_block",
        help="prepped data dir, or several comma-separated (e.g. the original demos plus newly collected ones)",
    )  # fmt: skip
    ap.add_argument("--weights", default="/data/users/dfv1344/openpi_data/pi05_base_pytorch/model.safetensors")
    ap.add_argument("--out", default="/data/users/dfv1344/mobilebench_checkpoints/put_back_block")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--episodes_per_batch", type=int, default=8)
    ap.add_argument(
        "--grad_accum", type=int, default=1,
        help="micro-batches of --episodes_per_batch per optimizer step (same update, less GPU memory)",
    )  # fmt: skip
    ap.add_argument("--stride", type=int, default=50)
    ap.add_argument("--lr_pretrained", type=float, default=2.5e-5)
    ap.add_argument("--lr_new", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=300)
    # openpi LoRA recipe of the pi0.5 baseline (gemma_2b_lora + gemma_300m_lora).
    ap.add_argument("--vlm_lora", action="store_true")
    ap.add_argument("--expert_lora", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--wandb_project", default="rmbench-pi05")
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--max_steps_debug", type=int, default=None, help="stop early (throughput check)")
    ap.add_argument("--baseline_config", default=BASELINE_CONFIG, help="openpi config for transforms + norm stats")
    ap.add_argument(
        "--ema", type=float, default=0.99,
        help="EMA decay of the trainable weights (openpi's default 0.99; 0 = off). Checkpoints store the EMA "
        "weights as trainable.safetensors (what evaluation loads) and the raw ones as trainable_raw.safetensors.",
    )  # fmt: skip
    ap.add_argument(
        "--window", type=int, default=0,
        help="TBPTT window in updates (0 = whole episodes per batch). >0 streams episodes through "
        "--episodes_per_batch slots and carries the memory across windows (sec. 9.6).",
    )  # fmt: skip
    ap.add_argument(
        "--init_ckpt", default=None,
        help="continue from <exp>/<step> of an earlier run: its trainable.safetensors (EMA weights) initialise the "
        "model and the EMA shadow; optimizer, LR schedule (warmup + cosine) and step counter start fresh.",
    )  # fmt: skip
    ap.add_argument("--init_frames", type=int, default=0, help="frames the --init_ckpt run had seen (logging offset)")
    ap.add_argument(
        "--task_shares", default=None,
        help='e.g. "put_back_block=0.3": that task gets 30%% of the sampled frames, the rest split over the other '
        "tasks by their frames (openpi.training.task_mix). Default: frame-proportional.",
    )  # fmt: skip
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    out_dir = os.path.join(args.out, args.exp)
    os.makedirs(out_dir, exist_ok=True)

    mb_cfg = mobilebench_config(args.arch)
    data_dirs = args.data.split(",")
    nav = args.dataset == "magicsim_nav"
    if nav and (args.arch != "v1.11" or len(data_dirs) != 1):
        raise ValueError("--dataset magicsim_nav needs --arch v1.11 and one data dir (its norm stats)")
    if nav:
        tf, _ = nav_episodes.frame_transform(f"{data_dirs[0]}/assets")
    else:
        tf, data_cfg = frame_transform(config_name=args.baseline_config)
    if nav:
        lib = None
    elif args.init_ckpt:  # the workspace library must be the one the init weights were trained with
        lib = rme.Library.load(f"{os.path.dirname(args.init_ckpt.rstrip('/'))}/library.npz")
    else:
        lib = rme.Library.build(np.concatenate([np.load(f"{d}/meta.npz")["state"] for d in data_dirs]))
    if lib is not None:
        lib.save(f"{out_dir}/library.npz")
    json.dump(dataclasses.asdict(mb_cfg), open(f"{out_dir}/{CONFIG_FILES[args.arch]}", "w"), indent=1)
    json.dump(vars(args), open(f"{out_dir}/train_args.json", "w"), indent=1)
    # Norm stats travel with the checkpoint so evaluation un-normalises exactly like training.
    assets = f"{data_dirs[0]}/assets" if nav else _config.get_config(args.baseline_config).assets_dirs
    shutil.copytree(assets, f"{out_dir}/assets", dirs_exist_ok=True)

    if nav:
        parts = [nav_episodes.NavEpisodeSequences(d, tf, mb_cfg, stride=args.stride) for d in data_dirs]
    else:
        parts = [rme.EpisodeSequences(d, tf, mb_cfg, lib, stride=args.stride, trace_lags=4) for d in data_dirs]
    ds = parts[0] if len(parts) == 1 else torch.utils.data.ConcatDataset(parts)
    ep_task = [t for p in parts for t in p.ep_task]
    ep_len = np.concatenate([p.meta["ep_len"] for p in parts])
    logging.info("data: %s", ", ".join(f"{d} ({len(p)} episodes)" for d, p in zip(data_dirs, parts)))
    phase_text = None
    if args.arch == "v1.11":
        if any(p.phase_texts is None for p in parts) or any(p.phase_texts != parts[0].phase_texts for p in parts):
            raise RuntimeError("v1.11 needs the same phase_labels.npz text table in every data dir")
        np.savez(f"{out_dir}/phase_texts.npz", texts=np.array(parts[0].phase_texts), text_emb=parts[0].phase_text)
        phase_text = torch.as_tensor(parts[0].phase_text, device=device)
    phase_group = ("side", torch.as_tensor(parts[0].phase_group, device=device)) if nav else None
    sampler = None
    if args.task_shares:
        shares = {k: float(v) for k, v in (kv.split("=") for kv in args.task_shares.split(","))}
        # One draw = one whole episode, so per-episode task weights give frame shares (task_mix docstring).
        w = task_mix.task_weights(ep_task, ep_len, shares)
        logging.info("task frame shares: %s", {k: round(v, 3) for k, v in task_mix.frame_shares(ep_task, ep_len, w).items()})
        sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(w), num_samples=len(ds), replacement=True)
    if args.window > 0:
        ep_loader = torch.utils.data.DataLoader(
            ds, batch_size=1, shuffle=sampler is None, sampler=sampler, num_workers=args.workers,
            collate_fn=rme.first, persistent_workers=True, prefetch_factor=4,
        )  # fmt: skip

        def episodes():
            while True:
                yield from ep_loader

        stream = rme.WindowStream(episodes(), slots=args.episodes_per_batch, window=args.window)
        batches = iter(stream.next_batch, None)
    else:
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=args.episodes_per_batch,
            shuffle=sampler is None,
            sampler=sampler,
            drop_last=True,
            num_workers=args.workers,
            collate_fn=rme.collate,
            persistent_workers=True,
            prefetch_factor=4,
            pin_memory=True,
        )

        def whole_episodes():
            while True:
                yield from loader

        batches = whole_episodes()

    model = build_model(args.weights, mb_cfg, device, vlm_lora=args.vlm_lora, expert_lora=args.expert_lora)
    if args.init_ckpt:
        state = safetensors.torch.load_file(f"{args.init_ckpt}/trainable.safetensors", device=str(device))
        _, unexpected = model.load_state_dict(state, strict=False)
        missing = sorted({k for k, p in model.named_parameters() if p.requires_grad} - set(state))
        if missing or unexpected:
            raise RuntimeError(f"--init_ckpt mismatch: missing {missing[:5]} unexpected {unexpected[:5]}")
        logging.info("initialised %d trainable tensors from %s", len(state), args.init_ckpt)
    model.train()
    groups = trainable_parameter_groups(model, args.lr_pretrained, args.lr_new)
    for g in groups:
        g["base_lr"] = g["lr"]
    n_train = sum(p.numel() for g in groups for p in g["params"])
    n_total = sum(p.numel() for p in model.parameters())
    logging.info("params: %.1fM trainable / %.1fM total", n_train / 1e6, n_total / 1e6)
    optim = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=1e-10, eps=1e-8)
    train_named = [(k, p) for k, p in model.named_parameters() if p.requires_grad]
    train_params = [p for _, p in train_named]
    # fp32 shadow copy, updated after every optimizer step: w_ema = d * w_ema + (1 - d) * w.
    ema = [p.detach().clone().float() for p in train_params] if args.ema > 0 else None
    weights = LossWeightsV111() if args.arch == "v1.11" else LossWeights()
    # RMBench put_back_block: 14 real action dims (left 6+1, right 6+1) of the padded 32 for the upper head.
    # MagicSim nav: no arm; the base head predicts (vx, vy, yaw rate) in its first 3 dims.
    action_mask = torch.zeros(model.action_dim, dtype=torch.bool, device=device)
    base_mask = torch.zeros(model.action_dim, dtype=torch.bool, device=device)
    if nav:
        base_mask[: nav_episodes.BASE_DIMS] = True
    else:
        action_mask[:14] = True

    run = None
    if not args.no_wandb:
        import wandb

        run = wandb.init(project=args.wandb_project, name=args.exp, config={**vars(args), **dataclasses.asdict(mb_cfg)})

    step, frames_seen, memory = 0, args.init_frames, None
    t_last, wait_acc, frames_acc = time.time(), 0.0, 0
    total_steps = args.max_steps_debug or args.steps
    while step < total_steps:
        scale = lr_at(step, args.warmup, args.steps)
        for g in optim.param_groups:
            g["lr"] = g["base_lr"] * scale
        optim.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            t_wait = time.time()
            batch = next(batches)
            wait_acc += time.time() - t_wait
            batch = to_device(batch, device)
            obs = _model.Observation.from_dict(batch["observation"])
            n = batch["actions"].shape[0]
            frames_acc += n
            frames_seen += n
            batch["observation"] = obs
            batch["action_mask"] = action_mask[None].expand(n, -1)
            if "base_actions" in batch:
                batch["base_mask"] = base_mask[None].expand(n, -1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                carry = memory if args.window > 0 else None
                if args.arch == "v1.11":
                    loss, logs, memory = model.episode_loss(batch, weights, phase_text, carry, phase_group)
                else:
                    loss, logs, memory = model.episode_loss(batch, weights, carry)
            (loss / args.grad_accum).backward()
        # Clip per group: the new modules' auxiliary-loss gradients are large early on
        # and a single global norm would shrink the pretrained expert's FM update with them.
        gnorms = [torch.nn.utils.clip_grad_norm_(g["params"], 1.0) for g in groups]
        optim.step()
        if ema is not None:
            with torch.no_grad():
                torch._foreach_mul_(ema, args.ema)
                torch._foreach_add_(ema, [p.float() for p in train_params], alpha=1.0 - args.ema)
        step += 1

        if step % args.log_every == 0 or step == 1:
            dt = time.time() - t_last
            logs.update(
                grad_norm_pretrained=float(gnorms[0]),
                grad_norm_new=float(gnorms[1]),
                lr_pretrained=optim.param_groups[0]["lr"],
                lr_new=optim.param_groups[1]["lr"],
                sec_per_step=dt / (args.log_every if step > 1 else 1),
                frames_per_sec=frames_acc / dt,
                frames_seen=frames_seen,
                data_wait_frac=wait_acc / dt,
                gpu_mem_gb=torch.cuda.max_memory_allocated() / 1e9,
            )
            logging.info("step %d %s", step, " ".join(f"{k}={v:.4g}" for k, v in logs.items()))
            if run is not None:
                run.log(logs, step=step)
            t_last, wait_acc, frames_acc = time.time(), 0.0, 0

        if step % args.save_every == 0 or step == args.steps:
            ck = f"{out_dir}/{step}"
            os.makedirs(ck, exist_ok=True)
            # Only what training changes; frozen pretrained weights come from --weights.
            raw = {k: p.detach().contiguous() for k, p in train_named}
            if ema is not None:
                safetensors.torch.save_file(raw, f"{ck}/trainable_raw.safetensors")
                ema_state = {k: e.to(p.dtype).contiguous() for (k, p), e in zip(train_named, ema, strict=True)}
                safetensors.torch.save_file(ema_state, f"{ck}/trainable.safetensors")
            else:
                safetensors.torch.save_file(raw, f"{ck}/trainable.safetensors")
            logging.info("saved %s (%d frames seen)", ck, frames_seen)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
