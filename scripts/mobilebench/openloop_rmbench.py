"""Open-loop action accuracy of MobileBench pi0.5 checkpoints on recorded RMBench episodes.

Replays recorded episodes through serve_rmbench.MemoryPolicy exactly as the closed-loop client
does (head -> cam_high, front -> cam_left_wrist, one call per 50-frame update, memory reset at
episode start, recorded state as input) and compares each predicted 50-step chunk with the
recorded actions, in joint space and as TCP position (aloha_fk). Flow-matching noise is seeded
per (episode, update), so several checkpoints see identical noise.

    python openloop_rmbench.py --data <prep dir with images_*.npy> --episodes 20 \
        --ckpt <exp>/<step> [--ckpt ...]
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aloha_fk  # noqa: E402
import serve_rmbench  # noqa: E402

STRIDE, H = 50, 50
WEIGHTS = "/data/users/dfv1344/openpi_data/pi05_base_pytorch/model.safetensors"


def run(ckpt, meta, imgs, eps, prompt):
    pol = serve_rmbench.MemoryPolicy(ckpt, WEIGHTS)
    joint, tcp, tcp_end, grip = [], [], [], []
    for e in eps:
        s0, n = int(meta["ep_start"][e]), int(meta["ep_len"][e])
        for u, t in enumerate(range(0, n, STRIDE)):
            torch.manual_seed(1000 * e + u)
            obs = {
                "images": {
                    "cam_high": np.ascontiguousarray(imgs["head"][s0 + t].transpose(2, 0, 1)),
                    "cam_left_wrist": np.ascontiguousarray(imgs["front"][s0 + t].transpose(2, 0, 1)),
                },
                "state": meta["state"][s0 + t],
                "prompt": prompt,
                "reset": u == 0,
            }
            pred = np.asarray(pol.infer(obs)["actions"])[:H, :14]
            idx = np.clip(np.arange(t, t + H), 0, n - 1)
            gt = meta["action"][s0 + idx]
            joint.append(np.abs(pred - gt).mean(0))
            p_pos, _ = aloha_fk.both_tcp(pred.astype(np.float64))
            g_pos, _ = aloha_fk.both_tcp(gt.astype(np.float64))
            d = np.linalg.norm(p_pos - g_pos, axis=-1)  # [H, 2 arms]
            tcp.append(d.mean(0))
            tcp_end.append(d[-1])
            grip.append(np.abs(pred[:, [6, 13]] - gt[:, [6, 13]]).mean(0))
    del pol
    torch.cuda.empty_cache()
    return np.array(joint), np.array(tcp), np.array(tcp_end), np.array(grip)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/users/dfv1344/mobilebench_data/put_back_block")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--ckpt", action="append", required=True)
    a = ap.parse_args()
    meta = dict(np.load(f"{a.data}/meta.npz"))
    imgs = {c: np.load(f"{a.data}/images_{c}.npy", mmap_mode="r") for c in ("head", "front")}
    prompt = str(meta["prompt"]) if "prompt" in meta else str(meta["prompts"][0])
    eps = np.linspace(0, len(meta["ep_len"]) - 1, a.episodes).astype(int)
    print(f"{len(eps)} episodes from {a.data}, prompt: {prompt!r}")
    for ck in a.ckpt:
        joint, tcp, tcp_end, grip = run(ck, meta, imgs, eps, prompt)
        print(f"\n== {ck}  ({len(joint)} chunks)")
        print(f"  joint MAE (rad) left arm  {joint[:, :6].mean():.4f}   right arm {joint[:, 7:13].mean():.4f}")
        print(f"  right-arm joint MAE per joint {np.round(joint[:, 7:13].mean(0), 4)}")
        print(f"  TCP error over chunk (cm)  left {100 * tcp[:, 0].mean():.2f}  right {100 * tcp[:, 1].mean():.2f}"
              f"   at chunk end: left {100 * tcp_end[:, 0].mean():.2f}  right {100 * tcp_end[:, 1].mean():.2f}")
        print(f"  right TCP chunk-end error > 3 cm in {100 * (tcp_end[:, 1] > 0.03).mean():.0f}% of chunks")
        print(f"  gripper MAE left {grip[:, 0].mean():.3f}  right {grip[:, 1].mean():.3f}")


if __name__ == "__main__":
    main()
