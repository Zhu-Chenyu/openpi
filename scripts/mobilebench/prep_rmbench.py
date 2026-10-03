"""Flatten an RMBench LeRobot (GR00T-converted) dataset into arrays for MobileBench training.

The episode-sequence trainer reads ~7 policy-update frames spread over a whole episode
per sample; random-access mp4 decoding for that is slow, so every frame is decoded once
into uint8 memmaps here (put_back_block: 90,857 frames x 2 cams x 240x320x3 = 42 GB).

Outputs in <out>:
  images_<cam>.npy  uint8 [N, H, W, 3]   (memmap; cams: head, front)
  meta.npz          state/action [N, 14], ep_start/ep_len [E], tcp_pos [N, 2, 3],
                    tcp_rot [N, 2, 3, 3] (footprint frame, 0=left 1=right),
                    events [M, 4] = (episode, frame-in-episode, arm, closed) and the prompt.

Usage: python prep_rmbench.py <lerobot_dataset_dir> <out_dir> [num_procs]
"""

import json
import multiprocessing as mp
import os
import sys

import av
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aloha_fk  # noqa: E402

CAMS = {"head": "observation.images.head", "front": "observation.images.front"}
GRIP_DIMS = (6, 13)  # left, right gripper in the 14-dim aloha state (1 = open)


def _decode(path: str) -> np.ndarray:
    with av.open(path) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])


def _load_episode(args):
    root, ep = args
    df = pd.read_parquet(f"{root}/data/chunk-000/episode_{ep:06d}.parquet")
    imgs = {k: _decode(f"{root}/videos/chunk-000/{v}/episode_{ep:06d}.mp4") for k, v in CAMS.items()}
    n = len(df)
    for k, im in imgs.items():
        if len(im) < n:
            raise ValueError(f"episode {ep} cam {k}: {len(im)} frames < {n} rows")
        imgs[k] = im[:n]
    return ep, np.stack(df["observation.state"].values), np.stack(df["action"].values), imgs


def gripper_events(state: np.ndarray, thresh: float = 0.5) -> list[tuple[int, int, int]]:
    """(frame, arm, closed) at every crossing of the commanded gripper opening through `thresh`."""
    out = []
    for arm, dim in enumerate(GRIP_DIMS):
        closed = state[:, dim] < thresh
        for t in np.nonzero(closed[1:] != closed[:-1])[0] + 1:
            out.append((int(t), arm, int(closed[t])))
    return sorted(out)


def main():
    root, out = sys.argv[1], sys.argv[2]
    procs = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    os.makedirs(out, exist_ok=True)
    info = json.load(open(f"{root}/meta/info.json"))
    episodes = [json.loads(line) for line in open(f"{root}/meta/episodes.jsonl")]
    lengths = {e["episode_index"]: e["length"] for e in episodes}
    order = sorted(lengths)
    total = sum(lengths.values())
    assert total == info["total_frames"], (total, info["total_frames"])
    h, w, _ = info["features"][CAMS["head"]]["shape"]
    mm = {k: np.lib.format.open_memmap(f"{out}/images_{k}.npy", "w+", np.uint8, (total, h, w, 3)) for k in CAMS}

    starts, cur = {}, 0
    for ep in order:
        starts[ep], cur = cur, cur + lengths[ep]
    state = np.zeros((total, 14), np.float32)
    action = np.zeros((total, 14), np.float32)
    events = []
    with mp.Pool(procs) as pool:
        for i, (ep, st, ac, imgs) in enumerate(pool.imap_unordered(_load_episode, [(root, e) for e in order])):
            s, n = starts[ep], lengths[ep]
            assert len(st) == n, (ep, len(st), n)
            state[s : s + n], action[s : s + n] = st, ac
            for k, im in imgs.items():
                mm[k][s : s + n] = im
            events += [(ep, *e) for e in gripper_events(st)]
            if i % 25 == 0:
                print(f"{i + 1}/{len(order)} episodes", flush=True)
    for v in mm.values():
        v.flush()

    tcp_pos, tcp_rot = aloha_fk.both_tcp(state)
    task = json.loads(open(f"{root}/meta/tasks.jsonl").readline())["task"]
    np.savez(
        f"{out}/meta.npz",
        state=state,
        action=action,
        ep_index=np.array(order),
        ep_start=np.array([starts[e] for e in order]),
        ep_len=np.array([lengths[e] for e in order]),
        tcp_pos=tcp_pos.astype(np.float32),
        tcp_rot=tcp_rot.astype(np.float32),
        events=np.array(sorted(events), dtype=np.int64),
        prompt=np.array(task),
        fps=np.array(info["fps"]),
    )
    pattern = {}
    for ep in order:
        key = tuple((a, c) for e, _, a, c in events if e == ep)
        pattern[key] = pattern.get(key, 0) + 1
    print(f"wrote {total} frames / {len(order)} episodes to {out}")
    print("gripper event patterns (arm 0=L 1=R, closed):")
    for k, v in sorted(pattern.items(), key=lambda kv: -kv[1]):
        print(f"  {v:4d}x  {k}")


if __name__ == "__main__":
    main()
