"""LeRobot v2.0 dataset for ONE RMBench task = its episodes of a multi-task LeRobot dataset + newly collected demos.

The pi0.5 LoRA baseline (openpi configs pi05_rmbench_<task>_lora) trains on this; MobileBench trains on the
prep dirs of the same episodes (prep_rmbench_task.py + prep_rmbench_hdf5.py --jpeg), so both see the same demos.

Original episodes (those whose task is the collected demos' prompt) are renumbered from 0: their parquets are
rewritten with the new episode / frame index and task_index 0, their mp4s symlinked. Collected demos come from
a prep_rmbench_hdf5.py --jpeg dir and are written in the same layout (one parquet per episode with the
original's columns, h264/yuv420p mp4s per camera at the dataset fps), as Memory-Augmented-VLA
scripts/rmbench/make_lerobot_with_collected.py did for put_back_block_clean500.

Usage: python make_lerobot_rmbench_task.py <multi-task lerobot root> <collected prep dir> <out root> [--ffmpeg ffmpeg]
"""

import argparse
import json
import os
import shutil
import subprocess

import cv2
import numpy as np
import pandas as pd

CAMS = {"head": "observation.images.head", "front": "observation.images.front"}


def encode(frames: np.ndarray, path: str, fps: int, ffmpeg: str):
    n, h, w, _ = frames.shape
    cmd = [ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "2", "-crf", "18", path]  # fmt: skip
    subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("orig")
    ap.add_argument("prep")
    ap.add_argument("out")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    a = ap.parse_args()
    info = json.load(open(f"{a.orig}/meta/info.json"))
    fps, chunk = info["fps"], info["chunks_size"]
    meta = np.load(f"{a.prep}/meta.npz")
    task = str(meta["prompt"])
    orig_eps = [e for e in map(json.loads, open(f"{a.orig}/meta/episodes.jsonl")) if e["tasks"] == [task]]
    if not orig_eps:
        raise ValueError(f"no episodes with task {task!r} in {a.orig}")

    data_dir = f"{a.out}/data/chunk-000"
    vid_dirs = {c: f"{a.out}/videos/chunk-000/{k}" for c, k in CAMS.items()}
    for d in [data_dir, *vid_dirs.values(), f"{a.out}/meta"]:
        os.makedirs(d, exist_ok=True)
    episodes, index, cols = [], 0, None
    for ep, e in enumerate(orig_eps):
        i = e["episode_index"]
        df = pd.read_parquet(f"{a.orig}/data/chunk-{i // chunk:03d}/episode_{i:06d}.parquet")
        n = len(df)
        df["episode_index"] = np.full(n, ep, np.int64)
        df["index"] = np.arange(index, index + n, dtype=np.int64)
        df["task_index"] = np.zeros(n, np.int64)
        df.to_parquet(f"{data_dir}/episode_{ep:06d}.parquet")
        cols = cols or list(df.columns)
        for c, k in CAMS.items():
            dst = f"{vid_dirs[c]}/episode_{ep:06d}.mp4"
            if not os.path.lexists(dst):
                os.symlink(os.path.realpath(f"{a.orig}/videos/chunk-{i // chunk:03d}/{k}/episode_{i:06d}.mp4"), dst)
        episodes.append({"episode_index": ep, "tasks": [task], "length": n})
        index += n
    print(f"{len(orig_eps)} original episodes, {index} frames", flush=True)

    jpeg = {c: (np.memmap(f"{a.prep}/jpeg_{c}.bin", np.uint8, "r"), np.load(f"{a.prep}/jpeg_{c}_off.npy")) for c in CAMS}
    first = len(episodes)
    for j, (s0, n) in enumerate(zip(meta["ep_start"], meta["ep_len"])):
        s0, n, ep = int(s0), int(n), first + j
        df = pd.DataFrame({
            "observation.state": list(meta["state"][s0 : s0 + n].astype(np.float32)),
            "action": list(meta["action"][s0 : s0 + n].astype(np.float32)),
            "timestamp": np.arange(n) / fps,
            "task_index": np.zeros(n, np.int64),
            "episode_index": np.full(n, ep, np.int64),
            "frame_index": np.arange(n, dtype=np.int64),
            "index": np.arange(index, index + n, dtype=np.int64),
            "next.reward": np.zeros(n),
            "next.done": np.arange(n) == n - 1,
        })  # fmt: skip
        for c in cols:  # the original's extra annotation columns (GR00T conversion), zero-filled
            if c not in df:
                df[c] = np.zeros(n, np.int64)
        df[cols].to_parquet(f"{data_dir}/episode_{ep:06d}.parquet")
        for c in CAMS:
            buf, off = jpeg[c]
            frames = np.stack([cv2.imdecode(np.asarray(buf[off[t] : off[t + 1]]), cv2.IMREAD_COLOR) for t in range(s0, s0 + n)])
            encode(frames, f"{vid_dirs[c]}/episode_{ep:06d}.mp4", fps, a.ffmpeg)
        episodes.append({"episode_index": ep, "tasks": [task], "length": n})
        index += n
        if j % 25 == 0:
            print(f"{j + 1}/{len(meta['ep_len'])} collected episodes", flush=True)

    info.update(total_episodes=len(episodes), total_frames=index, total_tasks=1, total_videos=len(episodes) * len(CAMS),
                total_chunks=1, splits={"train": f"0:{len(episodes)}"}, codebase_version="v2.0")  # fmt: skip
    json.dump(info, open(f"{a.out}/meta/info.json", "w"), indent=4)
    with open(f"{a.out}/meta/episodes.jsonl", "w") as f:
        f.writelines(json.dumps(e) + "\n" for e in episodes)
    shutil.copy(f"{a.orig}/meta/stats.json", f"{a.out}/meta/stats.json")  # LeRobot v2.0 reads it; openpi computes its own
    with open(f"{a.out}/meta/tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": task}) + "\n")
    print(f"{a.out}: {len(episodes)} episodes ({len(orig_eps)} original + {len(episodes) - len(orig_eps)} collected), {index} frames")


if __name__ == "__main__":
    main()
