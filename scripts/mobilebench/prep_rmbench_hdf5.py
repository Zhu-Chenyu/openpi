"""Flatten freshly collected RMBench expert demos (script/collect_data.py hdf5) into the array
format of prep_rmbench.py, so they can be trained on next to an existing prepped dataset.

Matches the GR00T LeRobot conversions the existing data came from: state = action =
joint_action/vector (the commanded joint vector the eval client also sends as state; those
conversions store action[t] == state[t], so chunks are state[t : t+50]), cameras head and
front at the collection resolution (LargeView, 240x320), fps 25.

Outputs in <out>: images_head.npy, images_front.npy (uint8 memmaps) and meta.npz with the
single-task keys of prep_rmbench.py (prompt instead of prompts / ep_prompt / ep_task, plus task).
With --jpeg the collector's JPEG frames are copied as they are (jpeg_<cam>.bin + jpeg_<cam>_off.npy,
see rmbench_episodes.EpisodeSequences) instead of decoded into memmaps: ~15x smaller.

Usage: python prep_rmbench_hdf5.py <out_dir> <collect_dir> [<collect_dir> ...] \
           [--prompt "..."] [--task put_back_block] [--jpeg] [--procs 16]
       (collect_dir = <save_path>/<task>/<task_config>, containing data/episode*.hdf5)
"""

import argparse
import glob
import multiprocessing as mp
import os
import re
import sys

import cv2
import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aloha_fk  # noqa: E402
from prep_rmbench import gripper_events  # noqa: E402

CAMS = {"head": "observation/head_camera/rgb", "front": "observation/front_camera/rgb"}
PUT_BACK_PROMPT = "Put back the block that was removed from the shelf to its original position."


def _ep_files(d):
    files = glob.glob(f"{d}/data/episode*.hdf5")
    return sorted(files, key=lambda p: int(re.search(r"episode(\d+)\.hdf5$", p).group(1)))


def _length(path):
    with h5py.File(path, "r") as f:
        return len(f["joint_action/vector"])


def _load(path):
    with h5py.File(path, "r") as f:
        vec = f["joint_action/vector"][:].astype(np.float32)
        # The collector JPEG-encodes the renderer's RGB frames with cv2, so imdecode gives RGB back.
        imgs = {c: np.stack([cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) for b in f[k][:]])
                for c, k in CAMS.items()}  # fmt: skip
    return vec, imgs


def _load_jpeg(path):
    with h5py.File(path, "r") as f:
        return f["joint_action/vector"][:].astype(np.float32), {c: [bytes(b) for b in f[k][:]] for c, k in CAMS.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("collect_dirs", nargs="+")
    ap.add_argument("--prompt", default=PUT_BACK_PROMPT)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--task", default="put_back_block")
    ap.add_argument("--jpeg", action="store_true", help="keep the collector's JPEG frames instead of memmaps")
    args = ap.parse_args()
    files = [p for d in args.collect_dirs for p in _ep_files(d)]
    with mp.Pool(args.procs) as pool:
        lengths = pool.map(_length, files)
    total = sum(lengths)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(int)
    with h5py.File(files[0], "r") as f:
        h, w = cv2.imdecode(np.frombuffer(f[CAMS["head"]][0], np.uint8), cv2.IMREAD_COLOR).shape[:2]
    os.makedirs(args.out, exist_ok=True)
    if args.jpeg:
        bins = {c: open(f"{args.out}/jpeg_{c}.bin", "wb") for c in CAMS}
        offs = {c: [0] for c in CAMS}
    else:
        mm = {c: np.lib.format.open_memmap(f"{args.out}/images_{c}.npy", "w+", np.uint8, (total, h, w, 3)) for c in CAMS}
    state = np.zeros((total, 14), np.float32)
    events = []
    with mp.Pool(args.procs) as pool:
        for i, (vec, imgs) in enumerate(pool.imap(_load_jpeg if args.jpeg else _load, files, chunksize=2)):
            s, n = starts[i], lengths[i]
            state[s : s + n] = vec
            for c, im in imgs.items():
                assert len(im) == n, (files[i], c, len(im))
                if args.jpeg:
                    for b in im:
                        bins[c].write(b)
                        offs[c].append(offs[c][-1] + len(b))
                else:
                    assert im.shape == (n, h, w, 3), (files[i], c, im.shape)
                    mm[c][s : s + n] = im
            events += [(i, *e) for e in gripper_events(vec)]
            if i % 50 == 0:
                print(f"{i + 1}/{len(files)} episodes", flush=True)
    if args.jpeg:
        for c in CAMS:
            bins[c].close()
            np.save(f"{args.out}/jpeg_{c}_off.npy", np.array(offs[c], np.int64))
    else:
        for v in mm.values():
            v.flush()
    tcp_pos, tcp_rot = aloha_fk.both_tcp(state)
    np.savez(
        f"{args.out}/meta.npz",
        state=state,
        action=state.copy(),
        ep_index=np.arange(len(files)),
        ep_start=starts,
        ep_len=np.array(lengths),
        tcp_pos=tcp_pos.astype(np.float32),
        tcp_rot=tcp_rot.astype(np.float32),
        events=np.array(sorted(events), dtype=np.int64),
        prompt=np.array(args.prompt),
        task=np.array(args.task),
        fps=np.array(25),
        source_files=np.array(files),
    )
    print(f"wrote {total} frames / {len(files)} episodes (mean {total / len(files):.0f} frames) to {args.out}")


if __name__ == "__main__":
    main()
