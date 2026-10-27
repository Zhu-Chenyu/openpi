"""One task of a multi-task RMBench prep (prep_rmbench.py --video, e.g. mobilebench_data/ten_task) as a
single-task dir with JPEG frames, the layout prep_rmbench_hdf5.py --jpeg writes for collected demos.

The episodes' mp4s (meta video_root) are decoded and re-encoded with cv2 the way the collector stores
its frames (cv2.imencode on the RGB array, so cv2.imdecode gives RGB back), quality 95.

Usage: python prep_rmbench_task.py <multi-task prep dir> <task> <out_dir> [--procs 16]
"""

import argparse
import multiprocessing as mp
import os

import av
import cv2
import numpy as np

CAMS = ("head", "front")


def _encode(args):
    video_root, ep, n = args
    out = {}
    for c in CAMS:
        path = f"{video_root}/videos/chunk-{ep // 1000:03d}/observation.images.{c}/episode_{ep:06d}.mp4"
        with av.open(path) as con:
            frames = [fr.to_ndarray(format="rgb24") for fr in con.decode(video=0)]
        frames = frames[:n] + [frames[-1]] * max(0, n - len(frames))  # hold the last frame if the video is short
        out[c] = [cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes() for f in frames]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    m = dict(np.load(f"{a.src}/meta.npz"))
    sel = np.flatnonzero(m["ep_task"] == a.task)
    if not len(sel):
        raise ValueError(f"no episodes of {a.task} in {a.src}")
    prompts = {str(m["prompts"][m["ep_prompt"][i]]) for i in sel}
    if len(prompts) != 1:
        raise ValueError(f"{a.task}: {len(prompts)} prompts")
    lens = m["ep_len"][sel]
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    rows = np.concatenate([np.arange(m["ep_start"][i], m["ep_start"][i] + m["ep_len"][i]) for i in sel])
    old_to_new = {int(m["ep_index"][i]): k for k, i in enumerate(sel)}
    events = np.array([(old_to_new[e], f, arm, c) for e, f, arm, c in m["events"] if int(e) in old_to_new], np.int64)
    os.makedirs(a.out, exist_ok=True)
    bins = {c: open(f"{a.out}/jpeg_{c}.bin", "wb") for c in CAMS}
    offs = {c: [0] for c in CAMS}
    jobs = [(str(m["video_root"]), int(m["ep_index"][i]), int(m["ep_len"][i])) for i in sel]
    with mp.Pool(a.procs) as pool:
        for k, enc in enumerate(pool.imap(_encode, jobs)):
            for c in CAMS:
                for b in enc[c]:
                    bins[c].write(b)
                    offs[c].append(offs[c][-1] + len(b))
            if k % 50 == 0:
                print(f"{k + 1}/{len(jobs)} episodes", flush=True)
    for c in CAMS:
        bins[c].close()
        np.save(f"{a.out}/jpeg_{c}_off.npy", np.array(offs[c], np.int64))
    np.savez(
        f"{a.out}/meta.npz", state=m["state"][rows], action=m["action"][rows], ep_index=np.arange(len(sel)),
        ep_start=starts, ep_len=lens, tcp_pos=m["tcp_pos"][rows], tcp_rot=m["tcp_rot"][rows], events=events,
        prompt=np.array(prompts.pop()), task=np.array(a.task), fps=m["fps"], source_episodes=m["ep_index"][sel],
    )  # fmt: skip
    print(f"{a.task}: {len(sel)} episodes, {len(rows)} frames -> {a.out}")


if __name__ == "__main__":
    main()
