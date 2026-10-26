"""Prepare MagicSim MobileBench Go2Plus mobile-manipulation episodes (C2-05 pen -> holder) for training.

    python scripts/mobilebench/prep_magicsim_mm.py <out dir> <raw root> [<raw root> ...]

Each <raw root> is a run.py / collect_loop.py output dir; every lerobot_v21 dir below it whose
episodes have a completion receipt is used: <run>/lerobot_v21 (parquet + mp4) with
<run>/sim_info/<trajectory_id> (simulator states) and <run>/completed/<trajectory_id>.json.

Writes (labels: mm_labels.py):
  images_{front,wrist,head}.npy  uint8 [M, 224, 224, 3]  camera0 (Go2 front), camera1 (Piper wrist
                      D435), camera3 (head D435), resized with padding, every IMG_EVERY-th frame
  meta.npz            per frame (50 fps) of all episodes concatenated:
                      state [N, 20] (mm_labels.state_vector), base_cmd [N, 4], eef_cmd_pos [N, 3],
                      eef_cmd_rot6d [N, 6] (world), grip_cmd [N], pos [N, 3], yaw [N],
                      aff_point [N, 2, 3], aff_use_valid [N, 2], img_row [N] (-1 = no stored image)
                      ep_index, ep_start, ep_len, fps, img_every, prompt, source (run dir + episode)
  phase_labels.npz    frame_phase [N] into texts, texts, text_emb (frozen CLIP), group (pen held),
                      group_name
  assets/norm_stats.json  quantile stats of state [20] and actions [14] = upper chunk (EEF pos +
                      rot6d + gripper in the chunk-start body frame) + base command; the action
                      stats are over chunks sampled every IMG_EVERY frames
"""

import glob
import json
import os
import sys

import av
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import annotate_phases  # noqa: E402
import mm_labels as L  # noqa: E402

from openpi.shared import image_tools  # noqa: E402
from openpi.shared import normalize as _normalize  # noqa: E402

CAMS = {"front": "camera0", "wrist": "camera1", "head": "camera3"}
IMG_EVERY = 5  # images at 10 Hz; training offsets are multiples of this
HORIZON = 50
PROMPT = "Find the pen standing in the stand on the table and put it into the pen holder."


def find_episodes(roots: list[str]) -> list[tuple[str, int, int]]:
    """(run dir, lerobot episode index, trajectory id) of every completed episode, sorted."""
    out = []
    for root in roots:
        for lr in sorted(glob.glob(f"{root}/**/lerobot_v21", recursive=True)):
            run = os.path.dirname(lr)
            for line in open(f"{lr}/meta/episodes.jsonl"):
                e = json.loads(line)["episode_index"]
                tid = json.load(open(f"{lr}/meta/extra/episode_{e:06d}.json"))["trajectory_id"]
                if os.path.exists(f"{run}/completed/{tid}.json"):
                    out.append((run, e, tid))
    return out


def decode(path: str, keep: np.ndarray) -> np.ndarray:
    """Frames `keep` of a video, resized with padding to 224x224."""
    want, frames = set(keep.tolist()), {}
    with av.open(path) as con:
        for i, fr in enumerate(con.decode(video=0)):
            if i in want:
                frames[i] = fr.to_ndarray(format="rgb24")
    last = frames[max(frames)]
    x = np.stack([frames.get(i, last) for i in keep])  # hold the last frame if the video is short
    t = torch.from_numpy(x)
    return image_tools.resize_with_pad_torch(t, 224, 224).numpy().astype(np.uint8)


def main():
    out, roots = sys.argv[1], sys.argv[2:]
    os.makedirs(out, exist_ok=True)
    eps = find_episodes(roots)
    print(f"{len(eps)} completed episodes", flush=True)
    keys = ("state", "base_cmd", "eef_cmd_pos", "eef_cmd_rot6d", "grip_cmd", "pos", "yaw", "aff_point", "aff_use_valid", "img_row")
    cols = {k: [] for k in keys}
    texts, lens, sources, chunk_samples = [], [], [], []
    tmp = f"{out}/images_tmp"
    os.makedirs(tmp, exist_ok=True)
    n_img = 0
    for i, (run, e, tid) in enumerate(eps):
        try:
            ep = L.load(f"{run}/lerobot_v21", f"{run}/sim_info/{tid}", e)
            ph = L.phases(ep)
            pts, valid, ev = L.affordance(ep)
        except ValueError as err:
            print(f"skip {run} #{e}: {err}", flush=True)
            continue
        n = ep["n"]
        keep = np.arange(0, n, IMG_EVERY)
        for name, cam in CAMS.items():
            v = f"{run}/lerobot_v21/videos/chunk-000/observation.images.{cam}/episode_{e:06d}.mp4"
            np.save(f"{tmp}/{len(lens)}_{name}.npy", decode(v, keep))
        row = np.full(n, -1)
        row[keep] = n_img + np.arange(len(keep))
        n_img += len(keep)
        cols["state"].append(L.state_vector(ep))
        cols["base_cmd"].append(ep["base_cmd"])
        cols["eef_cmd_pos"].append(ep["eef_cmd_pos"].astype(np.float32))
        cols["eef_cmd_rot6d"].append(L.rot6d(ep["eef_cmd_rot"]).astype(np.float32))
        cols["grip_cmd"].append(ep["grip_cmd"])
        cols["pos"].append(ep["pos"].astype(np.float32))
        cols["yaw"].append(ep["yaw"].astype(np.float32))
        cols["aff_point"].append(pts)
        cols["aff_use_valid"].append(valid)
        cols["img_row"].append(row)
        for t in keep:
            chunk_samples.append(np.concatenate([L.upper_chunk(ep, int(t), HORIZON), ep["base_cmd"][np.clip(np.arange(t, t + HORIZON), 0, n - 1)]], 1))
        texts += [L.TEXT[p] for p in ph]
        lens.append(n)
        sources.append(f"{run}#{e}")
        print(f"episode {len(lens) - 1:3d} ({run} #{e}): {n} frames, grasp {ev[0]}, release {ev[1]}", flush=True)
    lens = np.array(lens)
    meta = {k: np.concatenate(v) for k, v in cols.items()}
    np.savez(
        f"{out}/meta.npz", **meta, ep_index=np.arange(len(lens)), ep_start=np.concatenate([[0], np.cumsum(lens)[:-1]]),
        ep_len=lens, fps=L.FPS, img_every=IMG_EVERY, prompt=PROMPT, source=np.array(sources),
    )  # fmt: skip
    for name in CAMS:
        img = np.lib.format.open_memmap(f"{out}/images_{name}.npy", "w+", np.uint8, (n_img, 224, 224, 3))
        s = 0
        for i in range(len(lens)):
            x = np.load(f"{tmp}/{i}_{name}.npy")
            img[s : s + len(x)] = x
            s += len(x)
            os.remove(f"{tmp}/{i}_{name}.npy")
        img.flush()
    os.rmdir(tmp)

    table = [L.TEXT[p] for p in L.PHASES]
    group = np.array([L.HOLDING[p] for p in L.PHASES])
    frame_phase = np.array([table.index(t) for t in texts])
    np.savez(
        f"{out}/phase_labels.npz", frame_phase=frame_phase, texts=np.array(table), text_emb=annotate_phases.embed(table),
        text_model=annotate_phases.TEXT_MODEL, group=group, group_name="holding",
    )  # fmt: skip
    acts = np.concatenate(chunk_samples).astype(np.float64)
    st = meta["state"].astype(np.float64)
    stats = {
        "state": _normalize.NormStats(mean=st.mean(0), std=st.std(0), q01=np.quantile(st, 0.01, 0), q99=np.quantile(st, 0.99, 0)),
        "actions": _normalize.NormStats(mean=acts.mean(0), std=acts.std(0), q01=np.quantile(acts, 0.01, 0), q99=np.quantile(acts, 0.99, 0)),
    }  # fmt: skip
    _normalize.save(f"{out}/assets", stats)
    counts = np.bincount(frame_phase, minlength=len(table))
    print(f"{len(lens)} episodes, {lens.sum()} frames, {n_img} image rows; phases:")
    for x, c in zip(table, counts):
        print(f"  {c:7d}  {x}")


if __name__ == "__main__":
    main()
