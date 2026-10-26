"""Prepare MagicSim go2 navigation episodes (onboard camera, NavTo the fridge) for MobileBench training.

    python scripts/mobilebench/prep_magicsim_nav.py <raw root> <sources.json> <out dir>

<raw root>/<run>/navto/000/magicsim/<run_episode>/ are MagicSim recordings (env/info, action,
collect, language, env/camera/cam_0/rgb/rgb.mp4); sources.json (from the labelled release,
e.g. magicsim_nav_furniture_v1/sources.json) lists them in episode order.

Writes, per frame (50 fps) of all episodes concatenated:
  images_onboard.npy  uint8 [N, 240, 320, 3]  onboard camera (the policy's only image)
  meta.npz            state [N, 3]   measured body-frame (vx, vy, yaw rate)
                      action [N, 3]  commanded (vx, vy, yaw rate) -- the base action
                      pos [N, 3], yaw [N]  base pose (odometry for the executed-motion input)
                      nav_point [N, 3], obj_point [N, 3]  NavTo stop point / fridge centre, body frame
                      obj_visible [N]  target in view and not occluded (nav_labels.target_visibility)
                      ep_index, ep_start, ep_len, fps, prompt, base_height
  phase_labels.npz    slow-memory text targets in the annotate_phases.py format: frame_phase [N]
                      (index into texts), texts, text_emb (frozen CLIP), plus group / group_name
                      (the remembered detour side, for an accuracy log)
  assets/norm_stats.json  quantile stats of state and action (openpi format)

The text facts come from nav_phase_text.annotate evaluated at EVERY frame (training starts each
episode at a random offset); the first second is "start" and the last second, if within
ARRIVE_M of the stop point, "arrived" -- the same rule annotate applies to the first / last update.
"""

import json
import os
import sys

import av
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import annotate_phases  # noqa: E402
import nav_labels  # noqa: E402
import nav_phase_text  # noqa: E402

from openpi.shared import normalize as _normalize  # noqa: E402

FPS = 50
STRIDE = 50  # one policy update per second


def body_velocity(ep_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Measured body-frame (vx, vy, yaw rate) and the commanded base height per frame."""
    info = json.load(open(f"{ep_dir}/env/info/env_0_merged.json"))
    act = json.load(open(f"{ep_dir}/action/action_merged.json"))
    vel, height = [], []
    for f in sorted(info):
        rs = info[f]["obs"]["policy_obs"]["robot_state"][0]["Robot1"]
        vel.append([rs["base_lin_vel"][0], rs["base_lin_vel"][1], rs["base_ang_vel"][2]])
        a = act.get(f, {}).get("raw_action", {}).get("Robot1", {}).get("base_action")
        height.append(a[2] if a is not None else np.nan)
    return np.array(vel, np.float32), np.array(height)


def frame_texts(tep: dict, n: int) -> tuple[list[str], list[str | None]]:
    """Canonical slow-memory sentence and remembered detour side of every frame."""
    facts = [u["facts"] for u in nav_phase_text.annotate(tep, 1, 1)]
    dist = np.linalg.norm(tep["pos"] - tep["goal"], axis=1)
    texts, sides = [], []
    for t, f in enumerate(facts):
        phase = f["phase"]
        if t < STRIDE:
            phase = "start"
        elif t >= n - STRIDE and dist[t] < nav_phase_text.ARRIVE_M:
            phase = "arrived"
        f = {**f, "phase": phase, "completed_events": nav_phase_text.completed_events(phase, int(t >= STRIDE))}
        if phase in ("passing", "cleared", "final_approach", "arrived") and f["detour_side"] is None:
            raise ValueError(f"frame {t}: phase {phase} without a committed detour side")
        texts.append(nav_phase_text.render(f)[0])
        sides.append(f["detour_side"])
    return texts, sides


def decode(path: str) -> np.ndarray:
    with av.open(path) as con:
        return np.stack([fr.to_ndarray(format="rgb24") for fr in con.decode(video=0)])


def main():
    raw, sources, out = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    eps = json.load(open(sources))["episodes"]
    cols = {k: [] for k in ("state", "action", "pos", "yaw", "nav_point", "obj_point", "obj_visible", "height")}
    texts, lens = [], []
    tmp = f"{out}/images_tmp"  # per-episode frames, joined at the end (all of them do not fit in RAM)
    os.makedirs(tmp, exist_ok=True)
    for e in eps:
        d = f"{raw}/{e['run']}/navto/000/magicsim/{e['run_episode']}"
        ep = nav_labels.load(d, "fridge")
        tep = nav_phase_text.load_episode(d, "fridge")
        vel, height = body_velocity(d)
        frames = decode(f"{d}/env/camera/cam_0/rgb/rgb.mp4")
        n = len(ep["pos"])
        if len(frames) < n:  # hold the last frame if the video is a frame short
            frames = np.concatenate([frames, np.repeat(frames[-1:], n - len(frames), 0)])
        np.save(f"{tmp}/{len(lens)}.npy", frames[:n])
        goal = ep["nav_goal"].copy()
        cols["state"].append(vel)
        cols["action"].append(ep["base_cmd"].astype(np.float32))
        cols["pos"].append(ep["pos"])
        cols["yaw"].append(ep["yaw"])
        cols["height"].append(height)
        nav, obj, vis = [], [], []
        for t in range(n):
            goal[2] = ep["pos"][t][2]
            nav.append(nav_labels.to_body(ep, t, goal))
            obj.append(nav_labels.to_body(ep, t, ep["target"]))
            vis.append(nav_labels.target_visibility(ep, t)[0])
        cols["nav_point"].append(np.array(nav, np.float32))
        cols["obj_point"].append(np.array(obj, np.float32))
        cols["obj_visible"].append(np.array(vis))
        t_ep, s_ep = frame_texts(tep, n)
        texts += t_ep
        lens.append(n)
        print(f"episode {e['episode']:3d} ({e['run']} #{e['run_episode']}): {n} frames, "
              f"visible {np.mean(vis):.2f}, side {s_ep[-1]}", flush=True)  # fmt: skip
    meta = {k: np.concatenate(v) for k, v in cols.items()}
    height = meta.pop("height")
    h = height[~np.isnan(height)]
    if np.ptp(h) > 1e-6:
        raise ValueError(f"base height command is not constant: {h.min()}..{h.max()}")
    lens = np.array(lens)
    np.savez(
        f"{out}/meta.npz", **meta, ep_index=np.arange(len(eps)), ep_start=np.concatenate([[0], np.cumsum(lens)[:-1]]),
        ep_len=lens, fps=FPS, prompt="Navigate to the fridge.", base_height=np.float32(h[0]),
    )  # fmt: skip
    shape = np.load(f"{tmp}/0.npy", mmap_mode="r").shape[1:]
    img = np.lib.format.open_memmap(f"{out}/images_onboard.npy", "w+", np.uint8, (int(lens.sum()), *shape))
    s = 0
    for i in range(len(lens)):
        img[s : s + lens[i]] = np.load(f"{tmp}/{i}.npy")
        s += lens[i]
        os.remove(f"{tmp}/{i}.npy")
    img.flush()
    os.rmdir(tmp)

    table = sorted(set(texts))
    # Group = the detour side a sentence states (0 none, 1 left, 2 right): logged as slow_side_acc.
    group = np.array([1 if " left" in x else 2 if " right" in x else 0 for x in table])
    np.savez(
        f"{out}/phase_labels.npz", frame_phase=np.array([table.index(t) for t in texts]), texts=np.array(table),
        text_emb=annotate_phases.embed(table), text_model=annotate_phases.TEXT_MODEL, group=group,
        group_name="side",
    )  # fmt: skip
    stats = {}  # keyed like the transformed sample ("state", "actions")
    for k, key in (("state", "state"), ("action", "actions")):
        x = meta[k].astype(np.float64)
        stats[key] = _normalize.NormStats(mean=x.mean(0), std=x.std(0), q01=np.quantile(x, 0.01, 0), q99=np.quantile(x, 0.99, 0))
    _normalize.save(f"{out}/assets", stats)
    counts = np.bincount([table.index(t) for t in texts], minlength=len(table))
    print(f"{len(eps)} episodes, {lens.sum()} frames; {len(table)} slow-memory sentences:")
    for x, c in zip(table, counts):
        print(f"  {c:6d}  {x}")


if __name__ == "__main__":
    main()
