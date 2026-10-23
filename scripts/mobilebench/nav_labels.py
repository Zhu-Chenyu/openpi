# Vendored unchanged from Memory-Augmented-VLA scripts/magicsim/nav_labels.py @ 670c753 (prep_magicsim_nav.py uses it).
"""Numeric MobileBench labels for MagicSim go2 navigation episodes (onboard camera).

Computed from logged simulator state; nothing is hand-written. Per policy update t
(every `stride` frames; 50 frames = 1 s):

  nav_point_body      [3]   the NavTo goal point (where the base must stop) in the CURRENT body frame
  obj_point_body      [3]   the target object's centre (bbox centre) in the current body frame
  obs_valid           [2]   (nav, obj) knowable from the CURRENT onboard image: the target's centre or
                            upper part is inside the camera frustum and the camera->target ray is not
                            blocked by any other furniture box (the nav point is defined by the target,
                            so it shares the target's visibility)
  use_valid           [2]   (nav, obj) known to the task at all -- always True here (the instruction
                            names the target and it was seen or is reachable by memory)
  base_actions        [H,3] commanded (vx, vy, yaw_rate) for frames t..t+H-1 (clamped at the end)
  exec_increment      [4]   executed body motion since the previous update, in the previous body
                            frame: (dx, dy, dyaw, dt); zeros + dt=0 at the first update
  trace_body          [L,3] body pose (x, y, yaw) 1..L updates ago in the current body frame, with
  trace_avail         [L]   availability (no placeholder pasts)
  mode                int   0 = NAV
  target_pixel        [2]   projected target centre in the onboard image (for QA overlays), NaN if behind

The target is the scene object whose category is `target` (the NavTo obj_name: small_cube for the
red-cube runs, fridge for the furniture runs). Every other scene object is an occluder box.

Usage: python nav_labels.py <episode_dir> <out.npz> [--target fridge] [--stride 50] [--horizon 50] [--lags 4]
"""

import argparse
import json
import math

import numpy as np

# Onboard camera mount (go2_room_nav_chenyu.yaml, camera0): base-frame offset and rotation.
MOUNT_POS = np.array([0.32, 0.0, 0.06])
W, H = 320, 240
FX = 18.0 / 36.0 * W
FY = 18.0 / 27.0 * H
CX, CY = W / 2, H / 2
# Config yaw (deg) of each object category; scene.json stores only the local bbox, not the pose's yaw.
CATEGORY_YAW = {"fridge": -90.0, "cabinet": -90.0, "side_table": 90.0}


def euler_xyz_deg(a, b, c):
    """INTRINSIC XYZ Euler (degrees) -> matrix, R = Rx(a) Ry(b) Rz(c) -- MagicSim's camera `ori`."""
    a, b, c = np.radians([a, b, c])
    rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return rx @ ry @ rz


MOUNT_ROT = euler_xyz_deg(0.0, -78.0, -90.0)  # USD camera: looks along -Z, +Y up


def quat_to_mat(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def segment_hits_box(p0, p1, lo, hi) -> bool:
    """Slab test: does the segment p0->p1 pass through the axis-aligned box [lo, hi]?"""
    d = p1 - p0
    t0, t1 = 0.0, 1.0
    for k in range(3):
        if abs(d[k]) < 1e-9:
            if p0[k] < lo[k] or p0[k] > hi[k]:
                return False
            continue
        a, b = (lo[k] - p0[k]) / d[k], (hi[k] - p0[k]) / d[k]
        t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
        if t0 > t1:
            return False
    return True


def scene_to_world(rel, base_pos0, base_quat0):
    """scene.json object positions are in the robot's frame at frame 0 (its ACTUAL pose,
    yaw included; scene["robots"][0]["base_pose"] is stale). Verified against the cube's
    logged world pose to ~2 cm."""
    w, x, y, z = base_quat0
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = math.cos(yaw), math.sin(yaw)
    rel = np.asarray(rel[:2], float)
    return np.asarray(base_pos0[:2], float) + np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]])


def object_box(o: dict, base_pos0, base_quat0):
    """World axis-aligned box (lo, hi) of a scene.json object. Its pos is in the frame-0 robot frame
    (scene_to_world); z is relative to the base, whose roll/pitch at frame 0 is ~0."""
    w, x, y, z = base_quat0
    robot_yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    yaw = robot_yaw + math.radians(CATEGORY_YAW.get(o["cat_name"], 0.0))  # world yaw of the object
    c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
    ex, ey, ez = o["bbox"]["extent"]
    cx, cy, cz = o["bbox"]["center"]
    rc = np.array([math.cos(yaw) * cx - math.sin(yaw) * cy, math.sin(yaw) * cx + math.cos(yaw) * cy])
    xy = scene_to_world(o["pos"], base_pos0, base_quat0) + rc
    half = np.array([c * ex + s * ey, s * ex + c * ey]) / 2  # AABB of the yawed box
    zc = base_pos0[2] + o["pos"][2] + cz
    return (np.array([xy[0] - half[0], xy[1] - half[1], zc - ez / 2]),
            np.array([xy[0] + half[0], xy[1] + half[1], zc + ez / 2]))  # fmt: skip


def load(ep_dir: str, target: str = "fridge") -> dict:
    info = json.load(open(f"{ep_dir}/env/info/env_0_merged.json"))
    act = json.load(open(f"{ep_dir}/action/action_merged.json"))
    col = json.load(open(f"{ep_dir}/collect/collect_merged.json"))
    scene = json.load(open(f"{ep_dir}/language/scene.json"))
    frames = sorted(info)
    pos, quat, base_cmd = [], [], []
    for f in frames:
        rs = info[f]["obs"]["policy_obs"]["robot_state"][0]["Robot1"]
        pos.append(rs["base_pos"])
        quat.append(rs["base_quat"])
        a = act.get(f, {}).get("raw_action", {}).get("Robot1", {}).get("base_action")
        base_cmd.append([a[0], a[1], a[3]] if a is not None else [np.nan] * 3)  # vx, vy, yaw_rate
    base_cmd = np.array(base_cmd, float)
    # Fill frames without a fresh command (planner pending) with the last command held.
    for t in range(len(base_cmd)):
        if np.isnan(base_cmd[t]).any():
            base_cmd[t] = base_cmd[t - 1] if t else 0.0
    nav_goal = np.array(col[frames[0]]["global_planner_info"]["command"][2][:2])
    boxes = {o["cat_name"]: object_box(o, pos[0], quat[0]) for o in scene["objects"]}
    if target not in boxes:
        raise KeyError(f"{ep_dir}: no '{target}' in scene.json ({sorted(boxes)})")
    lo, hi = boxes[target]
    occ = {k: v for k, v in boxes.items() if k != target}
    return {
        "pos": np.array(pos),
        "quat": np.array(quat),
        "yaw": np.array([math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)) for q in quat]),
        "base_cmd": base_cmd,
        "target": (lo + hi) / 2,
        "target_top": np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2] + 0.85 * (hi[2] - lo[2])]),
        "target_name": target,
        "nav_goal": np.array([nav_goal[0], nav_goal[1], 0.0]),
        "occluder_names": np.array(sorted(occ)),
        "occluder_lo": np.array([occ[k][0] for k in sorted(occ)]).reshape(-1, 3),
        "occluder_hi": np.array([occ[k][1] for k in sorted(occ)]).reshape(-1, 3),
    }


def to_body(ep, t, p_world):
    r = quat_to_mat(ep["quat"][t])
    return r.T @ (p_world - ep["pos"][t])


def project(ep, t, p):
    """Pixel (u, v) of world point p in the onboard image at frame t, or None if behind the camera."""
    r_base = quat_to_mat(ep["quat"][t])
    cam_pos = ep["pos"][t] + r_base @ MOUNT_POS
    pc = (r_base @ MOUNT_ROT).T @ (p - cam_pos)
    depth = -pc[2]
    if depth <= 0.05:
        return cam_pos, None
    return cam_pos, np.array([CX + FX * pc[0] / depth, CY - FY * pc[1] / depth])


def target_visibility(ep, t):
    """Visible if the target's centre or upper part is in view and its ray clears every occluder box."""
    vis = False
    for p in (ep["target"], ep["target_top"]):
        cam_pos, uv = project(ep, t, p)
        if uv is None or not (0 <= uv[0] < W and 0 <= uv[1] < H):
            continue
        if not any(segment_hits_box(cam_pos, p, lo, hi) for lo, hi in zip(ep["occluder_lo"], ep["occluder_hi"])):
            vis = True
    _, uv = project(ep, t, ep["target"])
    return vis, (uv if uv is not None else np.array([np.nan, np.nan]))


def labels(ep: dict, stride: int, horizon: int, lags: int) -> dict:
    n = len(ep["pos"])
    ups = np.arange(0, n, stride)
    out = {k: [] for k in ("frame", "nav_point_body", "obj_point_body", "obs_valid", "use_valid", "base_actions",
                           "exec_increment", "trace_body", "trace_avail", "mode", "target_pixel")}  # fmt: skip
    for i, t in enumerate(ups):
        vis, px = target_visibility(ep, t)
        out["frame"].append(t)
        out["nav_point_body"].append(to_body(ep, t, ep["nav_goal"] + np.array([0, 0, ep["pos"][t][2]])))
        out["obj_point_body"].append(to_body(ep, t, ep["target"]))
        out["obs_valid"].append([vis, vis])
        out["use_valid"].append([True, True])
        idx = np.clip(np.arange(t, t + horizon), 0, n - 1)
        out["base_actions"].append(ep["base_cmd"][idx])
        if i == 0:
            out["exec_increment"].append([0.0, 0.0, 0.0, 0.0])
        else:
            tp = ups[i - 1]
            d = to_body(ep, tp, ep["pos"][t])
            out["exec_increment"].append([d[0], d[1], wrap(ep["yaw"][t] - ep["yaw"][tp]), (t - tp) / 50.0])
        tr, av = np.zeros((lags, 3)), np.zeros(lags, bool)
        for k in range(lags):
            j = i - (k + 1)
            if j < 0:
                continue
            tpast = ups[j]
            d = to_body(ep, t, ep["pos"][tpast])
            tr[k] = [d[0], d[1], wrap(ep["yaw"][tpast] - ep["yaw"][t])]
            av[k] = True
        out["trace_body"].append(tr)
        out["trace_avail"].append(av)
        out["mode"].append(0)
        out["target_pixel"].append(px)
    return {k: np.asarray(v) for k, v in out.items()}


def save(path: str, ep: dict, lab: dict):
    np.savez(path, **lab, target_world=ep["target"], target_name=ep["target_name"], nav_goal_world=ep["nav_goal"],
             occluder_names=ep["occluder_names"], occluder_lo=ep["occluder_lo"], occluder_hi=ep["occluder_hi"],
             base_pos=ep["pos"], base_yaw=ep["yaw"], base_cmd=ep["base_cmd"])  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode_dir")
    ap.add_argument("out")
    ap.add_argument("--target", default="fridge", help="scene category of the NavTo target")
    ap.add_argument("--stride", type=int, default=50)
    ap.add_argument("--horizon", type=int, default=50)
    ap.add_argument("--lags", type=int, default=4)
    args = ap.parse_args()
    ep = load(args.episode_dir, args.target)
    lab = labels(ep, args.stride, args.horizon, args.lags)
    save(args.out, ep, lab)
    vis = lab["obs_valid"][:, 1]
    print(f"{args.episode_dir}: {len(lab['frame'])} updates, {args.target} visible in {vis.sum()}/{len(vis)}, "
          f"final dist {np.linalg.norm(lab['nav_point_body'][-1][:2]):.2f} m to nav goal")  # fmt: skip


if __name__ == "__main__":
    main()
