"""Labels for MagicSim MobileBench Go2Plus mobile-manipulation episodes (C2-05: pen -> pen holder).

Reads one recorded episode (run.py, formats lerobot_v21 + sim_info) and derives, per 50 fps frame:

  frame        gravity-aligned body frame: origin at the base position, x along the base heading
  base_cmd     commanded (vx, vy, yaw rate, body height): the recorded base action (body frame)
  upper_cmd    commanded EEF pose (world, from the action) + gripper (1 = closed)
  eef          measured EEF pose (world), gripper opening, arm joints
  events       t_grasp = first gripper close, t_release = the following open
  NAV / OBJ    NAV = base position at the next manipulation event (the stance the robot chose for
               the grasp, then for the release), invalid after the release; OBJ = the pen until
               it is grasped, then the pen holder
  phase        start, walk_to_pen, reach_pen, lift_pen, carry_pen, insert_pen, done, from the
               events and the scripted planner's stage (collect_info) -- logged state, not by hand

Text: one canonical sentence per phase that also states the remembered facts (pen held, pen
placed), so phases that look alike on camera differ in what the slow memory must carry.
"""

import json

import numpy as np
import pyarrow.parquet as pq

FPS = 50
PHASES = ("start", "walk_to_pen", "reach_pen", "lift_pen", "carry_pen", "insert_pen", "done")
TEXT = {
    "start": "Just started; the pen is still in its stand on the table and the gripper is empty.",
    "walk_to_pen": "Walking to the table with the pen; the gripper is empty and nothing is placed yet.",
    "reach_pen": "At the pen table, reaching the arm to the pen; the gripper is empty and nothing is placed yet.",
    "lift_pen": "Grasped the pen and lifting it; the pen is in the gripper, not placed yet.",
    "carry_pen": "Carrying the pen to the pen holder; the pen is in the gripper, not placed yet.",
    "insert_pen": "At the pen holder, putting the pen in; the pen is in the gripper, not placed yet.",
    "done": "The pen is in the pen holder and the gripper is empty; the task is done.",
}
HOLDING = {p: int(p in ("lift_pen", "carry_pen", "insert_pen")) for p in PHASES}
# observation.state / sim_info joint_pos order (lerobot observation.joint_vel names) -> Piper arm joints 1-6
ARM_Q = [4, 9, 14, 15, 16, 17]
PEN, HOLDER = "obj__object_pose.tgt0", "obj__object_pose.holder__floor"


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(w, x, y, z) [..., 4] -> [..., 3, 3]."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)  # fmt: skip


def yaw_of(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def yaw_mat(yaw) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    o, z = np.ones_like(c), np.zeros_like(c)
    return np.stack([np.stack([c, -s, z], -1), np.stack([s, c, z], -1), np.stack([z, z, o], -1)], -2)


def rot6d(r: np.ndarray) -> np.ndarray:
    """First two columns, [..., 3, 3] -> [..., 6]."""
    return np.concatenate([r[..., :, 0], r[..., :, 1]], -1)


def to_body(pos, yaw, p_world):
    """World point(s) -> body frame of (pos, yaw); broadcasts over leading dims."""
    return np.einsum("...ji,...j->...i", yaw_mat(yaw), np.asarray(p_world) - pos)


def load(lerobot_dir: str, sim_info_dir: str, episode_index: int) -> dict:
    """One episode: arrays aligned to the parquet frames (n), sim_info states trimmed to match."""
    d = pq.read_table(f"{lerobot_dir}/data/chunk-000/episode_{episode_index:06d}.parquet").to_pydict()
    s = np.load(f"{sim_info_dir}/sim_states.npz", allow_pickle=True)
    n = len(d["frame_index"])
    if len(s["robot__Go2Plus__base_pos"]) < n:
        raise ValueError(f"{sim_info_dir}: {len(s['robot__Go2Plus__base_pos'])} sim states < {n} frames")
    act = np.array(d["action"], np.float32)
    state = np.array(d["observation.state"], np.float32)
    base_q = s["robot__Go2Plus__base_quat"][:n]
    stages = []
    for c in d["collect_info"]:
        g = (json.loads(c) if c else {}).get("global_planner_info") or {}
        stages.append(g.get("type") or "")
    ep = {
        "n": n,
        "pos": s["robot__Go2Plus__base_pos"][:n].astype(np.float64),
        "yaw": yaw_of(base_q),
        "eef_pos": s["robot__Go2Plus__eef_pos"][:n].astype(np.float64),
        "eef_rot": quat_to_mat(s["robot__Go2Plus__eef_quat"][:n].astype(np.float64)),
        "arm_q": s["robot__Go2Plus__joint_pos"][:n][:, ARM_Q].astype(np.float32),
        "grip": state[:, 24:26].mean(1),
        "base_cmd": act[:, :4],
        "eef_cmd_pos": act[:, 4:7].astype(np.float64),
        "eef_cmd_rot": quat_to_mat(act[:, 7:11].astype(np.float64)),
        "grip_cmd": act[:, 11],
        "pen": s[PEN][:n, :3].astype(np.float64),
        "holder": s[HOLDER][:n, :3].astype(np.float64),
        "stage": np.array(stages),
        "prompt": None,
    }
    return ep


def events(ep: dict) -> tuple[int, int]:
    g = ep["grip_cmd"] > 0.5
    close = np.flatnonzero(g)
    if not len(close):
        raise ValueError("the gripper never closes")
    t_grasp = int(close[0])
    opened = np.flatnonzero(~g[t_grasp:])
    if not len(opened):
        raise ValueError("the gripper never opens after the grasp")
    return t_grasp, t_grasp + int(opened[0])


def phases(ep: dict) -> np.ndarray:
    n, (t_grasp, t_release) = ep["n"], events(ep)
    nav = ep["stage"] == "NavTo"
    nav_after = np.flatnonzero(nav[t_grasp:t_release]) + t_grasp
    if not len(nav_after):
        raise ValueError("no NavTo between the grasp and the release")
    carry0, carry1 = int(nav_after[0]), int(nav_after[-1]) + 1
    out = np.empty(n, object)
    for t in range(n):
        if t < t_grasp:
            out[t] = "walk_to_pen" if nav[t] or not nav[:t].any() else "reach_pen"
        elif t < carry0:
            out[t] = "lift_pen"
        elif t < carry1:
            out[t] = "carry_pen"
        elif t < t_release:
            out[t] = "insert_pen"
        else:
            out[t] = "done"
    out[:FPS] = "start"
    return out


def affordance(ep: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NAV / OBJ points in the body frame [n, 2, 3] and use-validity [n, 2]."""
    n, (t_grasp, t_release) = ep["n"], events(ep)
    t = np.arange(n)
    nav_world = np.where((t < t_grasp)[:, None], ep["pos"][t_grasp], ep["pos"][t_release])
    nav_world[:, 2] = ep["pos"][:, 2]  # a stop point on the floor plane of the base
    obj_world = np.where((t < t_grasp)[:, None], ep["pen"][np.minimum(t, t_grasp)], ep["holder"])
    pts = np.stack([to_body(ep["pos"], ep["yaw"], nav_world), to_body(ep["pos"], ep["yaw"], obj_world)], 1)
    valid = np.stack([t < t_release, np.ones(n, bool)], 1)
    return pts.astype(np.float32), valid, np.array([t_grasp, t_release])


def body_velocity(ep: dict) -> np.ndarray:
    """Measured body-frame (vx, vy, yaw rate) from the base pose, central differences."""
    v = np.gradient(ep["pos"][:, :2], axis=0) * FPS
    w = np.gradient(np.unwrap(ep["yaw"])) * FPS
    c, s = np.cos(ep["yaw"]), np.sin(ep["yaw"])
    return np.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1], w], 1).astype(np.float32)


def eef_body(ep: dict, pos_w, rot_w, t: int | np.ndarray):
    """World EEF pose(s) -> body frame of the base at frame t: (pos [..., 3], rot [..., 3, 3])."""
    rt = yaw_mat(ep["yaw"][t])
    return to_body(ep["pos"][t], ep["yaw"][t], pos_w), np.swapaxes(rt, -1, -2) @ rot_w


def state_vector(ep: dict) -> np.ndarray:
    """Raw proprio state [n, 20]: EEF pos (3) + rot6d (6) in the body frame, gripper opening (1),
    arm joints (6), measured body (vx, vy, yaw rate) (3), base height (1)."""
    t = np.arange(ep["n"])
    p, r = eef_body(ep, ep["eef_pos"], ep["eef_rot"], t)
    return np.concatenate([p, rot6d(r), ep["grip"][:, None], ep["arm_q"], body_velocity(ep), ep["pos"][:, 2:3]], 1).astype(np.float32)


def upper_chunk(ep: dict, t: int, horizon: int) -> np.ndarray:
    """Commanded EEF pose + gripper of frames t..t+H-1 in the body frame of frame t: [H, 10]."""
    k = np.clip(np.arange(t, t + horizon), 0, ep["n"] - 1)
    p, r = eef_body(ep, ep["eef_cmd_pos"][k], ep["eef_cmd_rot"][k], np.full(len(k), t))
    return np.concatenate([p, rot6d(r), ep["grip_cmd"][k, None]], 1).astype(np.float32)
