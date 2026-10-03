"""Episode-sequence dataset for MobileBench pi0.5 on RMBench (arrays from prep_rmbench.py).

One item = one whole episode as a sequence of policy updates every `stride` frames
(stride = the executed chunk length at evaluation, so the memory sees the same cadence
it will see in closed loop), starting at a random offset in [0, stride). Each update
carries the openpi-transformed observation and action chunk (identical transforms to the
pi0.5 LoRA baseline), the conditioner's online inputs, and labels that only reach losses.

Labels from the gripper events (frame, arm, closed) -- every episode of put_back_block is
R-close, R-open, L-close, R-close, R-open:
  affordance [NAV, OBJ]  OBJ = TCP position of the arm at the NEXT event (any arm).
                         NAV is labelled invalid (fixed base, no navigation target).
                         Current-observation validity is False when the next event is the
                         final right release: returning the block to its ORIGINAL mat
                         cannot be read off the current image (the mats look alike), only
                         recalled; the joint branch is still supervised there.
  goal [L, R]            TCP pose of that arm at its own next event; invalid after its last.
  trace                  TCP pose + gripper of both arms 1..L updates ago (fast group only).
  mode                   MANIP throughout.
All poses are in the robot footprint frame (aloha_fk), which is fixed for this robot.
"""

import dataclasses
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aloha_fk  # noqa: E402

GRIP_DIMS = (6, 13)
ARM_Q = (slice(0, 6), slice(7, 13))
NUM_EE = 2
QDOT_MAX = 1.0  # rad/s, coarse joint speed for the library cost c_i


def rotvec(r: np.ndarray) -> np.ndarray:
    """SO(3) log map, [..., 3, 3] -> [..., 3]."""
    cos = np.clip((np.trace(r, axis1=-2, axis2=-1) - 1) / 2, -1.0, 1.0)
    ang = np.arccos(cos)
    v = np.stack([r[..., 2, 1] - r[..., 1, 2], r[..., 0, 2] - r[..., 2, 0], r[..., 1, 0] - r[..., 0, 1]], -1)
    s = np.sin(ang)
    scale = np.where(s > 1e-6, ang / (2 * np.maximum(s, 1e-6)), 0.5)
    return v * scale[..., None]


@dataclasses.dataclass
class Library:
    """Fixed reachable-pose library P_r per arm, sampled from demonstrated configurations."""

    q: np.ndarray  # [N_W, 6]
    pos: np.ndarray  # [N_W, 3]
    rot: np.ndarray  # [N_W, 3, 3]
    ee: np.ndarray  # [N_W] arm index
    margin: np.ndarray  # [N_W] normalised joint-range margin in [0, 1]
    q_lo: np.ndarray  # [2, 6] demonstrated joint range per arm (URDF limits are +-10 rad)
    q_hi: np.ndarray

    @classmethod
    def build(cls, state: np.ndarray, per_arm: int = 64, seed: int = 0) -> "Library":
        rng = np.random.default_rng(seed)
        q_lo = np.stack([state[:, s].min(0) for s in ARM_Q])
        q_hi = np.stack([state[:, s].max(0) for s in ARM_Q])
        qs, pos, rot, ee = [], [], [], []
        for arm, s in enumerate(ARM_Q):
            idx = rng.choice(len(state), per_arm, replace=False)
            q = state[idx, s]
            p, r = aloha_fk.tcp_pose(("left", "right")[arm], q)
            qs.append(q), pos.append(p), rot.append(r), ee.append(np.full(per_arm, arm))
        q = np.concatenate(qs)
        ee = np.concatenate(ee)
        lo, hi = q_lo[ee], q_hi[ee]
        margin = np.clip((np.minimum(q - lo, hi - q) / np.maximum(hi - lo, 1e-6)).min(-1) * 2, 0, 1)
        return cls(q, np.concatenate(pos), np.concatenate(rot), ee, margin, q_lo, q_hi)

    def cost(self, state14: np.ndarray) -> np.ndarray:
        """c_i = max_j |q_ij - q_cur,j| / qdot_max against the CURRENT joints of the sample's arm; [U, N_W]."""
        cur = np.stack([state14[:, s] for s in ARM_Q], 1)[:, self.ee]  # [U, N_W, 6]
        return np.abs(self.q[None] - cur).max(-1) / QDOT_MAX

    def save(self, path: str):
        np.savez(path, **dataclasses.asdict(self))

    @classmethod
    def load(cls, path: str) -> "Library":
        d = np.load(path)
        return cls(**{f.name: d[f.name] for f in dataclasses.fields(cls)})


GRIPPER_DESC = np.array([1.0, aloha_fk.GRIPPER_BIAS, 0.05, NUM_EE, 0, 0, 0, 0], np.float32)  # parallel jaw
BASE_DESC = np.array([1.0, 0, 0, 0, 0, 0, 0, 0], np.float32)  # [fixed base, no body-motion command]


def step_inputs(state14: np.ndarray, prev14: np.ndarray | None, dt: float, lib: Library, state_norm: np.ndarray,
                cfg) -> dict:  # fmt: skip
    """Online conditioner inputs for ONE update (used by training and the eval server alike).

    state14     current raw joint state, prev14 the raw state at the previous update (None at t=0)
    state_norm  the openpi-normalised, padded state the VLM prompt also sees
    """
    pos, rot = aloha_fk.both_tcp(state14[None])
    pos, rot = pos[0], rot[0]
    xi = np.zeros(cfg.exec_dim, np.float32)
    if prev14 is not None:
        ppos, prot = aloha_fk.both_tcp(prev14[None])
        xi[0:6] = (pos - ppos[0]).reshape(-1)
        xi[6:12] = rotvec(np.swapaxes(prot[0], -1, -2) @ rot).reshape(-1)
        xi[12:14] = state14[list(GRIP_DIMS)] - prev14[list(GRIP_DIMS)]
        xi[14] = dt
    else:
        xi[15] = 1.0  # first update of the episode: nothing executed yet
    state_mask = np.zeros(cfg.state_dim, bool)
    state_mask[:14] = True
    return {
        "state": state_norm[: cfg.state_dim].astype(np.float32),
        "state_mask": state_mask,
        "exec_increment": xi,
        "exec_mask": np.ones(cfg.exec_dim, bool),
        "dt": np.float32(dt if prev14 is not None else 0.0),
        "gripper_desc": GRIPPER_DESC,
        "base_desc": BASE_DESC,
        "eef_pos": pos.astype(np.float32),
        "eef_rot": rot.astype(np.float32),
        "lib_pos": lib.pos.astype(np.float32),
        "lib_rot": lib.rot.astype(np.float32),
        "lib_cost": lib.cost(state14[None])[0].astype(np.float32),
        "lib_margin": lib.margin.astype(np.float32),
        "ws_mask": np.ones(len(lib.ee), bool),
        "lib_ee": lib.ee.astype(np.int64),
        "ee_mask": np.ones(NUM_EE, bool),
        "new_episode": np.bool_(prev14 is None),
    }


class EpisodeSequences(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, frame_transform, cfg, lib: Library, *, stride: int = 50, horizon: int = 50,
                 trace_lags: int = 4, train: bool = True, cams=("head", "front")):  # fmt: skip
        self.dir = data_dir
        self.meta = dict(np.load(f"{data_dir}/meta.npz"))
        self.tf = frame_transform
        self.cfg, self.lib = cfg, lib
        self.stride, self.horizon, self.lags, self.train = stride, horizon, trace_lags, train
        self.cams = cams
        self.prompt = str(self.meta["prompt"])
        self.dt = stride / float(self.meta["fps"])
        self.events = {}
        for e, f, a, c in self.meta["events"]:
            self.events.setdefault(int(e), []).append((int(f), int(a), int(c)))
        self._img = None

    def __len__(self):
        return len(self.meta["ep_index"])

    def _images(self):
        if self._img is None:  # opened lazily so every DataLoader worker has its own handles
            self._img = {c: np.load(f"{self.dir}/images_{c}.npy", mmap_mode="r") for c in self.cams}
        return self._img

    # -- labels ----------------------------------------------------------------------------------
    def _labels(self, ep: int, s0: int, frames: np.ndarray) -> dict:
        ev = self.events[ep]
        pos, rot, st = self.meta["tcp_pos"], self.meta["tcp_rot"], self.meta["state"]
        right_opens = [f for f, a, c in ev if a == 1 and c == 0]
        recall_only = right_opens[-1] if len(right_opens) >= 2 else None
        u = len(frames)
        lab = {
            "aff_point": np.zeros((u, 2, 3), np.float32),
            "aff_present": np.ones((u, 2), bool),
            "aff_obs_valid": np.zeros((u, 2), bool),
            "aff_use_valid": np.zeros((u, 2), bool),
            "goal_pos": np.zeros((u, NUM_EE, 3), np.float32),
            "goal_rot": np.tile(np.eye(3, dtype=np.float32), (u, NUM_EE, 1, 1)),
            "goal_valid": np.zeros((u, NUM_EE), bool),
            "goal_present": np.ones((u, NUM_EE), bool),
            "trace_pos": np.zeros((u, NUM_EE * self.lags, 3), np.float32),
            "trace_rot": np.tile(np.eye(3, dtype=np.float32), (u, NUM_EE * self.lags, 1, 1)),
            "trace_grip": np.zeros((u, NUM_EE * self.lags), np.float32),
            "trace_avail": np.zeros((u, NUM_EE * self.lags), bool),
            "mode": np.ones(u, np.int64),  # MANIP
            "mode_present": np.ones(u, bool),
        }
        for k, t in enumerate(frames):
            nxt = [(f, a) for f, a, _ in ev if f > t]
            if nxt:
                f, a = nxt[0]
                lab["aff_point"][k, 1] = pos[s0 + f, a]
                lab["aff_use_valid"][k, 1] = True
                lab["aff_obs_valid"][k, 1] = f != recall_only
            for arm in range(NUM_EE):
                own = [f for f, a, _ in ev if a == arm and f > t]
                if own:
                    lab["goal_pos"][k, arm] = pos[s0 + own[0], arm]
                    lab["goal_rot"][k, arm] = rot[s0 + own[0], arm]
                    lab["goal_valid"][k, arm] = True
                for j in range(self.lags):
                    tp = t - (j + 1) * self.stride
                    if tp < 0:
                        continue
                    i = arm * self.lags + j
                    lab["trace_pos"][k, i] = pos[s0 + tp, arm]
                    lab["trace_rot"][k, i] = rot[s0 + tp, arm]
                    lab["trace_grip"][k, i] = st[s0 + tp, GRIP_DIMS[arm]]
                    lab["trace_avail"][k, i] = True
        return lab

    # -- item --------------------------------------------------------------------------------------
    def __getitem__(self, i: int) -> dict:
        ep = int(self.meta["ep_index"][i])
        s0, n = int(self.meta["ep_start"][i]), int(self.meta["ep_len"][i])
        off = np.random.randint(self.stride) if self.train else 0
        frames = np.arange(off, n, self.stride)
        imgs = self._images()
        state, action = self.meta["state"], self.meta["action"]
        obs, steps = [], []
        prev = None
        for t in frames:
            chunk = np.clip(np.arange(t, t + self.horizon), 0, n - 1)  # LeRobot-style clamp at the end
            d = self.tf(
                {
                    "observation.images.head": np.ascontiguousarray(imgs["head"][s0 + t].transpose(2, 0, 1)),
                    "observation.images.front": np.ascontiguousarray(imgs["front"][s0 + t].transpose(2, 0, 1)),
                    "observation.state": state[s0 + t],
                    "action": action[s0 + chunk],
                    "prompt": self.prompt,
                }
            )
            obs.append(d)
            steps.append(step_inputs(state[s0 + t], prev, self.dt, self.lib, np.asarray(d["state"]), self.cfg))
            prev = state[s0 + t]
        return {
            "obs": obs,
            "step": {k: np.stack([s[k] for s in steps]) for k in steps[0]},
            "labels": self._labels(ep, s0, frames),
            "episode": ep,
        }


def collate(items: list[dict]) -> dict:
    """Pad episodes to the longest update count; only ACTIVE frames go through the VLM / heads."""
    b = len(items)
    u = max(len(it["obs"]) for it in items)
    active = np.zeros((b, u), bool)
    slot, upd, obs = [], [], []
    for i, it in enumerate(items):
        active[i, : len(it["obs"])] = True
        for k, d in enumerate(it["obs"]):
            slot.append(i), upd.append(k), obs.append(d)

    def pad(arrs):
        out = np.zeros((b, u, *arrs[0].shape[1:]), arrs[0].dtype)
        for i, a in enumerate(arrs):
            out[i, : len(a)] = a
        return torch.from_numpy(out)

    def stack_obs(key_path):
        vals = obs
        for k in key_path:
            vals = [v[k] for v in vals]
        return torch.from_numpy(np.stack([np.asarray(v) for v in vals]))

    observation = {
        "image": {k: stack_obs(("image", k)) for k in obs[0]["image"]},
        "image_mask": {k: stack_obs(("image_mask", k)) for k in obs[0]["image_mask"]},
        "state": stack_obs(("state",)),
        "tokenized_prompt": stack_obs(("tokenized_prompt",)),
        "tokenized_prompt_mask": stack_obs(("tokenized_prompt_mask",)),
    }
    return {
        "observation": observation,
        "actions": stack_obs(("actions",)),
        "slot": torch.tensor(slot),
        "update": torch.tensor(upd),
        "active": torch.from_numpy(active),
        "step": {k: pad([it["step"][k] for it in items]) for k in items[0]["step"]},
        "labels": {k: pad([it["labels"][k] for it in items]) for k in items[0]["labels"]},
        "episodes": [it["episode"] for it in items],
    }
