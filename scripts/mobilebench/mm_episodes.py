"""Episode-sequence dataset for MobileBench v1.11 on MagicSim Go2Plus mobile manipulation (prep_magicsim_mm.py).

Same item layout as rmbench_episodes / nav_episodes (collate / training / serving share it):
one item = one whole episode as policy updates every `stride` frames from a random offset
(a multiple of img_every, the frames with a stored image). Per update:

  images       Go2 front camera -> base_0_rgb, head D435 -> left_wrist_0_rgb, Piper wrist D435 ->
               right_wrist_0_rgb (one arm; the slot names are openpi's)
  state        mm_labels.state_vector, quantile-normalised, in the prompt
  actions      UPPER head dims 0-9: commanded EEF position + rot6d + gripper of the next 50 frames
               in the body frame of the update; BASE head dims 0-3: commanded (vx, vy, yaw rate,
               body height)
  motion       executed motion since the previous update: EEF position / rotation change in the
               body frame (0:3, 3:6), base (dx, dy, dyaw) in the previous body frame (6:9),
               gripper change (12), dt (14), first-update flag (15)
  kinematics   reachable-pose library sampled from the demonstrated arm configurations (EEF pose in
               the body frame, as Library in rmbench_episodes but without an FK model: the simulator
               logged the pose of every configuration); Piper parallel gripper; legged base with
               the demonstrations' command limits
  affordance   [NAV, OBJ] in the current body frame (mm_labels.affordance)
  slow text    frame_phase from phase_labels.npz; group = pen held (logged as an accuracy)
"""

import dataclasses
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rmbench_episodes import collate  # noqa: E402, F401  (same batch format)
from rmbench_episodes import rotvec  # noqa: E402

from openpi.models import pi0_config  # noqa: E402
from openpi.shared import normalize as _normalize  # noqa: E402
from openpi.training import config as _config  # noqa: E402
import openpi.transforms as _transforms  # noqa: E402

UPPER_DIMS = 10  # EEF pos (3) + rot6d (6) + gripper (1)
BASE_DIMS = 4  # vx, vy, yaw rate, body height
STATE_DIMS = 20
ARM_Q = slice(10, 16)  # arm joints in the state vector
QDOT_MAX = 1.0  # rad/s, coarse joint speed for the library cost (as rmbench_episodes)
# [parallel jaw, TCP bias, max opening (m), end effectors, 0...]: Piper gripper, 2 x 35 mm fingers
GRIPPER_DESC = np.array([1.0, 0.0, 0.07, 1, 0, 0, 0, 0], np.float32)


def rot6d_to_mat(x: np.ndarray) -> np.ndarray:
    a, b = x[..., 0:3], x[..., 3:6]
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b - (a * b).sum(-1, keepdims=True) * a
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.stack([a, b, np.cross(a, b)], -1)


def yaw_mat(yaw) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    o, z = np.ones_like(c), np.zeros_like(c)
    return np.stack([np.stack([c, -s, z], -1), np.stack([s, c, z], -1), np.stack([z, z, o], -1)], -2)


@dataclasses.dataclass
class Library:
    """Fixed reachable-pose library: demonstrated arm configurations and their body-frame EEF poses."""

    q: np.ndarray  # [N_W, 6]
    pos: np.ndarray  # [N_W, 3]
    rot: np.ndarray  # [N_W, 3, 3]
    margin: np.ndarray  # [N_W]
    q_lo: np.ndarray  # [6]
    q_hi: np.ndarray

    @classmethod
    def build(cls, state: np.ndarray, n: int = 64, seed: int = 0) -> "Library":
        q_all = state[:, ARM_Q]
        q_lo, q_hi = q_all.min(0), q_all.max(0)
        idx = np.random.default_rng(seed).choice(len(state), n, replace=False)
        q = q_all[idx]
        margin = np.clip((np.minimum(q - q_lo, q_hi - q) / np.maximum(q_hi - q_lo, 1e-6)).min(-1) * 2, 0, 1)
        return cls(q, state[idx, 0:3], rot6d_to_mat(state[idx, 3:9]), margin, q_lo, q_hi)

    def cost(self, state: np.ndarray) -> np.ndarray:
        return np.abs(self.q - state[ARM_Q]).max(-1) / QDOT_MAX

    def save(self, path: str):
        np.savez(path, **dataclasses.asdict(self))

    @classmethod
    def load(cls, path: str) -> "Library":
        d = np.load(path)
        return cls(**{f.name: d[f.name] for f in dataclasses.fields(cls)})


class MMInputs(_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        out = {
            "image": {"base_0_rgb": data["front"], "left_wrist_0_rgb": data["head"], "right_wrist_0_rgb": data["wrist"]},
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_, "right_wrist_0_rgb": np.True_},
            "state": np.asarray(data["state"], np.float32),
            "prompt": data["prompt"],
        }
        if "actions" in data:
            out["actions"] = np.asarray(data["actions"], np.float32)
        return out


def frame_transform(assets_dir: str):
    stats = _normalize.load(assets_dir)
    model_tf = _config.ModelTransformFactory()(pi0_config.Pi0Config(pi05=True))
    return _transforms.compose([MMInputs(), _transforms.Normalize(stats, use_quantiles=True), *model_tf.inputs]), stats


def unnormalize_actions(upper: np.ndarray, base: np.ndarray, stats) -> tuple[np.ndarray, np.ndarray]:
    """Head outputs (normalised) -> raw upper [H, 10] and base [H, 4]."""
    q01, q99 = np.asarray(stats["actions"].q01), np.asarray(stats["actions"].q99)
    x = np.concatenate([np.asarray(upper)[:, :UPPER_DIMS], np.asarray(base)[:, :BASE_DIMS]], 1)
    x = (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    return x[:, :UPPER_DIMS], x[:, UPPER_DIMS:]


def base_desc(base_cmd: np.ndarray) -> np.ndarray:
    lim = np.abs(base_cmd[:, :3]).max(0)
    return np.array([0.0, 1.0, 1.0, 1.0, lim[0], lim[1], lim[2], 1.0], np.float32)


def upper_chunk(m: dict, s0: int, n: int, t: int, horizon: int) -> np.ndarray:
    """Commanded EEF pose + gripper of frames t..t+H-1 in the body frame of frame t (episode-local t)."""
    k = s0 + np.clip(np.arange(t, t + horizon), 0, n - 1)
    rt = yaw_mat(float(m["yaw"][s0 + t]))
    p = (m["eef_cmd_pos"][k] - m["pos"][s0 + t]) @ rt
    r = rt.T @ rot6d_to_mat(m["eef_cmd_rot6d"][k])
    return np.concatenate([p, r[..., :, 0], r[..., :, 1], m["grip_cmd"][k, None]], 1).astype(np.float32)


def step_inputs(state, base_pos, base_yaw, prev, dt: float, lib: Library, state_norm, cfg, desc) -> dict:
    """Online inputs of ONE update. state: raw state vector; prev: (state, pos, yaw) of the previous update or None."""
    xi = np.zeros(cfg.exec_dim, np.float32)
    if prev is not None:
        ps, ppos, pyaw = prev
        xi[0:3] = state[0:3] - ps[0:3]
        xi[3:6] = rotvec(rot6d_to_mat(ps[3:9]).T @ rot6d_to_mat(state[3:9]))
        c, s = np.cos(pyaw), np.sin(pyaw)
        d = np.asarray(base_pos[:2]) - np.asarray(ppos[:2])
        xi[6:8] = [c * d[0] + s * d[1], -s * d[0] + c * d[1]]
        xi[8] = (base_yaw - pyaw + np.pi) % (2 * np.pi) - np.pi
        xi[12] = state[9] - ps[9]
        xi[14] = dt
    else:
        xi[15] = 1.0
    state_mask = np.zeros(cfg.state_dim, bool)
    state_mask[:STATE_DIMS] = True
    return {
        "state": state_norm[: cfg.state_dim].astype(np.float32),
        "state_mask": state_mask,
        "exec_increment": xi,
        "exec_mask": np.ones(cfg.exec_dim, bool),
        "dt": np.float32(dt if prev is not None else 0.0),
        "gripper_desc": GRIPPER_DESC,
        "base_desc": desc,
        "eef_pos": state[None, 0:3].astype(np.float32),
        "eef_rot": rot6d_to_mat(state[3:9])[None].astype(np.float32),
        "lib_pos": lib.pos.astype(np.float32),
        "lib_rot": lib.rot.astype(np.float32),
        "lib_cost": lib.cost(state).astype(np.float32),
        "lib_margin": lib.margin.astype(np.float32),
        "ws_mask": np.ones(len(lib.q), bool),
        "lib_ee": np.zeros(len(lib.q), np.int64),
        "ee_mask": np.ones(1, bool),
        "new_episode": np.bool_(prev is None),
    }


class MMEpisodeSequences(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, frame_transform, cfg, lib: Library, *, stride: int = 50, horizon: int = 50,
                 train: bool = True):  # fmt: skip
        self.dir = data_dir
        self.meta = dict(np.load(f"{data_dir}/meta.npz"))
        self.tf, self.cfg, self.lib = frame_transform, cfg, lib
        self.stride, self.horizon, self.train = stride, horizon, train
        self.img_every = int(self.meta["img_every"])
        if stride % self.img_every:
            raise ValueError(f"stride {stride} is not a multiple of the stored image interval {self.img_every}")
        self.prompt = str(self.meta["prompt"])
        self.ep_task = ["magicsim_mm"] * len(self.meta["ep_index"])
        self.dt = stride / float(self.meta["fps"])
        self.desc = base_desc(self.meta["base_cmd"])
        pl = np.load(f"{data_dir}/phase_labels.npz")
        self.phase_frame, self.phase_text = pl["frame_phase"], pl["text_emb"]
        self.phase_texts = [str(t) for t in pl["texts"]]
        self.phase_group = pl["group"]
        self._img = None

    def __len__(self):
        return len(self.meta["ep_index"])

    def __getitem__(self, i: int) -> dict:
        m = self.meta
        s0, n = int(m["ep_start"][i]), int(m["ep_len"][i])
        if self._img is None:  # opened lazily so every DataLoader worker has its own handle
            self._img = {k: np.load(f"{self.dir}/images_{k}.npy", mmap_mode="r") for k in ("front", "wrist", "head")}
        off = self.img_every * np.random.randint(self.stride // self.img_every) if self.train else 0
        frames = np.arange(off, n, self.stride)
        obs, steps, base = [], [], []
        prev = None
        for t in frames:
            row = int(m["img_row"][s0 + t])
            chunk = np.clip(np.arange(t, t + self.horizon), 0, n - 1)
            acts = np.concatenate([upper_chunk(m, s0, n, int(t), self.horizon), m["base_cmd"][s0 + chunk]], 1)
            raw = m["state"][s0 + t]
            d = self.tf({**{k: v[row] for k, v in self._img.items()}, "state": raw, "actions": acts, "prompt": self.prompt})
            b = np.zeros_like(d["actions"])
            b[:, :BASE_DIMS] = d["actions"][:, UPPER_DIMS : UPPER_DIMS + BASE_DIMS]
            d["actions"][:, UPPER_DIMS:] = 0.0
            base.append(b)
            obs.append(d)
            pos, yaw = m["pos"][s0 + t], float(m["yaw"][s0 + t])
            steps.append(step_inputs(raw, pos, yaw, prev, self.dt, self.lib, np.asarray(d["state"]), self.cfg, self.desc))
            prev = (raw, pos, yaw)
        u = len(frames)
        ph = self.phase_frame[s0 + frames]
        labels = {
            "aff_point": m["aff_point"][s0 + frames].astype(np.float32),
            "aff_present": np.ones((u, 2), bool),
            "aff_use_valid": m["aff_use_valid"][s0 + frames],
            "aff_obs_valid": m["aff_use_valid"][s0 + frames],  # unused by v1.11 (layout parity)
            "phase_id": np.maximum(ph, 0),
            "phase_present": ph >= 0,
        }
        return {
            "obs": obs,
            "base_actions": np.stack(base),
            "step": {k: np.stack([s[k] for s in steps]) for k in steps[0]},
            "labels": labels,
            "episode": int(m["ep_index"][i]),
            "task": "magicsim_mm",
        }
