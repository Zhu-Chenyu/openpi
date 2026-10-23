"""Episode-sequence dataset for MobileBench v1.11 on MagicSim go2 navigation (prep_magicsim_nav.py).

Same item layout as rmbench_episodes.EpisodeSequences (collate / training / serving share it):
one item = one whole episode as policy updates every `stride` frames from a random offset.
Differences for a legged base without an arm:

  image        the onboard camera only (base_0_rgb); both wrist slots are masked out
  state        measured body-frame (vx, vy, yaw rate), quantile-normalised, in the prompt
  actions      the 50-step chunk of commanded (vx, vy, yaw rate) goes to the BASE head
               (base_actions, dims 0-2); the upper head has no valid dimension
  motion       executed body motion since the previous update from odometry, in the
               previous body frame: (dx, dy, dyaw) + dt + first-update flag
  kinematics   no manipulator: an empty workspace (masked), a "no gripper" descriptor and a
               legged-base descriptor with the command limits
  affordance   [NAV, OBJ] = NavTo stop point / fridge centre in the current body frame; both
               known to the task throughout (use_valid), current-image validity = visibility
  slow text    frame_phase from phase_labels.npz
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rmbench_episodes import collate  # noqa: E402, F401  (same batch format)

from openpi.models import pi0_config  # noqa: E402
from openpi.shared import normalize as _normalize  # noqa: E402
from openpi.training import config as _config  # noqa: E402
import openpi.transforms as _transforms  # noqa: E402

BASE_DIMS = 3  # vx, vy, yaw rate
STATE_DIMS = 3
GRIPPER_DESC = np.zeros(8, np.float32)  # no gripper / end effector


class NavInputs(_transforms.DataTransformFn):
    """Raw nav frame -> openpi model inputs (onboard image as base_0_rgb, wrists masked)."""

    def __call__(self, data: dict) -> dict:
        img = np.asarray(data["image"])
        blank = np.zeros_like(img)
        out = {
            "image": {"base_0_rgb": img, "left_wrist_0_rgb": blank, "right_wrist_0_rgb": blank},
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.False_, "right_wrist_0_rgb": np.False_},
            "state": np.asarray(data["state"], np.float32),
            "prompt": data["prompt"],
        }
        if "actions" in data:
            out["actions"] = np.asarray(data["actions"], np.float32)
        return out


def frame_transform(assets_dir: str):
    """(input transform, norm stats) of a prepped nav data dir / checkpoint assets."""
    stats = _normalize.load(assets_dir)
    model_tf = _config.ModelTransformFactory()(pi0_config.Pi0Config(pi05=True))
    return _transforms.compose([NavInputs(), _transforms.Normalize(stats, use_quantiles=True), *model_tf.inputs]), stats


def unnormalize_base(actions: np.ndarray, stats) -> np.ndarray:
    """Base-head output [H, >=3] (normalised) -> commanded (vx, vy, yaw rate) [H, 3]."""
    q01, q99 = np.asarray(stats["actions"].q01), np.asarray(stats["actions"].q99)
    return (np.asarray(actions)[:, :BASE_DIMS] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


def base_desc(action: np.ndarray) -> np.ndarray:
    """[fixed base, vx allowed, vy allowed, yaw allowed, |vx| max, |vy| max, |yaw rate| max, legged]; the
    limits are the demonstrations' command range (scripted NavTo planner), not the hardware's."""
    lim = np.abs(action).max(0)
    return np.array([0.0, 1.0, 1.0, 1.0, lim[0], lim[1], lim[2], 1.0], np.float32)


def step_inputs(pos, yaw, prev_pos, prev_yaw, dt: float, state_norm: np.ndarray, cfg, desc: np.ndarray) -> dict:
    """Online inputs of ONE update (training and the eval server alike). prev_* None at the first update."""
    xi = np.zeros(cfg.exec_dim, np.float32)
    if prev_pos is not None:
        c, s = np.cos(prev_yaw), np.sin(prev_yaw)
        d = np.asarray(pos[:2]) - np.asarray(prev_pos[:2])
        xi[0:2] = [c * d[0] + s * d[1], -s * d[0] + c * d[1]]  # in the previous body frame
        xi[2] = (yaw - prev_yaw + np.pi) % (2 * np.pi) - np.pi
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
        "dt": np.float32(dt if prev_pos is not None else 0.0),
        "gripper_desc": GRIPPER_DESC,
        "base_desc": desc,
        # No manipulator: one placeholder workspace token, masked out of Z_t.
        "eef_pos": np.zeros((1, 3), np.float32),
        "eef_rot": np.eye(3, dtype=np.float32)[None],
        "lib_pos": np.zeros((1, 3), np.float32),
        "lib_rot": np.eye(3, dtype=np.float32)[None],
        "lib_cost": np.zeros(1, np.float32),
        "lib_margin": np.zeros(1, np.float32),
        "ws_mask": np.zeros(1, bool),
        "lib_ee": np.zeros(1, np.int64),
        "ee_mask": np.zeros(1, bool),
        "new_episode": np.bool_(prev_pos is None),
    }


class NavEpisodeSequences(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, frame_transform, cfg, *, stride: int = 50, horizon: int = 50, train: bool = True):
        self.dir = data_dir
        self.meta = dict(np.load(f"{data_dir}/meta.npz"))
        self.tf = frame_transform
        self.cfg, self.stride, self.horizon, self.train = cfg, stride, horizon, train
        self.prompt = str(self.meta["prompt"])
        self.ep_task = ["magicsim_nav"] * len(self.meta["ep_index"])
        self.dt = stride / float(self.meta["fps"])
        self.desc = base_desc(self.meta["action"])
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
            self._img = np.load(f"{self.dir}/images_onboard.npy", mmap_mode="r")
        off = np.random.randint(self.stride) if self.train else 0
        frames = np.arange(off, n, self.stride)
        obs, steps, base = [], [], []
        prev = None
        for t in frames:
            chunk = np.clip(np.arange(t, t + self.horizon), 0, n - 1)
            d = self.tf({"image": self._img[s0 + t], "state": m["state"][s0 + t], "actions": m["action"][s0 + chunk],
                         "prompt": self.prompt})  # fmt: skip
            base.append(d["actions"])
            d["actions"] = np.zeros_like(d["actions"])  # the upper head has no valid dimension
            obs.append(d)
            pos, yaw = m["pos"][s0 + t], float(m["yaw"][s0 + t])
            steps.append(step_inputs(pos, yaw, *(prev or (None, None)), self.dt, np.asarray(d["state"]), self.cfg, self.desc))
            prev = (pos, yaw)
        u = len(frames)
        ph = self.phase_frame[s0 + frames]
        labels = {
            "aff_point": np.stack([m["nav_point"][s0 + frames], m["obj_point"][s0 + frames]], 1).astype(np.float32),
            "aff_present": np.ones((u, 2), bool),
            "aff_use_valid": np.ones((u, 2), bool),
            "aff_obs_valid": np.repeat(m["obj_visible"][s0 + frames][:, None], 2, 1),
            "phase_id": np.maximum(ph, 0),
            "phase_present": ph >= 0,
        }
        return {
            "obs": obs,
            "base_actions": np.stack(base),
            "step": {k: np.stack([s[k] for s in steps]) for k in steps[0]},
            "labels": labels,
            "episode": int(m["ep_index"][i]),
            "task": "magicsim_nav",
        }
