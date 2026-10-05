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
        n_eps = len(self.meta["ep_index"])
        if "prompts" in self.meta:  # multi-task prep
            self.prompts = [str(p) for p in self.meta["prompts"]]
            self.ep_prompt = self.meta["ep_prompt"]
            self.ep_task = [str(t) for t in self.meta["ep_task"]]
        else:  # single-task put_back_block prep
            self.prompts = [str(self.meta["prompt"])]
            self.ep_prompt = np.zeros(n_eps, int)
            self.ep_task = ["put_back_block"] * n_eps
        self.video_root = str(self.meta.get("video_root", ""))
        self.dt = stride / float(self.meta["fps"])
        self.events = {}
        for e, f, a, c in self.meta["events"]:
            self.events.setdefault(int(e), []).append((int(f), int(a), int(c)))
        self._img = None

    def __len__(self):
        return len(self.meta["ep_index"])

    def _frames(self, ep: int, s0: int, n: int, frames: np.ndarray) -> dict:
        """{cam: uint8 [len(frames), H, W, 3]} from the memmaps, or decoded from the episode mp4s."""
        if not self.video_root:
            if self._img is None:  # opened lazily so every DataLoader worker has its own handles
                self._img = {c: np.load(f"{self.dir}/images_{c}.npy", mmap_mode="r") for c in self.cams}
            return {c: self._img[c][s0 + frames] for c in self.cams}
        import av

        out = {}
        want = set(int(t) for t in frames)
        last = int(frames.max())
        for c in self.cams:
            path = f"{self.video_root}/videos/chunk-{ep // 1000:03d}/observation.images.{c}/episode_{ep:06d}.mp4"
            got = {}
            with av.open(path) as con:
                for t, fr in enumerate(con.decode(video=0)):
                    if t in want:
                        got[t] = fr.to_ndarray(format="rgb24")
                    if t >= last:
                        break
            final = got[max(got)]  # a video a frame shorter than the parquet: hold the last frame
            out[c] = np.stack([got.get(int(t), final) for t in frames])
        return out

    # -- labels ----------------------------------------------------------------------------------
    def _labels(self, ep: int, s0: int, frames: np.ndarray, task: str) -> dict:
        ev = self.events.get(ep, [])
        pos, rot, st = self.meta["tcp_pos"], self.meta["tcp_rot"], self.meta["state"]
        # "Knowable from the current image" needs a per-task rule. Only put_back_block has one
        # (its final return to the original mat is recall-only); on other tasks the CURRENT
        # branch is left unlabelled rather than given a guessed label (sec. 9.3: "no label"
        # and "truly invalid" are different states). The JOINT branch is labelled everywhere.
        obs_rule = task == "put_back_block"
        right_opens = [f for f, a, c in ev if a == 1 and c == 0]
        recall_only = right_opens[-1] if obs_rule and len(right_opens) >= 2 else None
        u = len(frames)
        lab = {
            "aff_point": np.zeros((u, 2, 3), np.float32),
            "aff_present": np.ones((u, 2), bool),
            "aff_obs_present": np.full((u, 2), obs_rule),
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
        imgs = self._frames(ep, s0, n, frames)
        task, prompt = self.ep_task[i], self.prompts[int(self.ep_prompt[i])]
        state, action = self.meta["state"], self.meta["action"]
        obs, steps = [], []
        prev = None
        for k, t in enumerate(frames):
            chunk = np.clip(np.arange(t, t + self.horizon), 0, n - 1)  # LeRobot-style clamp at the end
            d = self.tf(
                {
                    "observation.images.head": np.ascontiguousarray(imgs["head"][k].transpose(2, 0, 1)),
                    "observation.images.front": np.ascontiguousarray(imgs["front"][k].transpose(2, 0, 1)),
                    "observation.state": state[s0 + t],
                    "action": action[s0 + chunk],
                    "prompt": prompt,
                }
            )
            obs.append(d)
            steps.append(step_inputs(state[s0 + t], prev, self.dt, self.lib, np.asarray(d["state"]), self.cfg))
            prev = state[s0 + t]
        return {
            "obs": obs,
            "step": {k: np.stack([s[k] for s in steps]) for k in steps[0]},
            "labels": self._labels(ep, s0, frames, task),
            "episode": ep,
            "task": task,
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
        "tasks": [it.get("task") for it in items],
    }


def first(items: list):
    """DataLoader collate_fn for batch_size=1 that keeps the raw (numpy) episode item."""
    return items[0]


def slice_item(item: dict, a: int, b: int) -> dict:
    """Updates [a, b) of an episode item (labels were computed with full-episode context)."""
    return {
        "obs": item["obs"][a:b],
        "step": {k: v[a:b] for k, v in item["step"].items()},
        "labels": {k: v[a:b] for k, v in item["labels"].items()},
        "episode": item["episode"],
        "task": item["task"],
    }


class WindowStream:
    """Truncated-BPTT stream (design doc sec. 9.6-9.7).

    Each of `slots` batch slots walks through one episode at a time in windows of `window`
    consecutive policy updates. The model carries every slot's memory from one window to
    the next (detached at the boundary, NOT cleared); the first update of a new episode has
    step["new_episode"] = True, which resets only that slot. A window never spans two
    episodes: a slot whose episode ends mid-window is padded (inactive) and starts its next
    episode in the next batch. Sampling is therefore frame-weighted, like random-frame
    training. Episodes come from `episodes`, an iterator of single episode items.
    """

    def __init__(self, episodes, slots: int, window: int):
        self.episodes, self.window = episodes, window
        self.cur = [None] * slots
        self.pos = [0] * slots

    def next_batch(self) -> dict:
        items = []
        for s in range(len(self.cur)):
            if self.cur[s] is None or self.pos[s] >= len(self.cur[s]["obs"]):
                self.cur[s], self.pos[s] = next(self.episodes), 0
            a = self.pos[s]
            items.append(slice_item(self.cur[s], a, a + self.window))
            self.pos[s] = a + self.window
        return collate(items)
