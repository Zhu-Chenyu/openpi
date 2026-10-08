"""Task-weighted sampling for multi-task datasets.

Plain shuffling samples every frame equally, so a task's share of training is its share of
the dataset's frames (RMBench put_back_block: 5.4% of the ten-task frames, because its
episodes are short). `task_weights` returns a per-item sampling weight that instead gives
each named task a fixed share of the sampled frames and splits the remaining share over the
other tasks in proportion to their frames (i.e. their natural mix, scaled down).

The same weight works for both samplers used in this repo:
  * per-frame sampling (openpi data loader): weight each frame by its task's weight;
  * per-episode sampling where an episode contributes all of its frames (MobileBench
    episode-sequence trainer): weight each episode by its task's weight. Episode e is then
    picked with probability ~ c_task and contributes len_e frames, so a task's expected
    frame share is c_task * F_task / sum = its target share.
"""

from collections.abc import Mapping, Sequence

import numpy as np


def task_weights(item_task: Sequence[str], item_frames: Sequence[int], shares: Mapping[str, float]) -> np.ndarray:
    """Sampling weight per item (frame or episode) so task t receives `shares[t]` of the frames.

    item_task:   task name of each item.
    item_frames: frames each item contributes (1 for per-frame sampling, the episode length
                 for per-episode sampling).
    """
    task = np.asarray(item_task)
    frames = np.asarray(item_frames, dtype=np.float64)
    unknown = set(shares) - set(task.tolist())
    if unknown:
        raise ValueError(f"task_frame_shares names tasks not in the dataset: {sorted(unknown)}")
    named = sum(shares.values())
    if not 0 < named <= 1:
        raise ValueError(f"task_frame_shares must sum to (0, 1], got {named}")
    rest = ~np.isin(task, list(shares))
    if named < 1 and not rest.any():
        raise ValueError("task_frame_shares sums to < 1 but no other task is left to take the remainder")
    w = np.zeros(len(task))
    for t, s in shares.items():
        m = task == t
        w[m] = s / frames[m].sum()
    if rest.any():
        w[rest] = (1.0 - named) / frames[rest].sum()
    return w


def frame_shares(item_task: Sequence[str], item_frames: Sequence[int], weights: np.ndarray) -> dict[str, float]:
    """Expected share of sampled frames per task under `weights` (for logging)."""
    task = np.asarray(item_task)
    mass = np.asarray(weights) * np.asarray(item_frames, dtype=np.float64)
    return {str(t): float(mass[task == t].sum() / mass.sum()) for t in np.unique(task)}
