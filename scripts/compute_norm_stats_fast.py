"""compute_norm_stats.py without video decoding, for LeRobot datasets whose norm stats need only state / actions.

Reads observation.state and action straight from the episode parquets, builds the same samples the torch
dataset would (action chunks of model.action_horizon frames clamped at the episode end, as LeRobot's
delta_timestamps query does), and runs the config's repack + data transforms with blank camera images
(the stats never read them). Same RunningStats, same output path as scripts/compute_norm_stats.py.

    python scripts/compute_norm_stats_fast.py --config-name <config>
"""

import glob
import os

import numpy as np
import pandas as pd
import tqdm
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.transforms as transforms


def main(config_name: str, state_key: str = "observation.state", action_key: str = "action", out: str | None = None):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    root = os.path.join(os.environ["HF_LEROBOT_HOME"], data_config.repo_id)
    tf = transforms.compose([*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs])
    blank = np.zeros((3, 8, 8), np.uint8)
    image_keys = [k for k in os.listdir(f"{root}/videos/chunk-000")]
    horizon = config.model.action_horizon
    stats = {k: normalize.RunningStats() for k in ("state", "actions")}
    for path in tqdm.tqdm(sorted(glob.glob(f"{root}/data/chunk-*/episode_*.parquet"))):
        df = pd.read_parquet(path, columns=[state_key, action_key])
        st, act = np.stack(df[state_key]), np.stack(df[action_key])
        idx = np.minimum(np.arange(len(df))[:, None] + np.arange(horizon)[None], len(df) - 1)
        ep = {"state": [], "actions": []}
        for t in range(len(df)):
            x = tf({**{k: blank for k in image_keys}, state_key: st[t], action_key: act[idx[t]], "prompt": ""})
            ep["state"].append(x["state"])
            ep["actions"].append(x["actions"])
        for k in ep:
            stats[k].update(np.stack(ep[k]))
    norm_stats = {k: s.get_statistics() for k, s in stats.items()}
    output_path = out or config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
