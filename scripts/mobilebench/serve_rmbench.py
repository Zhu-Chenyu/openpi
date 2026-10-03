"""Websocket policy server for MobileBench pi0.5 in RMBench closed-loop eval.

The RMBench harness runs in its own (JAX/SAPIEN) venv, so the PyTorch model is served
from this process and the harness talks to it through openpi_client. Unlike a stateless
pi0.5 server, this one keeps the dual memory across calls: one call = one policy update
= one memory write (never per denoising step), and the client sets `reset=True` on the
first call of every episode.

    python scripts/mobilebench/serve_rmbench.py --ckpt <out>/<exp>/<step> --port 8765

Request: {"images": {"cam_high": CHW uint8, "cam_left_wrist": CHW uint8}, "state": [14],
          "prompt": str, "reset": bool}
Reply:   {"actions": [50, 14]} absolute joint targets, like the openpi pi0.5 policy.
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import safetensors.torch
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rmbench_episodes as rme  # noqa: E402
import train_rmbench  # noqa: E402

from openpi.models import model as _model  # noqa: E402
from openpi.models_pytorch.mobilebench.conditioner import StepInputs  # noqa: E402
from openpi.models_pytorch.mobilebench.config import MobileBenchConfig  # noqa: E402
from openpi_client import base_policy  # noqa: E402
from openpi.serving import websocket_policy_server  # noqa: E402
import openpi.transforms as _transforms  # noqa: E402


class MemoryPolicy(base_policy.BasePolicy):
    def __init__(self, ckpt: str, weights: str, num_steps: int = 10, stride: int = 50, fps: float = 25.0):
        exp_dir = os.path.dirname(ckpt.rstrip("/"))
        self.cfg = MobileBenchConfig(**json.load(open(f"{exp_dir}/mobilebench_config.json")))
        self.lib = rme.Library.load(f"{exp_dir}/library.npz")
        self.dev = torch.device("cuda")
        self.model = train_rmbench.build_model(weights, self.cfg, self.dev)
        state = safetensors.torch.load_file(f"{ckpt}/trainable.safetensors", device="cuda")
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.startswith("vlm.")]
        if missing or unexpected:
            raise RuntimeError(f"checkpoint mismatch: missing {missing[:5]} unexpected {unexpected[:5]}")
        self.model.eval()
        _, data = train_rmbench.frame_transform(f"{exp_dir}/assets")
        norm = _transforms.Normalize(data.norm_stats, use_quantiles=data.use_quantile_norm)
        unnorm = _transforms.Unnormalize(data.norm_stats, use_quantiles=data.use_quantile_norm)
        self.input_tf = _transforms.compose([*data.data_transforms.inputs, norm, *data.model_transforms.inputs])
        self.output_tf = _transforms.compose([*data.model_transforms.outputs, unnorm, *data.data_transforms.outputs])
        self.num_steps, self.dt = num_steps, stride / fps
        self.memory, self.prev = None, None
        self.updates = 0

    def infer(self, obs: dict) -> dict:
        state14 = np.asarray(obs["state"], np.float32)
        if obs.get("reset") or self.memory is None:
            self.memory, self.prev, self.updates = self.model.conditioner.init_memory(1, self.dev), None, 0
        d = self.input_tf({"images": obs["images"], "state": state14, "prompt": obs["prompt"]})
        batch = {k: (v[None] if not isinstance(v, dict) else {kk: vv[None] for kk, vv in v.items()}) for k, v in d.items()}
        batch = train_rmbench.to_device(
            {k: (torch.from_numpy(np.asarray(v)) if not isinstance(v, dict)
                 else {kk: torch.from_numpy(np.asarray(vv)) for kk, vv in v.items()}) for k, v in batch.items()},
            self.dev,
        )  # fmt: skip
        observation = _model.Observation.from_dict(batch)
        fields = rme.step_inputs(state14, self.prev, self.dt, self.lib, np.asarray(d["state"]), self.cfg)
        fields = {k: torch.as_tensor(np.asarray(v))[None].to(self.dev) for k, v in fields.items()}
        step = StepInputs(h=None, h_mask=None, **fields)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            act, _, out = self.model.infer(observation, step, self.memory, num_steps=self.num_steps)
        self.memory, self.prev = out.memory, state14
        self.updates += 1
        actions = act[0].float().cpu().numpy()
        res = self.output_tf({"state": np.asarray(d["state"]), "actions": actions})
        logging.info(
            "update %d: slow_written=%s  A_use obj=%s (valid %.2f)  goal L/R valid=%s",
            self.updates, bool(out.slow_written[0]),
            np.round(out.a_use.points[0, 1].float().cpu().numpy(), 3),
            out.a_use.valid_logits[0, 1].sigmoid().item(),
            np.round(out.goal.valid_logits[0].sigmoid().float().cpu().numpy(), 2),
        )  # fmt: skip
        return {"actions": np.asarray(res["actions"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--weights", default="/data/users/dfv1344/openpi_data/pi05_base_pytorch/model.safetensors")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    policy = MemoryPolicy(args.ckpt, args.weights)
    server = websocket_policy_server.WebsocketPolicyServer(policy, host="127.0.0.1", port=args.port)
    logging.info("serving %s on port %d", args.ckpt, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
