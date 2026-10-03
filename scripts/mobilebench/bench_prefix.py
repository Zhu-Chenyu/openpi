"""Time the frozen-VLM prefix pass vs the rest of a training step (eager vs sdpa attention)."""

import sys
import time

import torch

sys.path.insert(0, "scripts/mobilebench")
import rmbench_episodes as rme  # noqa: E402
import train_rmbench as tr  # noqa: E402

from openpi.models import model as _model  # noqa: E402
from openpi.models_pytorch.mobilebench.model import LossWeights  # noqa: E402

dev = torch.device("cuda")
cfg = tr.mobilebench_config()
tf, _ = tr.frame_transform()
import numpy as np  # noqa: E402

lib = rme.Library.build(np.load("/data/users/dfv1344/mobilebench_data/put_back_block/meta.npz")["state"])
ds = rme.EpisodeSequences("/data/users/dfv1344/mobilebench_data/put_back_block", tf, cfg, lib)
batch = rme.collate([ds[i] for i in range(8)])
model = tr.build_model("/data/users/dfv1344/openpi_data/pi05_base_pytorch/model.safetensors", cfg, dev).train()
batch = tr.to_device(batch, dev)
batch["observation"] = _model.Observation.from_dict(batch["observation"])
n = batch["actions"].shape[0]
batch["action_mask"] = torch.ones(n, 32, dtype=torch.bool, device=dev)
print("frames", n, "prompt tokens used (max)", int(batch["observation"].tokenized_prompt_mask.sum(1).max()))


def timeit(f, reps=3):
    f()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(reps):
        f()
    torch.cuda.synchronize()
    return (time.time() - t) / reps


for impl in ("eager", "sdpa"):
    model.vlm.language_model.config._attn_implementation = impl  # noqa: SLF001
    with torch.autocast("cuda", dtype=torch.bfloat16):
        tp = timeit(lambda: model.encode_prefix(batch["observation"], train=True))

        def full():
            loss, _ = model.episode_loss(batch, LossWeights())
            loss.backward()

        tf_ = timeit(full)
    print(f"{impl}: prefix {tp:.2f}s  full step {tf_:.2f}s  ({n / tf_:.1f} frames/s)")
