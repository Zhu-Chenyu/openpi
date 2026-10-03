"""fp32 / TF32-off check that MobileBenchPi05.encode_prefix == the HF KV-cache path."""

import sys

import numpy as np
import safetensors.torch
import torch

sys.path.insert(0, "scripts/mobilebench")
import rmbench_episodes as rme  # noqa: E402
import train_rmbench as tr  # noqa: E402

from openpi.models import model as _model  # noqa: E402
from openpi.models import pi0_config  # noqa: E402
from openpi.models_pytorch.mobilebench.model import MobileBenchPi05  # noqa: E402
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch  # noqa: E402

DATA = "/data/users/dfv1344/mobilebench_data/put_back_block"
dev = torch.device("cuda")
cfg = tr.mobilebench_config()
pi0 = PI0Pytorch(pi0_config.Pi0Config(pi05=True, dtype="float32", pytorch_compile_mode=None))
safetensors.torch.load_model(pi0, "/data/users/dfv1344/openpi_data/pi05_base_pytorch/model.safetensors", strict=False)
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
model = MobileBenchPi05(pi0, cfg).to(dev).eval()
tf, _ = tr.frame_transform()
lib = rme.Library.build(np.load(f"{DATA}/meta.npz")["state"])
b = tr.to_device(rme.collate([rme.EpisodeSequences(DATA, tf, cfg, lib)[0]]), dev)
obs = _model.Observation.from_dict(b["observation"])
with torch.no_grad():
    h1, p1, kv1 = model.encode_prefix(obs, train=False)
    h2, p2, kv2 = model.encode_prefix_reference(obs)
v = p1[:, :, None].float()
rel = lambda a, c: ((a - c).abs() * v).sum().item() / ((c.abs() * v).sum().item())  # noqa: E731
print("H rel err %.2e, max abs %.2e" % (rel(h1, h2), ((h1 - h2).abs() * v).max().item()))
for i in (0, 1, 8, 17):
    print(f"layer {i}: K rel err {(kv1[i][0] - kv2[i][0]).abs().mean().item() / kv2[i][0].abs().mean().item():.2e}")
