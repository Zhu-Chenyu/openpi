"""On sheep: (1) the hand-written prefix forward equals the HF KV-cache path (fp32, TF32 off);
(2) time / memory of one LoRA training step for a few batch sizes."""

import sys
import time

import numpy as np
import torch

sys.path.insert(0, "scripts/mobilebench")
import rmbench_episodes as rme  # noqa: E402
import train_rmbench as tr  # noqa: E402

from openpi.models import model as _model  # noqa: E402
from openpi.models_pytorch.mobilebench.model import LossWeights  # noqa: E402
from openpi.models_pytorch.mobilebench.model import trainable_parameter_groups  # noqa: E402

W = "/data/users/dfv1344/openpi_data/pi05_base_pytorch/model.safetensors"
DATA = "/data/users/dfv1344/mobilebench_data/put_back_block"
dev = torch.device("cuda")
cfg = tr.mobilebench_config()
tf, _ = tr.frame_transform()
lib = rme.Library.build(np.load(f"{DATA}/meta.npz")["state"])
ds = rme.EpisodeSequences(DATA, tf, cfg, lib)


def make_batch(n_eps):
    b = tr.to_device(rme.collate([ds[i] for i in range(n_eps)]), dev)
    b["observation"] = _model.Observation.from_dict(b["observation"])
    b["action_mask"] = torch.ones(b["actions"].shape[0], 32, dtype=torch.bool, device=dev)
    return b


# (1) prefix equivalence in the frozen bf16 model
model = tr.build_model(W, cfg, dev).eval()
b = make_batch(1)
with torch.no_grad():
    h1, p1, kv1 = model.encode_prefix(b["observation"], train=False)
    h2, p2, kv2 = model.encode_prefix_reference(b["observation"])
valid = p1[:, :, None]
dh = ((h1.float() - h2.float()).abs() * valid).max().item()
dk = max((a.float() - c.float()).abs().max().item() for (a, _), (c, _) in zip(kv1, kv2))
print(f"prefix: max|dH|={dh:.3e} (|H|~{h2.float().abs().mean().item():.3f})  max|dK|={dk:.3e}  (bf16)")
del model
torch.cuda.empty_cache()

# (2) LoRA training step cost
model = tr.build_model(W, cfg, dev, vlm_lora=True, expert_lora=True).train()
groups = trainable_parameter_groups(model, 2.5e-5, 1e-4)
print("trainable M:", [round(sum(p.numel() for p in g["params"]) / 1e6, 1) for g in groups])
opt = torch.optim.AdamW(groups)
for n_eps in (4, 8):
    b = make_batch(n_eps)
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(3):
        torch.cuda.synchronize()
        t = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, logs = model.episode_loss(b, LossWeights())
        opt.zero_grad()
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        ts.append(time.time() - t)
    n = b["actions"].shape[0]
    print(f"eps={n_eps} frames={n}: {min(ts):.2f}s/step  {n / min(ts):.1f} frames/s  peak {torch.cuda.max_memory_allocated() / 1e9:.1f}GB  fm={logs['fm_upper']:.3f}")
