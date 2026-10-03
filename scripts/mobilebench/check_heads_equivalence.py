"""Sanity check on real pi05 weights: with an EMPTY condition stream and no base, the Upper
head must reproduce PI0Pytorch.denoise_step exactly (zero-init cross-attention, same layers)."""

import sys

import safetensors.torch
import torch

from openpi.models import pi0_config
from openpi.models_pytorch.mobilebench.action_heads import DualActionHeads
from openpi.models_pytorch.mobilebench.model import _layer_kv
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

weights = sys.argv[1]
torch.manual_seed(0)
dev = torch.device("cuda")
pi0 = PI0Pytorch(pi0_config.Pi0Config(pi05=True, dtype="float32", pytorch_compile_mode=None))
safetensors.torch.load_model(pi0, weights, strict=False)
pi0 = pi0.to(dev).eval()
heads = DualActionHeads.from_pi05(pi0, 256).to(dev).eval()

b, lp = 2, 24
prefix = torch.randn(b, lp, 2048, device=dev)
pad = torch.ones(b, lp, dtype=torch.bool, device=dev)
pad[1, -5:] = False
att = torch.zeros_like(pad)
m4 = torch.where(make_att_2d_masks(pad, att)[:, None], 0.0, -2.3819763e38)
pwe = pi0.paligemma_with_expert
pwe.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
x_t = torch.randn(b, 50, 32, device=dev)
time = torch.tensor([0.3, 0.8], device=dev)
with torch.no_grad():
    _, cache = pwe.forward(attention_mask=m4, position_ids=torch.cumsum(pad, 1) - 1, past_key_values=None,
                           inputs_embeds=[prefix, None], use_cache=True)  # fmt: skip
    kv = [tuple(t.clone() for t in _layer_kv(cache, i)) for i in range(len(pwe.paligemma.language_model.layers))]
    mine, _ = heads(kv, pad, torch.zeros(b, 0, 256, device=dev), torch.zeros(b, 0, dtype=torch.bool, device=dev),
                    x_t, None, time, None)  # fmt: skip
    ref = pi0.denoise_step(None, pad, cache, x_t, time)
err = (mine - ref).abs().max().item()
print(f"max |v_mine - v_ref| = {err:.3e}  (ref scale {ref.abs().mean().item():.3f})")
print("PASS" if err < 1e-3 else "FAIL")
