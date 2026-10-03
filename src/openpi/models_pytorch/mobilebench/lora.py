"""LoRA for the PyTorch pi0.5 Gemma stacks, matching openpi's JAX `gemma_*_lora` variants.

JAX openpi (models/lora.py, models/gemma.py) puts LoRA on the attention projections and
the FFN of each Gemma layer, scaling alpha / rank, with BOTH factors initialised
N(0, 0.01) (not the zero-B init of the LoRA paper):
    gemma_2b_lora    rank 16, alpha 16      (VLM language model)
    gemma_300m_lora  rank 32, alpha 32      (action expert)
and its freeze filter freezes only the two Gemma stacks ("llm"); everything else --
SigLIP, the image projector, action in/out projections and the time MLP -- is trained
fully. `apply_gemma_lora` reproduces that per stack.
"""

import torch
from torch import Tensor
from torch import nn

ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
FFN = ("gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, init_std: float = 0.01):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.scaling = alpha / rank
        self.lora_a = nn.Parameter(torch.randn(rank, base.in_features) * init_std)
        self.lora_b = nn.Parameter(torch.randn(base.out_features, rank) * init_std)

    @property
    def weight(self) -> Tensor:  # callers read .weight.dtype
        return self.base.weight

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: Tensor) -> Tensor:
        lora = (x.to(self.lora_a.dtype) @ self.lora_a.t()) @ self.lora_b.t()
        return self.base(x) + (lora * self.scaling).to(x.dtype)


def apply_gemma_lora(gemma_model: nn.Module, rank: int, alpha: float) -> list[nn.Parameter]:
    """Freeze a Gemma stack and add LoRA to every layer's attention + FFN. Returns the LoRA params."""
    gemma_model.requires_grad_(False)
    params = []
    for layer in gemma_model.layers:
        for parent, names in ((layer.self_attn, ATTN), (layer.mlp, FFN)):
            for name in names:
                lin = getattr(parent, name)
                if isinstance(lin, LoRALinear):
                    continue
                wrapped = LoRALinear(lin, rank, alpha).to(device=lin.weight.device)
                setattr(parent, name, wrapped)
                params += [wrapped.lora_a, wrapped.lora_b]
    return params
