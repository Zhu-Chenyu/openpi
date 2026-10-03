"""Upper / Base action heads with same-layer bidirectional cross-attention (design doc sec. 8.5, 9.2).

Each head is a pi0.5 action expert (Gemma-300M with adaRMS time conditioning). Per layer
l and head b in {U, B}:

    Xbar_l^b  = X_l^b + Attn^b(X_l^b, [H_t ; C_t ; X_l^b])           (joint attention, gated residual)
    X'^U      = Xbar^U + Attn_{B->U}(Xbar^U, Xbar^B)                 (both read PRE-update Xbar)
    X'^B      = Xbar^B + Attn_{U->B}(Xbar^B, Xbar^U)
    X_{l+1}^b = X'^b + FFN^b(X'^b)                                    (gated residual)

H_t enters as pi0.5's prefix KV cache (the original direct H_t path), and the typed
condition stream C_t (M^F, M^S, state, workspace, capability, affordance, goal, mode,
gripper, base) is a token block in front of the noisy actions inside each head's own
sequence. This is the "same typed token sequence" form of Attn_H + Attn_F + Attn_S +
Attn_geom that sec. 8.5 allows: every action token attends to all of them directly.
Condition tokens attend to the prefix and to each other; they never see the actions.

Both heads use the SAME flow time tau with independent noise (sec. 9.2). A batch with
no base (e.g. the fixed-base RMBench arms) skips the Base head entirely; in a mixed
batch, the B->U message of a sample without a base is masked to zero. The cross-
attention output projections are zero-initialised, so at step 0 the Upper head is
exactly the pretrained pi0.5 expert plus the condition tokens.
"""

import copy
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812
from transformers.models.gemma import modeling_gemma


def time_embedding(time: Tensor, dim: int, min_period: float = 4e-3, max_period: float = 4.0) -> Tensor:
    """pi0.5's sine-cosine embedding of the flow time, [B] -> [B, dim] (float32)."""
    fraction = torch.linspace(0.0, 1.0, dim // 2, dtype=torch.float64, device=time.device)
    period = min_period * (max_period / min_period) ** fraction
    ang = (2 * math.pi / period)[None, :] * time.double()[:, None]
    return torch.cat([ang.sin(), ang.cos()], dim=1).float()


class ActionBranch(nn.Module):
    """One action expert: Gemma layers + action in/out projections + time MLP + condition input."""

    def __init__(self, expert: nn.Module, action_in: nn.Linear, action_out: nn.Linear, time_in, time_out, d_cond):
        super().__init__()
        self.expert = expert  # transformers GemmaModel (embed_tokens is None)
        self.action_in = action_in
        self.action_out = action_out
        self.time_in = time_in
        self.time_out = time_out
        width = action_in.out_features
        self.cond_in = nn.Sequential(nn.LayerNorm(d_cond), nn.Linear(d_cond, width))

    @property
    def width(self) -> int:
        return self.action_in.out_features

    def adarms(self, time: Tensor) -> Tensor:
        x = F.silu(self.time_in(time_embedding(time, self.width)))
        return F.silu(self.time_out(x))


class CrossBranchAttention(nn.Module):
    """Attn_{src->dst}: dst action tokens query the other head's action tokens."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.ln_q = nn.LayerNorm(width)
        self.ln_kv = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, dst: Tensor, src: Tensor) -> Tensor:
        kv = self.ln_kv(src)
        return self.attn(self.ln_q(dst), kv, kv, need_weights=False)[0]


class DualActionHeads(nn.Module):
    def __init__(self, upper: ActionBranch, base: ActionBranch, action_horizon: int, cross_heads: int = 8):
        super().__init__()
        self.upper = upper
        self.base = base
        self.action_horizon = action_horizon
        n_layers = len(upper.expert.layers)
        if len(base.expert.layers) != n_layers:
            raise ValueError("Upper and Base heads must have the same depth for same-layer exchange")
        self.cross_to_upper = nn.ModuleList(CrossBranchAttention(upper.width, cross_heads) for _ in range(n_layers))
        self.cross_to_base = nn.ModuleList(CrossBranchAttention(base.width, cross_heads) for _ in range(n_layers))

    @classmethod
    def from_pi05(cls, pi0, d_cond: int, base_init: str = "copy") -> "DualActionHeads":
        """Upper head = the pretrained pi0.5 expert; Base head = a copy of it (or fresh projections)."""
        pwe = pi0.paligemma_with_expert
        upper = ActionBranch(
            pwe.gemma_expert.model, pi0.action_in_proj, pi0.action_out_proj, pi0.time_mlp_in, pi0.time_mlp_out, d_cond
        )
        base = ActionBranch(
            copy.deepcopy(pwe.gemma_expert.model),
            copy.deepcopy(pi0.action_in_proj),
            copy.deepcopy(pi0.action_out_proj),
            copy.deepcopy(pi0.time_mlp_in),
            copy.deepcopy(pi0.time_mlp_out),
            d_cond,
        )
        if base_init == "fresh_io":
            for m in (base.action_in, base.action_out):
                m.reset_parameters()
        return cls(upper, base, pi0.config.action_horizon)

    # ------------------------------------------------------------------------------------
    def _embed(self, br: ActionBranch, cond: Tensor, x_t: Tensor) -> Tensor:
        dtype = br.action_in.weight.dtype
        return torch.cat([br.cond_in(cond.to(dtype)), br.action_in(x_t.to(dtype))], dim=1)

    @staticmethod
    def _attend(layer, x: Tensor, adarms: Tensor, cos, sin, prefix_kv, mask: Tensor) -> Tensor:
        """Joint attention of one Gemma layer: queries from x, keys = [prefix KV ; x]. Returns Xbar."""
        h, gate = layer.input_layernorm(x, cond=adarms)
        attn = layer.self_attn
        shape = (*h.shape[:-1], -1, attn.head_dim)
        q = attn.q_proj(h).view(shape).transpose(1, 2)
        k = attn.k_proj(h).view(shape).transpose(1, 2)
        v = attn.v_proj(h).view(shape).transpose(1, 2)
        q, k = modeling_gemma.apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
        pk, pv = prefix_kv
        k = torch.cat([pk.to(k.dtype), k], dim=2)
        v = torch.cat([pv.to(v.dtype), v], dim=2)
        # Grouped-query attention: broadcast the single KV head over the query heads.
        k = k.expand(-1, q.shape[1], -1, -1)
        v = v.expand(-1, q.shape[1], -1, -1)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None], scale=attn.scaling)
        o = o.transpose(1, 2).reshape(*x.shape[:-1], -1)
        o = attn.o_proj(o.to(attn.o_proj.weight.dtype))
        return modeling_gemma._gated_residual(x, o, gate)  # noqa: SLF001

    @staticmethod
    def _ffn(layer, x: Tensor, adarms: Tensor) -> Tensor:
        h, gate = layer.post_attention_layernorm(x, cond=adarms)
        h = layer.mlp(h.to(layer.mlp.up_proj.weight.dtype))
        return modeling_gemma._gated_residual(x, h, gate)  # noqa: SLF001

    def forward(
        self,
        prefix_kv: list[tuple[Tensor, Tensor]],
        prefix_pad: Tensor,
        cond: Tensor,
        cond_mask: Tensor,
        x_t_upper: Tensor,
        x_t_base: Tensor | None,
        time: Tensor,
        base_present: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        """Velocity fields (v^U, v^B) for one flow-matching evaluation.

        prefix_kv   per-layer (K, V) [N, 1, L_p, head_dim] of the frozen VLM prefix (post-RoPE)
        prefix_pad  [N, L_p] True = valid prefix token
        cond        [N, N_c, d_cond] typed condition tokens C_t, cond_mask [N, N_c]
        x_t_*       [N, H, action_dim] noisy action chunks (same tau, independent noise)
        time        [N] flow time tau
        base_present [N] bool, or None = no sample has a base (Base head skipped)
        """
        n, hzn = x_t_upper.shape[:2]
        dev = x_t_upper.device
        run_base = base_present is not None and x_t_base is not None and bool(base_present.any())
        branches = [(self.upper, x_t_upper)] + ([(self.base, x_t_base)] if run_base else [])

        # Shared positions / masks: suffix = [cond block ; action block].
        nc = cond.shape[1]
        suffix_pad = torch.cat([cond_mask, torch.ones(n, hzn, dtype=torch.bool, device=dev)], dim=1)
        block = torch.zeros(nc + hzn, dtype=torch.long, device=dev)
        block[nc:] = 1  # cond tokens cannot see actions; actions see everything
        suffix_att = (block[None, :] <= block[:, None])[None] & suffix_pad[:, None, :] & suffix_pad[:, :, None]
        # Padded condition queries still see the prefix, so no attention row is empty.
        mask = torch.cat([prefix_pad[:, None, :].expand(n, nc + hzn, -1), suffix_att], dim=2)
        pos = prefix_pad.sum(-1, keepdim=True) + torch.cumsum(suffix_pad, dim=1) - 1

        xs, conds = [], []
        for br, x_t in branches:
            x = self._embed(br, cond, x_t)
            xs.append(x)
            conds.append(br.adarms(time).to(x.dtype))
        cos, sin = self.upper.expert.rotary_emb(xs[0], pos)

        for li in range(len(self.upper.expert.layers)):
            xbar = [
                self._attend(br.expert.layers[li], x, c, cos, sin, prefix_kv[li], mask)
                for (br, _), x, c in zip(branches, xs, conds, strict=True)
            ]
            if run_base:
                u_act, b_act = xbar[0][:, nc:], xbar[1][:, nc:]
                to_u = self.cross_to_upper[li](u_act, b_act) * base_present[:, None, None].to(u_act.dtype)
                to_b = self.cross_to_base[li](b_act, u_act)
                xbar[0] = torch.cat([xbar[0][:, :nc], u_act + to_u], dim=1)
                xbar[1] = torch.cat([xbar[1][:, :nc], b_act + to_b], dim=1)
            xs = [
                self._ffn(br.expert.layers[li], x, c) for (br, _), x, c in zip(branches, xbar, conds, strict=True)
            ]

        outs = []
        for (br, _), x, c in zip(branches, xs, conds, strict=True):
            x, _ = br.expert.norm(x, cond=c)
            outs.append(br.action_out(x[:, nc:].to(br.action_out.weight.dtype)).float())
        return outs[0], (outs[1] if run_base else None)
