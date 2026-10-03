"""MobileBench pi0.5: frozen pi0.5 VLM + MobileBench conditioner + dual action heads.

One policy update (design doc sec. 10.2) is

    H_t, KV_t = VLM(current images, ORIGINAL instruction, state)      frozen, no_grad
    out       = conditioner(H_t, state, executed increment, ..., memory)
    v^U, v^B  = DualActionHeads(KV_t, out.cond_tokens, Y_tau^U, Y_tau^B, tau)

Training runs whole episode sequences (sec. 9.6/9.7): the VLM pass of every update in
the batch is one batched no_grad call, the conditioner then steps through the updates in
order carrying the memory with gradients, and all updates' action heads run as one
batch. The VLM is frozen, but nothing downstream of H_t is inside no_grad, so the FM
and auxiliary losses reach the memory writers (sec. 9.5).
"""

import dataclasses
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812
from transformers.models.gemma import modeling_gemma

from openpi.models_pytorch import preprocessing_pytorch as _preprocessing
from openpi.models_pytorch.mobilebench import losses as L
from openpi.models_pytorch.mobilebench.action_heads import DualActionHeads
from openpi.models_pytorch.mobilebench.conditioner import MobileBenchConditioner
from openpi.models_pytorch.mobilebench.conditioner import StepInputs
from openpi.models_pytorch.mobilebench.conditioner import StepOutputs
from openpi.models_pytorch.mobilebench.config import MobileBenchConfig
from openpi.models_pytorch.mobilebench.lora import apply_gemma_lora
from openpi.models_pytorch.mobilebench.memory import MemoryState
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks


@dataclasses.dataclass
class LossWeights:
    fm_upper: float = 1.0
    fm_base: float = 1.0
    aff_obs: float = 1.0
    aff_use: float = 1.0
    valid: float = 0.1
    goal: float = 1.0
    trace: float = 0.5
    mode: float = 0.01
    phase_text: float = 0.0  # needs a frozen text teacher; off until phase labels exist


def _layer_kv(cache, i: int) -> tuple[Tensor, Tensor]:
    if hasattr(cache, "layers"):  # transformers >= 4.54
        return cache.layers[i].keys, cache.layers[i].values
    return cache.key_cache[i], cache.value_cache[i]


class MobileBenchPi05(nn.Module):
    def __init__(self, pi0, cfg: MobileBenchConfig, *, vlm_lora: bool = False, expert_lora: bool = False):
        """pi0: a PI0Pytorch with pi05 weights loaded. Its VLM gives H_t and the prefix KV;
        its action expert becomes the Upper head and a copy of it the Base head.

        vlm_lora=False   VLM fully frozen, prefix computed under no_grad.
        vlm_lora=True    openpi `gemma_2b_lora` recipe: LoRA r16 on the language model,
                         SigLIP + image projector fully trained (fp32 master weights).
        expert_lora      openpi `gemma_300m_lora` recipe for both heads: LoRA r32, the
                         expert's own weights frozen, action/time projections trained.
        """
        super().__init__()
        self.cfg = cfg
        self.vlm_lora, self.expert_lora = vlm_lora, expert_lora
        self.action_horizon = pi0.config.action_horizon
        self.action_dim = pi0.config.action_dim
        self.vlm = pi0.paligemma_with_expert.paligemma
        self.vlm.requires_grad_(False)
        if vlm_lora:
            apply_gemma_lora(self.vlm.language_model, rank=16, alpha=16.0)
            for m in (self.vlm.model.vision_tower, self.vlm.model.multi_modal_projector):
                m.float().requires_grad_(True)
            self._enable_siglip_checkpointing()
        self.heads = DualActionHeads.from_pi05(pi0, cfg.d_model).float()
        if expert_lora:
            for br in (self.heads.upper, self.heads.base):
                apply_gemma_lora(br.expert, rank=32, alpha=32.0)
        self.conditioner = MobileBenchConditioner(cfg)

    def _enable_siglip_checkpointing(self):
        tower = self.vlm.model.vision_tower
        try:
            tower.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except (AttributeError, ValueError):
            tower.gradient_checkpointing = True

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.vlm_lora:
            self.vlm.eval()  # frozen: always inference behaviour
        return self

    # -- VLM prefix --------------------------------------------------------------------------
    @staticmethod
    def _llm_layer(layer, x: Tensor, cos: Tensor, sin: Tensor, mask: Tensor):
        """One Gemma layer over the prefix; also returns this layer's (post-RoPE) K, V."""
        h, _ = layer.input_layernorm(x)
        attn = layer.self_attn
        shape = (*h.shape[:-1], -1, attn.head_dim)
        q = attn.q_proj(h).view(shape).transpose(1, 2)
        k = attn.k_proj(h).view(shape).transpose(1, 2)
        v = attn.v_proj(h).view(shape).transpose(1, 2)
        q, k = modeling_gemma.apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
        o = F.scaled_dot_product_attention(
            q, k.expand(-1, q.shape[1], -1, -1), v.expand(-1, q.shape[1], -1, -1), attn_mask=mask, scale=attn.scaling
        )
        x = x + attn.o_proj(o.transpose(1, 2).reshape(*x.shape[:-1], -1).to(x.dtype))
        h, _ = layer.post_attention_layernorm(x)
        x = x + layer.mlp(h.to(x.dtype))
        return x, k, v

    def encode_prefix(self, observation, *, train: bool) -> tuple[Tensor, Tensor, list[tuple[Tensor, Tensor]]]:
        """Returns (H_t [N, L_p, vlm_width], prefix_pad [N, L_p], per-layer prefix KV).

        Gradients flow only when the VLM has trainable parts (vlm_lora) and grad mode is on;
        each language-model layer is then checkpointed, so only layer inputs and the small
        single-head K/V are kept for the backward pass.
        """
        with torch.set_grad_enabled(self.vlm_lora and torch.is_grad_enabled()):
            obs = _preprocessing.preprocess_observation_pytorch(observation, train=train)
            embs, pads = [], []
            for key in obs.images:
                img, m = obs.images[key], obs.image_masks[key]
                if not bool(m.any()):
                    # An image that is masked for every sample (e.g. RMBench's missing right
                    # wrist) contributes only padding; skip its SigLIP pass entirely.
                    continue
                e = self.vlm.model.get_image_features(img)
                embs.append(e)
                pads.append(m[:, None].expand(e.shape[0], e.shape[1]))
            # Prompts are right-padded to max_token_len (200 for pi0.5); the padding is masked
            # anyway, so drop the columns no sample uses instead of running them through 18 layers.
            used = max(int(obs.tokenized_prompt_mask.sum(dim=1).max()), 1)
            lm = self.vlm.language_model
            lang = lm.embed_tokens(obs.tokenized_prompt[:, :used])
            embs.append(lang * math.sqrt(lang.shape[-1]))
            pads.append(obs.tokenized_prompt_mask[:, :used])
            dtype = lm.layers[0].self_attn.q_proj.weight.dtype
            x = torch.cat([e.to(dtype) for e in embs], dim=1)
            pad = torch.cat(pads, dim=1)
            # Full attention inside the prefix. Every query row (padding included) sees all
            # valid keys, so no row is empty -- valid rows are unaffected by this choice.
            mask = pad[:, None, None, :]
            cos, sin = lm.rotary_emb(x, torch.cumsum(pad, dim=1) - 1)
            kv = []
            ckpt = torch.is_grad_enabled()
            for layer in lm.layers:
                if ckpt:
                    x, k, v = torch.utils.checkpoint.checkpoint(
                        self._llm_layer, layer, x, cos, sin, mask, use_reentrant=False, preserve_rng_state=False
                    )
                else:
                    x, k, v = self._llm_layer(layer, x, cos, sin, mask)
                kv.append((k, v))
            h, _ = lm.norm(x)
            return h, pad, kv

    @torch.no_grad()
    def encode_prefix_reference(self, observation) -> tuple[Tensor, Tensor, list[tuple[Tensor, Tensor]]]:
        """The HF forward + KV cache path (as in PI0Pytorch.sample_actions), for checking encode_prefix."""
        obs = _preprocessing.preprocess_observation_pytorch(observation, train=False)
        embs, pads = [], []
        for key in obs.images:
            img, m = obs.images[key], obs.image_masks[key]
            if bool(m.any()):
                e = self.vlm.model.get_image_features(img)
                embs.append(e)
                pads.append(m[:, None].expand(e.shape[0], e.shape[1]))
        used = max(int(obs.tokenized_prompt_mask.sum(dim=1).max()), 1)
        lang = self.vlm.language_model.embed_tokens(obs.tokenized_prompt[:, :used])
        embs.append(lang * math.sqrt(lang.shape[-1]))
        pads.append(obs.tokenized_prompt_mask[:, :used])
        dtype = self.vlm.language_model.layers[0].self_attn.q_proj.weight.dtype
        embs = torch.cat([e.to(dtype) for e in embs], dim=1)
        pad = torch.cat(pads, dim=1)
        mask4d = torch.where(make_att_2d_masks(pad, torch.zeros_like(pad))[:, None], 0.0, -2.3819763e38).to(dtype)
        self.vlm.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        out = self.vlm.language_model.forward(
            inputs_embeds=embs, attention_mask=mask4d, position_ids=torch.cumsum(pad, dim=1) - 1,
            past_key_values=None, use_cache=True, adarms_cond=None,
        )  # fmt: skip
        kv = [_layer_kv(out.past_key_values, i) for i in range(len(self.vlm.language_model.layers))]
        return out.last_hidden_state, pad, kv

    # -- flow matching ---------------------------------------------------------------------
    @staticmethod
    def sample_time(n: int, device) -> Tensor:
        t = torch.distributions.Beta(torch.tensor(1.5, device=device), torch.tensor(1.0, device=device)).sample((n,))
        return t * 0.999 + 0.001

    def flow_loss(self, kv, pad, cond, cond_mask, actions, action_mask, base_actions=None, base_mask=None):
        """Masked FM loss of both heads; same tau, independent noise (sec. 9.2)."""
        n = actions.shape[0]
        time = self.sample_time(n, actions.device)
        tt = time[:, None, None]
        noise_u = torch.randn_like(actions)
        x_u = tt * noise_u + (1 - tt) * actions
        x_b = base_present = None
        if base_actions is not None and base_mask is not None and bool(base_mask.any()):
            noise_b = torch.randn_like(base_actions)
            x_b = tt * noise_b + (1 - tt) * base_actions
            base_present = base_mask.any(dim=-1)
        v_u, v_b = self.heads(kv, pad, cond, cond_mask, x_u, x_b, time, base_present)
        err_u = (v_u - (noise_u - actions)).pow(2)
        fm_u = L.masked_mean(err_u, action_mask[:, None, :].expand_as(err_u))
        fm_b = torch.zeros((), device=actions.device)
        if v_b is not None:
            err_b = (v_b - (noise_b - base_actions)).pow(2)
            fm_b = L.masked_mean(err_b, base_mask[:, None, :].expand_as(err_b))
        return fm_u, fm_b

    @torch.no_grad()
    def sample_actions(self, kv, pad, cond, cond_mask, *, base_dim_mask=None, num_steps: int = 10):
        n = pad.shape[0]
        dev = pad.device
        x_u = torch.randn(n, self.action_horizon, self.action_dim, device=dev)
        x_b = torch.randn_like(x_u) if base_dim_mask is not None else None
        base_present = base_dim_mask.any(dim=-1) if base_dim_mask is not None else None
        dt = -1.0 / num_steps
        t = 1.0
        while t >= -dt / 2:
            time = torch.full((n,), t, device=dev)
            v_u, v_b = self.heads(kv, pad, cond, cond_mask, x_u, x_b, time, base_present)
            x_u = x_u + dt * v_u
            if v_b is not None:
                x_b = x_b + dt * v_b
            t += dt
        return x_u, x_b

    # -- one policy update at inference --------------------------------------------------------
    @torch.no_grad()
    def infer(self, observation, step: StepInputs, memory: MemoryState, *, num_steps: int = 10):
        """Write the memory once for this NEW observation, then denoise (sec. 10.2)."""
        h, pad, kv = self.encode_prefix(observation, train=False)
        step = dataclasses.replace(step, h=h.float(), h_mask=pad)
        out = self.conditioner(step, memory)
        act_u, act_b = self.sample_actions(kv, pad, out.cond_tokens, out.cond_mask, num_steps=num_steps)
        return act_u, act_b, out

    # -- episode-sequence training ---------------------------------------------------------------
    def episode_loss(self, batch: dict, weights: LossWeights) -> tuple[Tensor, dict[str, float]]:
        """batch (see scripts/mobilebench/rmbench_episodes.py):
        observation   openpi Observation of the N ACTIVE update frames (flattened b-major)
        slot, update  [N] long: (episode slot, update index) of each active frame
        active        [B, U] bool
        actions       [N, H, A] normalised clean action chunks, action_mask [N, A]
        step          dict of [B, U, ...] StepInputs fields (everything except h / h_mask)
        labels        dict of [B, U, ...] label tensors
        """
        active = batch["active"]
        b, u = active.shape
        slot, upd = batch["slot"], batch["update"]
        h_flat, pad_flat, kv = self.encode_prefix(batch["observation"], train=True)
        n, lp, w = h_flat.shape
        h = h_flat.new_zeros(b, u, lp, w)
        h_mask = torch.zeros(b, u, lp, dtype=torch.bool, device=h.device)
        h[slot, upd], h_mask[slot, upd] = h_flat, pad_flat

        step = batch["step"]
        memory = self.conditioner.init_memory(b, h.device)
        outs: list[StepOutputs] = []
        for k in range(u):
            fields = {name: v[:, k] for name, v in step.items()}
            inp = StepInputs(h=h[:, k].float(), h_mask=h_mask[:, k], **fields)
            out = self.conditioner(inp, memory, with_readouts=True)
            memory = out.memory
            outs.append(out)

        cond = torch.stack([o.cond_tokens for o in outs], dim=1)[slot, upd]
        cond_mask = torch.stack([o.cond_mask for o in outs], dim=1)[slot, upd]
        fm_u, fm_b = self.flow_loss(
            kv, pad_flat, cond, cond_mask, batch["actions"], batch["action_mask"],
            batch.get("base_actions"), batch.get("base_mask"),
        )  # fmt: skip

        lab = batch["labels"]
        stack = lambda f: torch.stack([f(o) for o in outs], dim=1).flatten(0, 1)  # noqa: E731
        flat = {k: v.flatten(0, 1) for k, v in lab.items()}
        act = active.flatten()
        present = lambda m: m & act.view(-1, *([1] * (m.dim() - 1)))  # noqa: E731

        aff_obs = L.point_loss(stack(lambda o: o.a_obs.points), flat["aff_point"], flat["aff_obs_valid"],
                               present(flat["aff_present"]))  # fmt: skip
        aff_use = L.point_loss(stack(lambda o: o.a_use.points), flat["aff_point"], flat["aff_use_valid"],
                               present(flat["aff_present"]))  # fmt: skip
        valid = L.validity_loss(stack(lambda o: o.a_obs.valid_logits), flat["aff_obs_valid"],
                                present(flat["aff_present"])) + L.validity_loss(
            stack(lambda o: o.a_use.valid_logits), flat["aff_use_valid"], present(flat["aff_present"])
        )  # fmt: skip
        ee_mask = step["ee_mask"].flatten(0, 1)
        valid = valid + L.validity_loss(stack(lambda o: o.goal.valid_logits), flat["goal_valid"],
                                        present(flat["goal_present"]) & ee_mask)  # fmt: skip
        goal = L.goal_loss(stack(lambda o: o.goal.pos), stack(lambda o: o.goal.rot6d), flat["goal_pos"],
                           flat["goal_rot"], flat["goal_valid"], present(flat["goal_present"]), ee_mask)  # fmt: skip
        trace = L.trace_loss(stack(lambda o: o.trace[0]), stack(lambda o: o.trace[1]), stack(lambda o: o.trace[2]),
                             flat["trace_pos"], flat["trace_rot"], flat["trace_grip"],
                             present(flat["trace_avail"]))  # fmt: skip
        mode = L.mode_loss(stack(lambda o: o.mode_logits), flat["mode"], present(flat["mode_present"]))

        total = (
            weights.fm_upper * fm_u
            + weights.fm_base * fm_b
            + weights.aff_obs * aff_obs
            + weights.aff_use * aff_use
            + weights.valid * valid
            + weights.goal * goal
            + weights.trace * trace
            + weights.mode * mode
        )
        with torch.no_grad():
            pts = stack(lambda o: o.a_use.points)[:, 1]
            m = present(flat["aff_present"][:, 1] & flat["aff_use_valid"][:, 1])
            aff_err = ((pts - flat["aff_point"][:, 1]).norm(dim=-1) * m).sum() / m.sum().clamp_min(1)
            gm = present(flat["goal_present"] & flat["goal_valid"]) & ee_mask
            g_err = ((stack(lambda o: o.goal.pos) - flat["goal_pos"]).norm(dim=-1) * gm).sum() / gm.sum().clamp_min(1)
        logs = {
            "loss": total.item(),
            "fm_upper": fm_u.item(),
            "fm_base": fm_b.item(),
            "aff_obs": aff_obs.item(),
            "aff_use": aff_use.item(),
            "valid": valid.item(),
            "goal": goal.item(),
            "trace": trace.item(),
            "mode": mode.item(),
            "aff_use_obj_err_m": aff_err.item(),
            "goal_pos_err_m": g_err.item(),
            "updates_per_episode": act.sum().item() / b,
        }
        return total, logs


def trainable_parameter_groups(model: MobileBenchPi05, lr_pretrained: float, lr_new: float):
    """Pretrained expert weights get the fine-tuning LR, everything new the larger one."""
    pretrained, new = [], []
    pre_ids = set()
    for br in (model.heads.upper, model.heads.base):
        for mod in (br.expert, br.action_in, br.action_out, br.time_in, br.time_out):
            pre_ids |= {id(p) for p in mod.parameters()}
    pre_ids |= {id(p) for p in model.vlm.parameters()}  # LoRA / SigLIP / projector when trained
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (pretrained if id(p) in pre_ids else new).append(p)
    return [
        {"params": pretrained, "lr": lr_pretrained, "name": "pretrained"},
        {"params": new, "lr": lr_new, "name": "new"},
    ]


__all__ = ["LossWeights", "MobileBenchPi05", "trainable_parameter_groups"]
