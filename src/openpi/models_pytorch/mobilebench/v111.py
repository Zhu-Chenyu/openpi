"""MobileBench v1.11: memory tokens INTO the VLM, MLP memory writers, Kinematics outside the VLM.

Design doc: docs/MobileBench_Method_v1_11_Kinematics_Memory_VLM_Complete.md. One policy update:

    T^F, T^S = P^in(LN(old M^F)), P^in(LN(old M^S)) + type + slot      (sec. 4.2)
    T^mot    = MLP_mot(actual motion since the last update)            (sec. 4.3)
    Y_t      = VLM([image ; text/state ; T^mot ; T^F ; T^S]), full bidirectional prefix (sec. 4.4-4.5)
    H_t      = Y_t at the observation / instruction positions          (sec. 4.6)
    A_hat    = MLP_Aff(Read(H_t))                                       (sec. 5.2)
    h_t      = P_h(MaskedMean(H_t))
    M^F_t    = (1 - a_F) M^F + a_F MLP_F([h_t ; e^{F,state} ; LN(m^F_i) ; s^F_i])   every update
    M^S_t    = (1 - a_S) M^S + a_S MLP_S([h_t ; LN(m^S_i) ; s^S_i])     on t % K == 0, else kept
    Z_t      = [P^act_F M^F_t ; P^act_S M^S_t ; W_t ; e_q ; e_g ; e_b] (+ H_t, see below)   (sec. 5.1)
    v^U, v^B = DualActionHeads(Z_t, noisy actions, tau)

Losses (sec. 6.2): L_act^U + L_act^B + lambda_A L_aff + lambda_S L_slow-text. No fast, trace,
goal or mode loss. The slow-text loss reads ONLY the new M^S and is applied on slow-write updates.

Implementation choice: the P^act_H H_t part of Z_t is realised by pi0.5's own path -- the action
expert attends to the VLM's per-layer K/V at the observation positions (the memory / motion
positions are masked out for the heads, as H_t = Select_obs(Y_t)). Those K/V were computed by
layers that read the memory tokens, so they carry the history; keeping this path keeps the
pretrained VLM -> expert interface instead of re-learning it from a projected copy of H_t.

Training steps through an episode's updates in order: images and text are embedded once for all
frames, then each update's VLM pass reads the memory written by the previous update, so the
loss gradient reaches the memory projections and the previous writers THROUGH the VLM
(sec. 6.4; frozen VLM weights are not a no_grad VLM).
"""

import dataclasses
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

from openpi.models_pytorch import preprocessing_pytorch as _preprocessing
from openpi.models_pytorch.mobilebench import losses as L
from openpi.models_pytorch.mobilebench.action_heads import DualActionHeads
from openpi.models_pytorch.mobilebench.layers import QueryDecoder
from openpi.models_pytorch.mobilebench.layers import masked_token_mean
from openpi.models_pytorch.mobilebench.layers import mlp
from openpi.models_pytorch.mobilebench.lora import apply_gemma_lora
from openpi.models_pytorch.mobilebench.model import MobileBenchPi05
from openpi.models_pytorch.mobilebench.workspace import workspace_features

NAV, OBJ = 0, 1
COND_SOURCES = ("fast", "slow", "workspace", "state", "gripper", "base")


@dataclasses.dataclass(frozen=True)
class V111Config:
    vlm_width: int = 2048  # D, PaliGemma gemma_2b
    d_m: int = 256  # external memory width
    d_z: int = 256  # action-condition width
    d_h: int = 256  # pooled writer input h_t
    writer_hidden: int = 512
    num_heads: int = 4
    num_fast_slots: int = 32  # N_F
    num_slow_slots: int = 16  # N_S
    fast_alpha: float = 0.5
    slow_alpha: float = 0.5
    slow_write_every: int = 2  # K: slow writes on updates 0, K, 2K, ...
    motion_dim: int = 16  # executed EEF / gripper increments + dt + first-update flag
    motion_tokens: int = 2  # N_mot
    motion_hidden: int = 1024
    state_dim: int = 32
    gripper_dim: int = 8
    base_desc_dim: int = 8
    # x^W = [p ; rho(R) ; p - p_E ; rho(R_E^T R)] (sec. 2.3, with the optional relative geometry).
    workspace_feat_dim: int = 18
    phase_text_dim: int = 512  # frozen text-encoder width (CLIP ViT-B/32 text)
    phase_tau: float = 0.07

    @property
    def exec_dim(self) -> int:  # name used by rmbench_episodes.step_inputs
        return self.motion_dim


@dataclasses.dataclass
class MemoryV111:
    fast: Tensor  # [B, N_F, d_m]
    slow: Tensor  # [B, N_S, d_m]
    step: Tensor  # [B] long: policy updates since this slot's episode started

    def detach(self) -> "MemoryV111":
        """TBPTT boundary: keep the values, cut the graph. The memory is NOT cleared."""
        return MemoryV111(self.fast.detach(), self.slow.detach(), self.step.clone())


@dataclasses.dataclass
class UpdateOut:
    memory: MemoryV111  # after this update's writes (all B slots)
    slow_written: Tensor  # [n]
    cond: Tensor  # [n, N_Z, d_z]
    cond_mask: Tensor  # [n, N_Z]
    aff_points: Tensor  # [n, 2, 3]  NAV / OBJ
    aff_logits: Tensor  # [n, 2]
    phase_z: Tensor  # [n, phase_text_dim] normalised slow readout
    kv: list  # per-layer prefix (K, V) [n, 1, L_p, hd]
    head_pad: Tensor  # [n, L_p] prefix keys the action heads read (observation positions only)


def _masked_flags(x: Tensor, mask: Tensor) -> Tensor:
    """[x * mask ; mask]: a missing value is zeroed AND flagged, so 0 never means "exists"."""
    m = mask.to(x.dtype)
    return torch.cat([x * m, m], dim=-1)


class MobileBenchV111(nn.Module):
    # Flow matching, the Gemma layer and action sampling are unchanged from the v1.2 model.
    sample_time = staticmethod(MobileBenchPi05.sample_time)
    _llm_layer = staticmethod(MobileBenchPi05._llm_layer)  # noqa: SLF001
    flow_loss = MobileBenchPi05.flow_loss
    sample_actions = MobileBenchPi05.sample_actions
    _enable_siglip_checkpointing = MobileBenchPi05._enable_siglip_checkpointing  # noqa: SLF001

    def __init__(self, pi0, cfg: V111Config, *, vlm_lora: bool = False, expert_lora: bool = False):
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
        self.heads = DualActionHeads.from_pi05(pi0, cfg.d_z).float()
        if expert_lora:
            for br in (self.heads.upper, self.heads.base):
                apply_gemma_lora(br.expert, rank=32, alpha=32.0)

        big, dm, dz, dh = cfg.vlm_width, cfg.d_m, cfg.d_z, cfg.d_h
        nf, ns = cfg.num_fast_slots, cfg.num_slow_slots
        # -- memory -> VLM input tokens (sec. 4.2). Zero-initialised output projections: at step 0
        # the memory / motion tokens are zero vectors, so the pretrained prefix is barely disturbed.
        self.in_fast = nn.Sequential(nn.LayerNorm(dm), nn.Linear(dm, big))
        self.in_slow = nn.Sequential(nn.LayerNorm(dm), nn.Linear(dm, big))
        self.in_type = nn.Parameter(torch.randn(1, 3, 1, big) * 0.02)  # motion / fast / slow
        self.in_slot_fast = nn.Parameter(torch.randn(1, nf, big) * 0.02)
        self.in_slot_slow = nn.Parameter(torch.randn(1, ns, big) * 0.02)
        self.motion_enc = nn.Sequential(
            nn.Linear(2 * cfg.motion_dim, cfg.motion_hidden), nn.GELU(),
            nn.Linear(cfg.motion_hidden, cfg.motion_tokens * big),
        )  # fmt: skip
        for lin in (self.in_fast[1], self.in_slow[1], self.motion_enc[2]):
            nn.init.zeros_(lin.weight), nn.init.zeros_(lin.bias)
        # -- external memory and MLP writers (sec. 3)
        self.fast_init = nn.Parameter(torch.randn(1, nf, dm) * 0.02)
        self.slow_init = nn.Parameter(torch.randn(1, ns, dm) * 0.02)
        self.slot_fast = nn.Parameter(torch.randn(1, nf, dm) * 0.02)  # s_i^F
        self.slot_slow = nn.Parameter(torch.randn(1, ns, dm) * 0.02)  # s_i^S
        self.p_h = nn.Sequential(nn.LayerNorm(big), nn.Linear(big, dh))
        self.fast_state = nn.Linear(2 * cfg.state_dim + 2 * cfg.motion_dim, dh)  # e^{F,state}
        self.ln_fast, self.ln_slow = nn.LayerNorm(dm), nn.LayerNorm(dm)
        self.writer_fast = mlp(dh + dh + dm + dm, cfg.writer_hidden, dm)
        self.writer_slow = mlp(dh + dm + dm, cfg.writer_hidden, dm)
        # -- Kinematics encoder, outside the VLM (sec. 2)
        self.enc_workspace = mlp(cfg.workspace_feat_dim, dz, dz, norm=True)  # shared MLP_W per pose
        self.enc_state = mlp(2 * cfg.state_dim, dz, dz)
        self.enc_gripper = mlp(cfg.gripper_dim, dz, dz)
        self.enc_base = mlp(cfg.base_desc_dim, dz, dz)
        # -- fusion into Z_t (sec. 5.1)
        self.act_fast = nn.Linear(dm, dz)
        self.act_slow = nn.Linear(dm, dz)
        self.cond_type = nn.Embedding(len(COND_SOURCES), dz)
        # -- affordance from H_t (sec. 5.2): light attention read, one query per type (NAV / OBJ)
        self.aff_in = nn.Sequential(nn.LayerNorm(big), nn.Linear(big, dz))
        self.aff_queries = nn.Parameter(torch.randn(1, 2, dz) * 0.02)
        self.aff_read = QueryDecoder(dz, cfg.num_heads, 1)
        self.aff_point = mlp(dz, dz, 3)
        self.aff_valid = mlp(dz, dz, 1)
        # -- slow semantic readout, reads ONLY the new slow group (sec. 3.6)
        self.phase_query = nn.Parameter(torch.randn(1, 1, dm) * 0.02)
        self.phase_read = QueryDecoder(dm, cfg.num_heads, 1)
        self.phase_out = nn.Linear(dm, cfg.phase_text_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.vlm_lora:
            self.vlm.eval()
        return self

    # -- memory state ------------------------------------------------------------------------------
    def init_memory(self, batch: int, device=None) -> MemoryV111:
        device = device or self.fast_init.device
        return MemoryV111(
            self.fast_init.expand(batch, -1, -1).to(device),
            self.slow_init.expand(batch, -1, -1).to(device),
            torch.zeros(batch, dtype=torch.long, device=device),
        )

    def reset_memory(self, mem: MemoryV111, new_episode: Tensor) -> MemoryV111:
        m = new_episode.to(torch.bool)
        b = m.shape[0]
        return MemoryV111(
            torch.where(m[:, None, None], self.fast_init.expand(b, -1, -1), mem.fast),
            torch.where(m[:, None, None], self.slow_init.expand(b, -1, -1), mem.slow),
            torch.where(m, torch.zeros_like(mem.step), mem.step),
        )

    # -- VLM ---------------------------------------------------------------------------------------
    def embed_observation(self, observation, *, train: bool) -> tuple[Tensor, Tensor]:
        """Image + instruction/state embeddings of every frame: ([N, L_obs, D], pad [N, L_obs])."""
        with torch.set_grad_enabled(self.vlm_lora and torch.is_grad_enabled()):
            obs = _preprocessing.preprocess_observation_pytorch(observation, train=train)
            embs, pads = [], []
            for key in obs.images:
                img, m = obs.images[key], obs.image_masks[key]
                if not bool(m.any()):  # e.g. RMBench's missing right wrist: padding only, skip SigLIP
                    continue
                e = self.vlm.model.get_image_features(img)
                embs.append(e)
                pads.append(m[:, None].expand(e.shape[0], e.shape[1]))
            used = max(int(obs.tokenized_prompt_mask.sum(dim=1).max()), 1)
            lm = self.vlm.language_model
            lang = lm.embed_tokens(obs.tokenized_prompt[:, :used])
            embs.append(lang * math.sqrt(lang.shape[-1]))
            pads.append(obs.tokenized_prompt_mask[:, :used])
            dtype = lm.layers[0].self_attn.q_proj.weight.dtype
            return torch.cat([e.to(dtype) for e in embs], dim=1), torch.cat(pads, dim=1)

    def memory_tokens(self, fast: Tensor, slow: Tensor, motion: Tensor, motion_mask: Tensor) -> Tensor:
        """[T^mot ; T^F ; T^S] in the VLM width, [n, N_mot + N_F + N_S, D]."""
        n, big = fast.shape[0], self.cfg.vlm_width
        t_mot = self.motion_enc(_masked_flags(motion, motion_mask)).view(n, self.cfg.motion_tokens, big)
        t_f = self.in_fast(fast) + self.in_slot_fast
        t_s = self.in_slow(slow) + self.in_slot_slow
        e_mot, e_f, e_s = self.in_type.unbind(dim=1)
        return torch.cat([t_mot + e_mot, t_f + e_f, t_s + e_s], dim=1)

    def run_vlm(self, x: Tensor, pad: Tensor) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        """Gemma layers over the full bidirectional prefix (valid keys only). Returns (Y_t, per-layer K/V).

        In training each layer is checkpointed (only layer inputs + the single-head K/V are kept),
        and gradients flow to the inputs -- the memory tokens -- even when the VLM weights are frozen.
        """
        lm = self.vlm.language_model
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
        y, _ = lm.norm(x)
        return y, kv

    # -- one policy update ----------------------------------------------------------------------------
    def kinematics(self, st: dict) -> tuple[Tensor, Tensor]:
        """E^kin = [W_t ; e_q ; e_g ; e_b] in d_z, with the workspace padding mask. st: per-frame step fields."""
        feats = workspace_features(
            st["lib_pos"], st["lib_rot"], st["eef_pos"], st["eef_rot"], st["lib_ee"],
            torch.zeros_like(st["lib_margin"]), st["lib_margin"],
        )[..., : self.cfg.workspace_feat_dim]  # fmt: skip
        w = self.enc_workspace(feats)
        e_q = self.enc_state(_masked_flags(st["state"], st["state_mask"]))[:, None]
        e_g = self.enc_gripper(st["gripper_desc"])[:, None]
        e_b = self.enc_base(st["base_desc"])[:, None]
        return torch.cat([w, e_q, e_g, e_b], dim=1), st["ws_mask"]

    def update(self, obs_emb: Tensor, obs_pad: Tensor, st: dict, mem: MemoryV111, slots: Tensor) -> UpdateOut:
        """One policy update for frames `slots` (indices into the memory batch); `mem` already reset.

        obs_emb / obs_pad: [n, L_obs, D] embeddings of these frames; st: their step fields ([n, ...]).
        """
        cfg = self.cfg
        n, l_obs = obs_pad.shape
        old_f, old_s, step = mem.fast[slots], mem.slow[slots], mem.step[slots]
        extra = self.memory_tokens(old_f, old_s, st["exec_increment"], st["exec_mask"])
        x = torch.cat([obs_emb, extra.to(obs_emb.dtype)], dim=1)
        ones = torch.ones(n, extra.shape[1], dtype=torch.bool, device=x.device)
        y, kv = self.run_vlm(x, torch.cat([obs_pad, ones], dim=1))
        h_full = y[:, :l_obs].float()  # H_t = Select_obs(Y_t)

        # Affordance from H_t.
        a = self.aff_read(self.aff_queries.expand(n, -1, -1), self.aff_in(h_full), obs_pad)
        aff_points, aff_logits = self.aff_point(a), self.aff_valid(a).squeeze(-1)

        # External MLP writers.
        h = self.p_h(masked_token_mean(h_full, obs_pad, dim=1))  # [n, d_h]
        e_state = self.fast_state(
            torch.cat([_masked_flags(st["state"], st["state_mask"]), _masked_flags(st["exec_increment"], st["exec_mask"])], -1)
        )  # fmt: skip
        bcast = lambda v, k: v[:, None].expand(-1, k, -1)  # noqa: E731
        cand_f = self.writer_fast(torch.cat([bcast(h, cfg.num_fast_slots), bcast(e_state, cfg.num_fast_slots),
                                             self.ln_fast(old_f), self.slot_fast.expand(n, -1, -1)], -1))  # fmt: skip
        new_f = (1 - cfg.fast_alpha) * old_f + cfg.fast_alpha * cand_f
        write = (step % cfg.slow_write_every) == 0
        if bool(write.any()):
            cand_s = self.writer_slow(torch.cat([bcast(h, cfg.num_slow_slots), self.ln_slow(old_s),
                                                 self.slot_slow.expand(n, -1, -1)], -1))  # fmt: skip
            new_s = torch.where(write[:, None, None], (1 - cfg.slow_alpha) * old_s + cfg.slow_alpha * cand_s, old_s)
        else:
            new_s = old_s
        mem = MemoryV111(
            mem.fast.index_put((slots,), new_f), mem.slow.index_put((slots,), new_s), mem.step.index_put((slots,), step + 1)
        )

        # Slow semantic readout (training loss / logging only).
        z = self.phase_read(self.phase_query.expand(n, -1, -1), new_s)[:, 0]
        phase_z = F.normalize(self.phase_out(z), dim=-1)

        # Z_t = [P_F M^F ; P_S M^S ; W ; e_q ; e_g ; e_b] (+ H_t through the prefix K/V).
        kin, ws_mask = self.kinematics(st)
        groups = [self.act_fast(new_f), self.act_slow(new_s), kin]
        sizes = [cfg.num_fast_slots, cfg.num_slow_slots, ws_mask.shape[1], 1, 1, 1]
        types = torch.cat([torch.full((k,), i, dtype=torch.long, device=x.device) for i, k in enumerate(sizes)])
        cond = torch.cat(groups, dim=1) + self.cond_type(types)[None]
        on = lambda k: torch.ones(n, k, dtype=torch.bool, device=x.device)  # noqa: E731
        cond_mask = torch.cat([on(cfg.num_fast_slots + cfg.num_slow_slots), ws_mask, on(3)], dim=1)
        head_pad = torch.cat([obs_pad, torch.zeros_like(ones)], dim=1)
        return UpdateOut(mem, write, cond, cond_mask, aff_points, aff_logits, phase_z, kv, head_pad)

    # -- inference -------------------------------------------------------------------------------------
    @torch.no_grad()
    def infer(self, observation, st: dict, memory: MemoryV111, *, num_steps: int = 10, base_dim_mask=None):
        """One new observation: read the old memory in the VLM, write it once, then denoise with fixed Z_t.

        base_dim_mask [n, action_dim]: the base head's valid dims (None = fixed base, no base chunk)."""
        memory = self.reset_memory(memory, st["new_episode"])
        emb, pad = self.embed_observation(observation, train=False)
        slots = torch.arange(emb.shape[0], device=emb.device)
        out = self.update(emb, pad, st, memory, slots)
        act_u, act_b = self.sample_actions(
            out.kv, out.head_pad, out.cond, out.cond_mask, base_dim_mask=base_dim_mask, num_steps=num_steps
        )
        return act_u, act_b, out

    # -- episode-sequence training -----------------------------------------------------------------------
    def episode_loss(self, batch: dict, weights: "LossWeightsV111", phase_text: Tensor, memory: MemoryV111 | None = None,
                     phase_group: tuple[str, Tensor] | None = None):  # fmt: skip
        """Same batch layout as MobileBenchPi05.episode_loss (scripts/mobilebench/rmbench_episodes.py).

        phase_text [T, phase_text_dim]: the frozen, normalised text embeddings that labels["phase_id"]
        indexes. phase_group (name, [T] group id of every sentence) adds a slow_<name>_acc log (default:
        put_back_block's 7 phases x 4 mats, "mat" = id % 4). Returns (loss, logs, memory after the window,
        DETACHED).
        """
        active = batch["active"]
        b, u = active.shape
        slot, upd = batch["slot"], batch["update"]
        obs_emb, obs_pad = self.embed_observation(batch["observation"], train=True)
        step, lab = batch["step"], batch["labels"]
        mem = memory if memory is not None else self.init_memory(b, obs_emb.device)

        order, outs = [], []
        for k in range(u):
            idx = (upd == k).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            s = slot[idx]
            mem = self.reset_memory(mem, step["new_episode"][:, k] & active[:, k])
            st = {name: v[s, k] for name, v in step.items()}
            out = self.update(obs_emb[idx], obs_pad[idx], st, mem, s)
            mem = out.memory
            order.append(idx), outs.append((out, s, torch.full_like(s, k)))
        inv = torch.argsort(torch.cat(order))  # back to the flat (b-major) frame order of batch["actions"]
        cat = lambda f: torch.cat([f(o) for o, _, _ in outs])[inv]  # noqa: E731
        kv = [(cat(lambda o, i=i: o.kv[i][0]), cat(lambda o, i=i: o.kv[i][1])) for i in range(len(outs[0][0].kv))]
        fm_u, fm_b = self.flow_loss(
            kv, cat(lambda o: o.head_pad), cat(lambda o: o.cond), cat(lambda o: o.cond_mask),
            batch["actions"], batch["action_mask"], batch.get("base_actions"), batch.get("base_mask"),
        )  # fmt: skip

        s_all = torch.cat([s for _, s, _ in outs])[inv]
        k_all = torch.cat([k for _, _, k in outs])[inv]
        pick = lambda name: lab[name][s_all, k_all]  # noqa: E731
        pts, logits = cat(lambda o: o.aff_points), cat(lambda o: o.aff_logits)
        aff = L.point_loss(pts, pick("aff_point"), pick("aff_use_valid"), pick("aff_present"))
        valid = L.validity_loss(logits, pick("aff_use_valid"), pick("aff_present"))
        written = cat(lambda o: o.slow_written)
        ph_mask = written & pick("phase_present")
        ph_logits = cat(lambda o: o.phase_z) @ phase_text.T / self.cfg.phase_tau
        ph_ce = F.cross_entropy(ph_logits, pick("phase_id").clamp_min(0), reduction="none")
        slow_text = L.masked_mean(ph_ce, ph_mask)

        total = (
            weights.fm_upper * fm_u
            + weights.fm_base * fm_b
            + weights.aff * (aff + weights.aff_valid * valid)
            + weights.slow_text * slow_text
        )
        with torch.no_grad():
            m = pick("aff_present")[:, OBJ] & pick("aff_use_valid")[:, OBJ]
            aff_err = ((pts[:, OBJ] - pick("aff_point")[:, OBJ]).norm(dim=-1) * m).sum() / m.sum().clamp_min(1)
            ph_acc = ((ph_logits.argmax(-1) == pick("phase_id")) & ph_mask).sum() / ph_mask.sum().clamp_min(1)
            if phase_group is None:  # put_back_block: 7 phases x 4 mats (annotate_phases.py)
                ids = torch.arange(phase_text.shape[0], device=phase_text.device)
                phase_group = ("mat", ids % (phase_text.shape[0] // 7))
            g_name, g = phase_group
            g_acc = ((g[ph_logits.argmax(-1)] == g[pick("phase_id")]) & ph_mask).sum() / ph_mask.sum().clamp_min(1)
        logs = {
            "loss": total.item(),
            "fm_upper": fm_u.item(),
            "fm_base": fm_b.item(),
            "aff": aff.item(),
            "aff_valid": valid.item(),
            "slow_text": slow_text.item(),
            "aff_obj_err_m": aff_err.item(),
            "slow_text_acc": ph_acc.item(),
            f"slow_{g_name}_acc": g_acc.item(),
            "updates_per_slot": active.sum().item() / b,
        }
        return total, logs, mem.detach()


@dataclasses.dataclass
class LossWeightsV111:
    fm_upper: float = 1.0
    fm_base: float = 1.0
    aff: float = 1.0  # lambda_A
    aff_valid: float = 0.1  # validity BCE inside L_aff
    slow_text: float = 0.05  # lambda_S


__all__ = ["COND_SOURCES", "LossWeightsV111", "MemoryV111", "MobileBenchV111", "V111Config"]
