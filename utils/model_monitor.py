"""
model_monitor.py
----------------
Drop-in hooks to log everything that affects FiLM modulation and MoCo
contrastive learning. Attach once; works across all forward passes.

Usage:
    from model_monitor import attach_monitor, detach_monitor, print_summary

    monitor = attach_monitor(model, log_every=50, writer=tb_writer)  # tb_writer optional
    # ... training loop ...
    print_summary(monitor)
    detach_monitor(monitor)
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from typing import Optional


# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

class _MonitorState:
    def __init__(self, log_every: int, writer=None):
        self.log_every = log_every
        self.writer = writer          # optional TensorBoard SummaryWriter
        self.step = 0
        self.hooks = []
        self.buf = defaultdict(list)  # tag -> list of scalar values this window

    def record(self, tag: str, value: float):
        self.buf[tag].append(value)

    def maybe_flush(self, step: int):
        if step % self.log_every != 0:
            return
        lines = [f"\n{'='*60}", f"  Monitor @ step {step}", f"{'='*60}"]
        for tag in sorted(self.buf):
            vals = self.buf[tag]
            mean = np.mean(vals)
            std  = np.std(vals)
            mn   = np.min(vals)
            mx   = np.max(vals)
            lines.append(f"  {tag:<48}  mean={mean:+.4f}  std={std:.4f}  "
                         f"min={mn:+.4f}  max={mx:+.4f}")
            if self.writer is not None:
                self.writer.add_scalar(tag, mean, step)
        print("\n".join(lines))
        self.buf.clear()


# ---------------------------------------------------------------------------
# FiLM hooks  (ResidualGatedFiLM)
# ---------------------------------------------------------------------------

def _make_film_hook(state: _MonitorState, layer_idx: int):
    """
    Registers on ResidualGatedFiLM.forward.
    We capture gamma, beta, gate (from proj output) and x stats.
    """
    def hook(module, inputs, output):
        # inputs[0] = x  (B, T, D)
        # inputs[1] = time_embed  (B, D_time)
        x          = inputs[0].detach().float()
        time_embed = inputs[1].detach().float()
        out        = output.detach().float()

        # --- raw proj outputs ---
        with torch.no_grad():
            raw = module.proj(time_embed)          # (B, 3*D)
            gamma, beta, gate_pre = raw.chunk(3, dim=-1)
            gate = torch.sigmoid(gate_pre)

        tag = f"film/layer{layer_idx}"
        state.record(f"{tag}/gamma_mean",   gamma.mean().item())
        state.record(f"{tag}/gamma_std",    gamma.std().item())
        state.record(f"{tag}/gamma_abs_max",gamma.abs().max().item())

        state.record(f"{tag}/beta_mean",    beta.mean().item())
        state.record(f"{tag}/beta_std",     beta.std().item())
        state.record(f"{tag}/beta_abs_max", beta.abs().max().item())

        state.record(f"{tag}/gate_mean",    gate.mean().item())
        state.record(f"{tag}/gate_min",     gate.min().item())
        state.record(f"{tag}/gate_max",     gate.max().item())

        # --- input / output norms ---
        state.record(f"{tag}/x_in_norm",    x.norm(dim=-1).mean().item())
        state.record(f"{tag}/x_out_norm",   out.norm(dim=-1).mean().item())

        # --- time_embed stats (drives every FiLM layer) ---
        state.record(f"film/time_embed_norm", time_embed.norm(dim=-1).mean().item())
        state.record(f"film/time_embed_std",  time_embed.std().item())

    return hook


# ---------------------------------------------------------------------------
# MoCo hooks
# ---------------------------------------------------------------------------

def _make_moco_forward_hook(state: _MonitorState):
    """
    Registers on MoCo.forward.
    Captures l_pos, l_neg, soft_weights, queue_y, queue saturation.
    """
    def hook(module, inputs, output):
        logits, soft_weights, h_msm = output

        if logits is None:
            return  # eval mode - no contrastive outputs

        logits_d = logits.detach().float()
        # logits[:, 0] = l_pos (positive pair), logits[:, 1:] = l_neg
        l_pos = logits_d[:, 0]
        l_neg = logits_d[:, 1:]

        state.record("moco/l_pos_mean",      l_pos.mean().item())
        state.record("moco/l_pos_std",       l_pos.std().item())
        state.record("moco/l_neg_mean",      l_neg.mean().item())
        state.record("moco/l_neg_std",       l_neg.std().item())
        state.record("moco/l_pos_minus_neg", (l_pos.mean() - l_neg.mean()).item())

        if soft_weights is not None:
            sw = soft_weights.detach().float()
            state.record("moco/soft_weight_mean", sw.mean().item())
            state.record("moco/soft_weight_min",  sw.min().item())
            state.record("moco/soft_weight_max",  sw.max().item())
            # what fraction of negatives are "hard" (weight > 0.5)?
            state.record("moco/hard_neg_frac",    (sw > 0.5).float().mean().item())

        # queue fill: fraction of slots that are non-zero
        qy = module.queue_y.detach().float()
        state.record("moco/queue_y_mean",       qy.mean().item())
        state.record("moco/queue_y_std",        qy.std().item())
        state.record("moco/queue_zero_frac",    (qy == 0).float().mean().item())
        state.record("moco/queue_ptr",          float(module.queue_ptr.item()))

        # queue embedding diversity (mean pairwise cosine ~ -1/K for uniform)
        q = module.queue.detach().float()  # (D, K)
        q_norm = nn.functional.normalize(q, dim=0)
        state.record("moco/queue_embed_norm_mean", q.norm(dim=0).mean().item())

    return hook


def _make_moco_loss_hook(state: _MonitorState, model: nn.Module):
    """
    Wraps model.contrastive_loss to log the scalar loss and its components.
    Returns the patched method; caller should restore original on detach.
    """
    original_loss = model.contrastive_loss

    def patched_loss(logits, soft_weights, epoch, max_epoch):
        # --- replicate loss internals for logging ---
        if logits is not None:
            ld = logits.detach().float()
            log_probs = nn.functional.log_softmax(ld, dim=1)
            l_pos_loss = -log_probs[:, 0]
            state.record("moco/loss_l_pos", l_pos_loss.mean().item())

            if soft_weights is not None:
                queue_size = model.contrast_enc.moco.queue_size
                sw = soft_weights.detach().float()
                l_neg_loss = -(sw * log_probs[:, 1:]).sum(dim=1) / queue_size
                alpha = min(1.0, (np.exp(epoch / max_epoch) - 1) / (np.e - 1))
                state.record("moco/loss_l_neg",   l_neg_loss.mean().item())
                state.record("moco/loss_alpha",   alpha)
                state.record("moco/loss_total_approx",
                             (l_pos_loss + alpha * l_neg_loss).mean().item())

        return original_loss(logits, soft_weights, epoch, max_epoch)

    model.contrastive_loss = patched_loss
    return original_loss   # hand back so we can restore


def _make_projector_hook(state: _MonitorState, name: str):
    """Logs query/key projector output norms (collapsed representations → mean~0)."""
    def hook(module, inputs, output):
        out = output.detach().float()
        state.record(f"moco/{name}_proj_norm", out.norm(dim=-1).mean().item())
        state.record(f"moco/{name}_proj_std",  out.std().item())
    return hook


def _make_key_encoder_hook(state: _MonitorState):
    """Logs momentum of the key encoder (param drift from query encoder)."""
    def hook(module, inputs, output):
        pass   # parameter-level diff logged in attach_monitor below
    return hook


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def attach_monitor(model: nn.Module, log_every: int = 50, writer=None) -> _MonitorState:
    """
    Attach all monitoring hooks to *model*.

    Parameters
    ----------
    model       : Cl_TTE instance
    log_every   : print/flush stats every N forward passes
    writer      : optional torch.utils.tensorboard.SummaryWriter

    Returns
    -------
    state : _MonitorState  (pass to detach_monitor / print_summary)
    """
    state = _MonitorState(log_every=log_every, writer=writer)
    moco  = model.contrast_enc.moco
    film  = model.film

    # --- FiLM: one hook per ResidualGatedFiLM layer ---
    for i, layer in enumerate(film.layers):
        rgfilm = layer["film"]  # ResidualGatedFiLM instance
        h = rgfilm.register_forward_hook(_make_film_hook(state, i))
        state.hooks.append(h)

    # --- MoCo forward ---
    h = moco.register_forward_hook(_make_moco_forward_hook(state))
    state.hooks.append(h)

    # --- Projectors ---
    h = moco.mlp_q.register_forward_hook(_make_projector_hook(state, "query"))
    state.hooks.append(h)
    h = moco.mlp_k.register_forward_hook(_make_projector_hook(state, "key"))
    state.hooks.append(h)

    # --- Wrap contrastive_loss ---
    state._original_loss = _make_moco_loss_hook(state, model)
    state._model_ref = model

    # --- Step counter: hook on the top-level model forward ---
    def _count_hook(module, inputs, output):
        state.step += 1
        state.maybe_flush(state.step)

    h = model.register_forward_hook(_count_hook)
    state.hooks.append(h)

    # --- Log tau_I and temperature as constants ---
    print(f"[monitor] tau_I={moco.tau_I}  temperature={moco.temperature}  "
          f"queue_size={moco.queue_size}  mmt={moco.mmt}")
    print(f"[monitor] Attached {len(state.hooks)} hooks, log_every={log_every}")

    return state


def detach_monitor(state: _MonitorState):
    """Remove all hooks and restore patched methods."""
    for h in state.hooks:
        h.remove()
    state.hooks.clear()

    if hasattr(state, "_original_loss") and hasattr(state, "_model_ref"):
        state._model_ref.contrastive_loss = state._original_loss

    print("[monitor] All hooks detached.")


def print_summary(state: _MonitorState):
    """Print whatever is left in the buffer (end-of-epoch flush)."""
    if not state.buf:
        print("[monitor] Buffer empty — nothing to summarize.")
        return
    state.maybe_flush(state.step + (state.log_every - state.step % state.log_every))