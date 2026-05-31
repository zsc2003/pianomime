"""Conditional flow matching utilities for PianoMime generalist policies.

This module keeps the original PianoMime ConditionalUnet1D interface

    model(sample=x_t, timestep=t, global_cond=cond)

but changes the interpretation from DDPM epsilon prediction to velocity prediction.

Training objective used here:
    x0 ~ N(0, I)
    x1 = normalized expert action chunk
    t  ~ Uniform(0, 1) or LogitNormal
    x_t = (1 - t) * x0 + t * x1
    target velocity = x1 - x0
    loss = MSE(v_theta(x_t, t, cond), x1 - x0)

Sampling integrates dx/dt = v_theta(x, t, cond) from t=0 to t=1.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class FlowMatchingConfig:
    """Hyperparameters for straight-line conditional flow matching."""

    # The original DDPM model saw timesteps in [0, 100). Keeping this scale lets us
    # reuse the same sinusoidal timestep embedding without changing network.py.
    time_scale: float = 100.0

    # "uniform" is the cleanest baseline. "logit_normal" samples more middle times.
    time_sampler: str = "uniform"
    logit_normal_mean: float = 0.0
    logit_normal_std: float = 1.0

    # Avoid exact 0 or 1 during training.
    t_eps: float = 1.0e-5

    # Optional nonzero-noise endpoint. Leave at 0.0 for standard rectified-flow style.
    sigma_min: float = 0.0

    @property
    def sampler_name(self) -> str:
        name = self.time_sampler
        if name in {"logitnormal", "logit-normal"}:
            name = "logit_normal"
        return name


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sanitize_name(path: str) -> str:
    return Path(path.rstrip("/\\")).stem


def resolve_existing_path(candidates: Sequence[Optional[str]], *, what: str = "file") -> str:
    """Return the first existing path in candidates, raising a helpful error otherwise."""
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return str(candidate)
    tried = ", ".join(str(x) for x in candidates if x)
    raise FileNotFoundError(f"Could not find {what}. Tried: {tried}")


def _expand_time(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return t.view(t.shape[0], *([1] * (x.ndim - 1)))


def sample_time(
    batch_size: int,
    device: torch.device,
    *,
    sampler: str = "uniform",
    eps: float = 1.0e-5,
    logit_normal_mean: float = 0.0,
    logit_normal_std: float = 1.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    sampler = "logit_normal" if sampler in {"logitnormal", "logit-normal"} else sampler
    if sampler == "uniform":
        t = torch.rand(batch_size, device=device, dtype=dtype)
    elif sampler == "logit_normal":
        u = torch.randn(batch_size, device=device, dtype=dtype) * logit_normal_std + logit_normal_mean
        t = torch.sigmoid(u)
    else:
        raise ValueError(f"Unknown time sampler: {sampler!r}. Use 'uniform' or 'logit_normal'.")
    return t.clamp(eps, 1.0 - eps)


def build_flow_matching_batch(actions: torch.Tensor, *, cfg: FlowMatchingConfig):
    """Build x_t and target velocity for a batch of normalized action chunks."""
    x1 = actions
    x0 = torch.randn_like(x1)
    t = sample_time(
        actions.shape[0],
        actions.device,
        sampler=cfg.sampler_name,
        eps=cfg.t_eps,
        logit_normal_mean=cfg.logit_normal_mean,
        logit_normal_std=cfg.logit_normal_std,
        dtype=actions.dtype,
    )
    t_view = _expand_time(t, x1)

    if cfg.sigma_min == 0.0:
        xt = (1.0 - t_view) * x0 + t_view * x1
        target_v = x1 - x0
    else:
        # Optional path: x_t = (1 - (1-sigma_min)t)x0 + t x1.
        xt = (1.0 - (1.0 - cfg.sigma_min) * t_view) * x0 + t_view * x1
        target_v = x1 - (1.0 - cfg.sigma_min) * x0

    model_t = t * cfg.time_scale
    return xt, t, model_t, target_v, x0


def flow_matching_loss(
    model: torch.nn.Module,
    *,
    actions: torch.Tensor,
    global_cond: torch.Tensor,
    cfg: Optional[FlowMatchingConfig] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute conditional flow matching velocity MSE."""
    cfg = cfg or FlowMatchingConfig()
    xt, t, model_t, target_v, x0 = build_flow_matching_batch(actions, cfg=cfg)
    pred_v = model(sample=xt, timestep=model_t, global_cond=global_cond)
    loss = F.mse_loss(pred_v, target_v)
    logs = {
        "fm_loss": loss.detach(),
        "t_mean": t.detach().mean(),
        "pred_v_norm": pred_v.detach().pow(2).mean().sqrt(),
        "target_v_norm": target_v.detach().pow(2).mean().sqrt(),
        "x0_norm": x0.detach().pow(2).mean().sqrt(),
    }
    return loss, logs


@torch.no_grad()
def sample_flow(
    model: torch.nn.Module,
    *,
    sample_shape: Sequence[int],
    global_cond: torch.Tensor,
    num_steps: int = 20,
    solver: str = "euler",
    time_scale: float = 100.0,
    clip_mode: str = "final",
    noise_scale: float = 1.0,
    generator: Optional[torch.Generator] = None,
    x_init: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sample a normalized action chunk by integrating the learned flow.

    clip_mode:
        none  - no clipping
        final - clamp final sample to [-1, 1]
        step  - clamp after every integration step and at the end
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if solver not in {"euler", "heun"}:
        raise ValueError("solver must be 'euler' or 'heun'")
    if clip_mode not in {"none", "final", "step"}:
        raise ValueError("clip_mode must be one of: none, final, step")

    device = global_cond.device
    dtype = global_cond.dtype if global_cond.is_floating_point() else torch.float32
    sample_shape = tuple(int(v) for v in sample_shape)
    batch_size = sample_shape[0]

    if x_init is None:
        x = torch.randn(sample_shape, device=device, dtype=dtype, generator=generator) * noise_scale
    else:
        x = x_init.to(device=device, dtype=dtype)
        if tuple(x.shape) != sample_shape:
            raise ValueError(f"x_init has shape {tuple(x.shape)}, expected {sample_shape}")

    dt = 1.0 / float(num_steps)
    each_clip = clip_mode == "step"

    for i in range(num_steps):
        if solver == "euler":
            # Midpoint time often behaves slightly better than left-endpoint Euler for rectified flows.
            t_eval = (i + 0.5) * dt
            t = torch.full((batch_size,), t_eval * time_scale, device=device, dtype=dtype)
            v = model(sample=x, timestep=t, global_cond=global_cond)
            x = x + dt * v
        else:
            t0 = torch.full((batch_size,), i * dt * time_scale, device=device, dtype=dtype)
            v0 = model(sample=x, timestep=t0, global_cond=global_cond)
            x_euler = x + dt * v0
            if each_clip:
                x_euler = x_euler.clamp(-1.0, 1.0)
            t1 = torch.full((batch_size,), min((i + 1) * dt, 1.0) * time_scale, device=device, dtype=dtype)
            v1 = model(sample=x_euler, timestep=t1, global_cond=global_cond)
            x = x + 0.5 * dt * (v0 + v1)

        if each_clip:
            x = x.clamp(-1.0, 1.0)

    if clip_mode in {"final", "step"}:
        x = x.clamp(-1.0, 1.0)
    return x
