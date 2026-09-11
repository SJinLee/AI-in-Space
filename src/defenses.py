#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SEU defenses applied to float32 weight tensors.

none      Leave flipped bits as-is.
clip      nan_to_num + clamp every parameter to [-3, 3] after injection
          (and before eval).
tmr       Triple modular redundancy. Inference: 3 independently corrupted
          copies of clean W, per-weight median. Training: after each
          injection, 3 independent corruptions of the *pre-injection* W
          (same event count K, independent locations/bits), median → W.
ensemble  Inference: 5 independently corrupted models, average logits
          (after nan_to_num). NOTE: a replica whose weight became ~2^127
          (exponent-MSB flip) emits huge-but-FINITE logits that dominate the
          average — the mean is not robust. Kept as the "naive" baseline.
ensemble_median
          Same replicas, per-element MEDIAN of the logits instead of the mean.
          Robust to one wild replica (same idea as TMR, applied to outputs).
          Training: 3-model combination, only for cheap models
          (toy50, mnist_linear). CNN / ResNet: inference-only.
parity    Even parity bit stored per float32 weight (popcount(bits) mod 2).
          Inference: zero any weight whose parity no longer matches.
          Training: revert a parity-failing weight to its pre-flip value
          (needs a snapshot). After a legitimate optimizer step the stored
          parity is refreshed.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

CLIP_LO, CLIP_HI = -3.0, 3.0
TMR_REPLICAS = 3
ENSEMBLE_INFER_REPLICAS = 5
ENSEMBLE_TRAIN_REPLICAS = 3

# Training-time logit-average ensemble is cheap enough only on these models.
ENSEMBLE_TRAIN_MODELS = frozenset({"toy50", "mnist_linear"})
ENSEMBLE_DEFENSES = ("ensemble", "ensemble_median")
DEFENSE_NAMES = ("none", "clip", "tmr", "ensemble", "ensemble_median", "parity")


def ensemble_reduce_of(defense: str) -> str:
    return "median" if defense == "ensemble_median" else "mean"


def combine_logits(stack: torch.Tensor, how: str = "mean") -> torch.Tensor:
    """stack: (n_replicas, B, C) finite logits → (B, C).

    'mean'   arithmetic mean — one huge-but-finite replica dominates.
    'median' per-element median — robust to a minority of wild replicas.
    """
    if how == "median":
        return stack.median(dim=0).values
    if how == "mean":
        return stack.mean(dim=0)
    raise ValueError(f"unknown ensemble reduce {how!r}")


def clip_all_params(model: nn.Module, lo: float = CLIP_LO, hi: float = CLIP_HI) -> None:
    """Week-11 harden: nan_to_num + clip weights AND biases to [lo, hi]."""
    with torch.no_grad():
        for p in model.parameters():
            p.data.nan_to_num_(nan=0.0, posinf=hi, neginf=lo)
            p.data.clamp_(lo, hi)


def clip_param_list(params: Sequence[torch.nn.Parameter], lo: float = CLIP_LO, hi: float = CLIP_HI) -> None:
    with torch.no_grad():
        for p in params:
            p.data.nan_to_num_(nan=0.0, posinf=hi, neginf=lo)
            p.data.clamp_(lo, hi)


def snapshot_params(params: Sequence[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [p.data.detach().clone() for p in params]


def restore_params(params: Sequence[torch.nn.Parameter], snap: Sequence[torch.Tensor]) -> None:
    with torch.no_grad():
        for p, s in zip(params, snap):
            p.data.copy_(s)


def write_median(params: Sequence[torch.nn.Parameter], snaps: Sequence[Sequence[torch.Tensor]]) -> None:
    """Per-element median of several snapshots → live params.

    Non-finite replicas are nan_to_num'd before the median so a single Inf
    replica cannot poison the vote (median of finite values is defined).
    """
    with torch.no_grad():
        for i, p in enumerate(params):
            stacked = torch.stack([s[i].nan_to_num(nan=0.0, posinf=0.0, neginf=0.0) for s in snaps], dim=0)
            p.data.copy_(stacked.median(dim=0).values)


def params_finite(model: nn.Module) -> bool:
    for p in model.parameters():
        if not torch.isfinite(p.data).all():
            return False
    return True


def even_parity_u32(u32: np.ndarray) -> np.ndarray:
    """Even-parity bit of each uint32: popcount mod 2. 0 = even number of 1s."""
    x = np.asarray(u32, dtype=np.uint32).ravel()
    # SWAR popcount
    y = x.astype(np.uint32)
    y = y - ((y >> np.uint32(1)) & np.uint32(0x55555555))
    y = (y & np.uint32(0x33333333)) + ((y >> np.uint32(2)) & np.uint32(0x33333333))
    y = (y + (y >> np.uint32(4))) & np.uint32(0x0F0F0F0F)
    y = y + (y >> np.uint32(8))
    y = y + (y >> np.uint32(16))
    return (y & np.uint32(0x3F) & np.uint32(1)).astype(np.uint8)


def parity_of_params(params: Sequence[torch.nn.Parameter]) -> list[np.ndarray]:
    tables: list[np.ndarray] = []
    for p in params:
        u32 = p.data.detach().cpu().contiguous().numpy().view(np.uint32).ravel()
        tables.append(even_parity_u32(u32))
    return tables


def parity_fail_mask(params: Sequence[torch.nn.Parameter], stored: Sequence[np.ndarray]) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for p, s in zip(params, stored):
        u32 = p.data.detach().cpu().contiguous().numpy().view(np.uint32).ravel()
        now = even_parity_u32(u32)
        masks.append(now != np.asarray(s).ravel())
    return masks


def zero_failed_parity(params: Sequence[torch.nn.Parameter], stored: Sequence[np.ndarray]) -> int:
    """Inference repair: zero any weight whose even parity no longer matches."""
    n = 0
    with torch.no_grad():
        for p, s in zip(params, stored):
            u32 = p.data.detach().cpu().contiguous().numpy().view(np.uint32).ravel()
            fail = even_parity_u32(u32) != np.asarray(s).ravel()
            if not fail.any():
                continue
            flat = p.data.view(-1)
            idx = torch.from_numpy(np.flatnonzero(fail).astype(np.int64))
            flat[idx] = 0.0
            n += int(fail.sum())
    return n


def revert_failed_parity(
    params: Sequence[torch.nn.Parameter],
    stored: Sequence[np.ndarray],
    pre_snap: Sequence[torch.Tensor],
) -> int:
    """Training repair: revert parity-failing weights to the pre-flip snapshot."""
    n = 0
    with torch.no_grad():
        for p, s, pre in zip(params, stored, pre_snap):
            u32 = p.data.detach().cpu().contiguous().numpy().view(np.uint32).ravel()
            fail = even_parity_u32(u32) != np.asarray(s).ravel()
            if not fail.any():
                continue
            flat = p.data.view(-1)
            pre_flat = pre.reshape(-1)
            idx = torch.from_numpy(np.flatnonzero(fail).astype(np.int64))
            flat[idx] = pre_flat[idx]
            n += int(fail.sum())
    return n


def nan_to_num_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
