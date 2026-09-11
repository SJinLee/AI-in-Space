#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LEO SEU physics: true IEEE-754 float32 single-bit XOR (NOT Gaussian noise).

Rate model
----------
    r      : SEU / bit / day   (default 1e-6)
    N_bits : 32 * (number of flippable float32 weight elements)
    λ      : r * N_bits * Δt_days

Modes
-----
poisson  Homogeneous Poisson process on [0, T]. Sample K ~ Poisson(λ),
         drop event times uniform in [0, T]. Each event = (weight_index, bit 0-31).

leo_saa  One "mission day" (or a scaled window of length T). Geometry of a
         1-day LEO mission: 3 South-Atlantic-Anomaly (SAA) passes × 12 min.
         Put ~90% of expected daily SEUs into those passes, remainder in quiet
         time:

             r_saa   ≈ r_avg * 0.90 / (3 * 12 / 1440)  ≈ 36 * r_avg
             r_quiet ≈ r_avg * 0.10 / (1 - 3*12/1440)  ≈ 0.1026 * r_avg

         Pass start times are equally spaced slots with uniform jitter
         (non-overlapping). The number of passes has expectation exactly
         3 * T (integer part + Bernoulli on the fractional part), so
         E[λ] = r * N_bits * T holds for every T, not only integer days.
         A window no longer than one pass (T <= 12/1440, the `saa_pass`
         scenario) sits entirely inside a pass and gets the SAA rate 36 r.

allowed_bits  Optional list of bit positions (0-31) the flips may land on
         (e.g. [30] = exponent MSB only). Default: all 32.
single_hit    Exactly one event in the window (bit-criticality experiments).

stress   Ignore physical r. Hit each weight independently with probability p
         (default 0.01), then XOR one random bit. Course-style accelerator.
         No time T.

Flippable tensors: parameters with dim >= 2 (matrices / conv kernels).
Biases (dim == 1) are skipped unless flip_bias=True.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn

BITS_PER_FLOAT32 = 32
DEFAULT_R = 1e-6  # SEU / bit / day
DEFAULT_STRESS_P = 0.01

SAA_PASS_MINUTES = 12.0
SAA_PASSES_PER_DAY = 3
MINUTES_PER_DAY = 1440.0
SAA_FRACTION_OF_DAILY = 0.90  # ~90% of expected daily SEUs in SAA passes

SAA_PASS_DAYS = SAA_PASS_MINUTES / MINUTES_PER_DAY  # 12/1440 = 1/120 day
SAA_TIME_FRACTION = SAA_PASSES_PER_DAY * SAA_PASS_DAYS  # 0.025
# r_saa ≈ r * 0.90 / 0.025 = 36 r
R_SAA_MULTIPLIER = SAA_FRACTION_OF_DAILY / SAA_TIME_FRACTION
# remainder of fluence spread over quiet fraction 0.975
R_QUIET_MULTIPLIER = (1.0 - SAA_FRACTION_OF_DAILY) / (1.0 - SAA_TIME_FRACTION)


@dataclass
class SeuEvent:
    """One SEU: XOR `bit` of flippable weight element `weight_index` at time `t_days`."""

    t_days: float
    weight_index: int
    bit: int  # 0..31


@dataclass
class RadiationSchedule:
    """Sampled SEU events plus the λ / rate bookkeeping used to generate them."""

    mode: str
    events: list[SeuEvent] = field(default_factory=list)
    lam: float = 0.0
    T_days: float = 1.0
    r: float = DEFAULT_R
    p: float = DEFAULT_STRESS_P
    n_bits: int = 0
    n_weights: int = 0
    n_saa_passes: int = 0
    saa_intervals: list[tuple[float, float]] = field(default_factory=list)

    @property
    def k(self) -> int:
        return len(self.events)


def collect_weight_params(model: nn.Module, flip_bias: bool = False) -> list[torch.nn.Parameter]:
    """Weight tensors (dim >= 2). Biases only if flip_bias. Includes frozen params
    (they still sit in SRAM and can take SEUs)."""
    out: list[torch.nn.Parameter] = []
    for _name, p in model.named_parameters():
        if p.dim() >= 2:
            out.append(p)
        elif flip_bias and p.dim() == 1:
            out.append(p)
    return out


def n_weight_elements(params: Sequence[torch.nn.Parameter]) -> int:
    return int(sum(int(p.numel()) for p in params))


def n_bits_of(params: Sequence[torch.nn.Parameter]) -> int:
    return n_weight_elements(params) * BITS_PER_FLOAT32


def weight_bank_meta(params: Sequence[torch.nn.Parameter]) -> tuple[np.ndarray, int]:
    """Prefix sums of numel, so a global flat index can be mapped to (tensor, local)."""
    sizes = [int(p.numel()) for p in params]
    cum = np.cumsum([0] + sizes).astype(np.int64)
    return cum, int(cum[-1]) if len(cum) else 0


def _u32_view(w: torch.Tensor) -> np.ndarray:
    """uint32 view that SHARES MEMORY with a contiguous CPU float32 tensor.

    Guard: on a CUDA tensor `.cpu().numpy()` is a copy, so a flip would be
    lost silently and the model would look radiation-proof. Keep models on CPU.
    """
    if w.device.type != "cpu":
        raise RuntimeError(
            "SEU injection needs a CPU tensor (got %s): .cpu().numpy() on a CUDA tensor is a copy, "
            "so the bit-flip would be lost silently." % w.device
        )
    if w.dtype != torch.float32:
        raise TypeError(f"SEU injection expects float32 weights, got {w.dtype}")
    return w.detach().numpy().view(np.uint32).ravel()


def parse_allowed_bits(spec) -> list[int] | None:
    """'30' | '23-30' | '31,30' | 'sign' | 'exponent' | 'mantissa' | 'all' | None -> list of bit positions."""
    if spec is None:
        return None
    if isinstance(spec, (list, tuple)):
        bits = [int(b) for b in spec]
    else:
        s = str(spec).strip().lower()
        if s in ("", "all", "none"):
            return None
        named = {"sign": [31], "exponent": list(range(23, 31)), "mantissa": list(range(0, 23))}
        if s in named:
            return named[s]
        bits = []
        for part in s.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-")
                bits.extend(range(int(lo), int(hi) + 1))
            elif part:
                bits.append(int(part))
    bits = sorted(set(bits))
    if not bits or min(bits) < 0 or max(bits) > 31:
        raise ValueError(f"allowed_bits must be within 0..31, got {spec!r}")
    return bits


def xor_one_bit(params: Sequence[torch.nn.Parameter], cum: np.ndarray, flat_idx: int, bit: int) -> None:
    """In-place IEEE-754 float32 single-bit XOR on a weight element. True SEU."""
    layer = int(np.searchsorted(cum, flat_idx, side="right") - 1)
    local = int(flat_idx - cum[layer])
    w = params[layer].data
    if not w.is_contiguous():
        w = w.contiguous()
        params[layer].data = w
    u32 = _u32_view(w)
    u32[local] ^= np.uint32(1) << np.uint32(int(bit) & 31)


def xor_one_bit_tensor(tensor: torch.Tensor, local: int, bit: int) -> None:
    w = tensor.data
    if not w.is_contiguous():
        w = w.contiguous()
        tensor.data = w
    u32 = _u32_view(w)
    u32[int(local)] ^= np.uint32(1) << np.uint32(int(bit) & 31)


def apply_events(
    params: Sequence[torch.nn.Parameter],
    cum: np.ndarray,
    events: Iterable[SeuEvent],
) -> int:
    n = 0
    for ev in events:
        xor_one_bit(params, cum, ev.weight_index, ev.bit)
        n += 1
    return n


def apply_stress_inplace(
    params: Sequence[torch.nn.Parameter],
    p: float,
    rng: np.random.Generator,
) -> int:
    """Vectorized per-weight hit: each element independently flipped with prob p."""
    n_hits = 0
    p = float(p)
    for param in params:
        w = param.data
        if not w.is_contiguous():
            w = w.contiguous()
            param.data = w
        n = int(w.numel())
        if n == 0:
            continue
        mask = rng.random(n) < p
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        bits = rng.integers(0, 32, size=idx.size, endpoint=False, dtype=np.int64)
        u32 = _u32_view(w)
        u32[idx] ^= np.uint32(1) << bits.astype(np.uint32)
        n_hits += int(idx.size)
    return n_hits


def bit_kind(bit: int) -> str:
    b = int(bit) & 31
    if b == 31:
        return "sign"
    if 23 <= b <= 30:
        return "exponent"
    return "mantissa"


# ---------------------------------------------------------------------------
# Event sampling
# ---------------------------------------------------------------------------
def _sample_bits(k: int, rng: np.random.Generator, allowed_bits=None) -> np.ndarray:
    """k bit positions, uniform over 0-31 or over `allowed_bits`."""
    if allowed_bits is None:
        return rng.integers(0, 32, size=k, endpoint=False)
    return rng.choice(np.asarray(list(allowed_bits), dtype=np.int64), size=k)


def _emit_events(
    times: np.ndarray,
    n_weights: int,
    rng: np.random.Generator,
    allowed_bits=None,
) -> list[SeuEvent]:
    k = int(times.size)
    if k == 0 or n_weights <= 0:
        return []
    widx = rng.integers(0, n_weights, size=k, endpoint=False)
    bits = _sample_bits(k, rng, allowed_bits)
    return [
        SeuEvent(t_days=float(t), weight_index=int(w), bit=int(b))
        for t, w, b in zip(times.tolist(), widx.tolist(), bits.tolist())
    ]


def sample_poisson_events(
    r: float,
    n_bits: int,
    T_days: float,
    rng: np.random.Generator,
    allowed_bits=None,
) -> tuple[list[SeuEvent], float]:
    """Homogeneous Poisson process on [0, T]. λ = r * N_bits * T."""
    T_days = float(max(T_days, 0.0))
    n_weights = int(n_bits) // BITS_PER_FLOAT32
    lam = float(r) * float(n_bits) * T_days
    K = int(rng.poisson(lam)) if lam > 0 and n_weights > 0 else 0
    if K == 0:
        return [], lam
    times = rng.uniform(0.0, T_days, size=K)
    return _emit_events(times, n_weights, rng, allowed_bits), lam


def _place_saa_passes(
    T_days: float,
    n_passes: int,
    pass_dur: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Non-overlapping pass start times: one pass per equally spaced slot, jittered."""
    T_days = float(T_days)
    n_passes = int(n_passes)
    pass_dur = float(pass_dur)
    if n_passes <= 0:
        return np.zeros(0, dtype=np.float64)
    if n_passes * pass_dur >= T_days - 1e-15:
        dur = T_days / n_passes
        return np.arange(n_passes, dtype=np.float64) * dur
    slot = T_days / n_passes
    starts = np.empty(n_passes, dtype=np.float64)
    for i in range(n_passes):
        lo = i * slot
        hi = (i + 1) * slot - pass_dur
        starts[i] = float(lo if hi <= lo else rng.uniform(lo, hi))
    return starts


def _sample_times_in_intervals(
    intervals: list[tuple[float, float]],
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n <= 0 or not intervals:
        return np.zeros(0, dtype=np.float64)
    durs = np.array([hi - lo for lo, hi in intervals], dtype=np.float64)
    total = float(durs.sum())
    if total <= 0:
        return np.zeros(0, dtype=np.float64)
    starts = np.array([lo for lo, _hi in intervals], dtype=np.float64)
    choices = rng.choice(len(intervals), size=n, p=durs / total)
    offs = rng.random(n) * durs[choices]
    return starts[choices] + offs


def _complement_intervals(T_days: float, occupied: list[tuple[float, float]]) -> list[tuple[float, float]]:
    quiet: list[tuple[float, float]] = []
    t = 0.0
    for lo, hi in sorted(occupied):
        if lo > t:
            quiet.append((t, lo))
        t = max(t, hi)
    if t < T_days:
        quiet.append((t, T_days))
    return quiet


def sample_leo_saa_events(
    r: float,
    n_bits: int,
    T_days: float,
    rng: np.random.Generator,
    allowed_bits=None,
) -> tuple[list[SeuEvent], float, int, list[tuple[float, float]]]:
    """LEO day with 3 × 12 min SAA passes (scaled to window T_days).

    E[number of passes] = 3 * T exactly, so E[λ] = r * N_bits * T for any T.
    """
    T_days = float(max(T_days, 0.0))
    n_weights = int(n_bits) // BITS_PER_FLOAT32
    lam_avg = float(r) * float(n_bits) * T_days
    if T_days <= 0.0 or n_weights <= 0:
        return [], lam_avg, 0, []

    pass_dur = SAA_PASS_DAYS
    if T_days <= pass_dur + 1e-12:
        # Window no longer than one pass: it sits entirely inside an SAA pass
        # (the `--T saa_pass` scenario). Dose = 36 r * N_bits * T.
        n_passes = 1
        pass_dur = T_days
    else:
        # Integer part + Bernoulli(fractional part) → E[n_passes] = 3 T exactly.
        n_float = SAA_PASSES_PER_DAY * T_days
        n_passes = int(np.floor(n_float)) + int(rng.random() < (n_float - np.floor(n_float)))
        if n_passes * pass_dur > T_days:  # cannot happen for T > pass_dur, kept as a guard
            n_passes = int(np.floor(T_days / pass_dur + 1e-12))

    starts = _place_saa_passes(T_days, n_passes, pass_dur, rng)
    saa_intervals = [(float(s), float(s + pass_dur)) for s in starts.tolist()]
    quiet_intervals = _complement_intervals(T_days, saa_intervals)

    total_saa = float(sum(hi - lo for lo, hi in saa_intervals))
    total_quiet = float(max(0.0, T_days - total_saa))

    r_saa = float(r) * R_SAA_MULTIPLIER
    r_quiet = float(r) * R_QUIET_MULTIPLIER
    lam_saa = r_saa * float(n_bits) * total_saa
    lam_quiet = r_quiet * float(n_bits) * total_quiet
    lam = lam_saa + lam_quiet

    events: list[SeuEvent] = []
    K_saa = int(rng.poisson(lam_saa)) if lam_saa > 0 else 0
    if K_saa:
        times = _sample_times_in_intervals(saa_intervals, K_saa, rng)
        events.extend(_emit_events(times, n_weights, rng, allowed_bits))
    K_q = int(rng.poisson(lam_quiet)) if lam_quiet > 0 else 0
    if K_q:
        times = _sample_times_in_intervals(quiet_intervals, K_q, rng)
        events.extend(_emit_events(times, n_weights, rng, allowed_bits))
    events.sort(key=lambda e: e.t_days)
    return events, lam, n_passes, saa_intervals


def sample_stress_events(
    n_weights: int,
    p: float,
    rng: np.random.Generator,
    T_days: float = 1.0,
    allowed_bits=None,
) -> list[SeuEvent]:
    """Independent per-weight hit; event times uniform in [0, T] so training can map them to steps."""
    if n_weights <= 0 or p <= 0:
        return []
    mask = rng.random(int(n_weights)) < float(p)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    bits = _sample_bits(int(idx.size), rng, allowed_bits)
    times = rng.uniform(0.0, max(float(T_days), 1e-12), size=idx.size)
    return [
        SeuEvent(t_days=float(t), weight_index=int(w), bit=int(b))
        for t, w, b in zip(times.tolist(), idx.tolist(), bits.tolist())
    ]


def sample_schedule(
    mode: str,
    *,
    n_bits: int,
    rng: np.random.Generator,
    r: float = DEFAULT_R,
    T_days: float = 1.0,
    p: float = DEFAULT_STRESS_P,
    allowed_bits=None,
    single_hit: bool = False,
) -> RadiationSchedule:
    mode = str(mode).lower().strip()
    n_weights = int(n_bits) // BITS_PER_FLOAT32
    allowed_bits = parse_allowed_bits(allowed_bits)
    sched = RadiationSchedule(
        mode=mode, T_days=float(T_days), r=float(r), p=float(p),
        n_bits=int(n_bits), n_weights=n_weights,
    )
    if single_hit:
        # Exactly one (weight, bit) event, uniform in time — bit-criticality experiments.
        t = rng.uniform(0.0, max(float(T_days), 1e-12), size=1)
        sched.events = _emit_events(t, n_weights, rng, allowed_bits)
        sched.lam = 1.0
        return sched
    if mode == "stress":
        sched.events = sample_stress_events(n_weights, p, rng, T_days=T_days, allowed_bits=allowed_bits)
        sched.lam = float(p) * n_weights  # expected hits, not a Poisson λ
        return sched
    if mode == "poisson":
        events, lam = sample_poisson_events(r, n_bits, T_days, rng, allowed_bits)
        sched.events = events
        sched.lam = lam
        return sched
    if mode in ("leo_saa", "leo", "saa"):
        events, lam, n_passes, intervals = sample_leo_saa_events(r, n_bits, T_days, rng, allowed_bits)
        sched.events = events
        sched.lam = lam
        sched.n_saa_passes = n_passes
        sched.saa_intervals = intervals
        return sched
    raise ValueError(f"unknown radiation mode: {mode!r} (use poisson|leo_saa|stress)")


def events_by_step(events: Sequence[SeuEvent], n_steps: int, T_days: float) -> dict[int, list[SeuEvent]]:
    """Map event times in [0, T] onto optimizer steps: t=0 → step 0, t=T → last step.

    An event at 30% of the window is injected at step floor(0.3 * n_steps).
    """
    from collections import defaultdict

    n_steps = max(int(n_steps), 1)
    T_days = float(T_days) if T_days > 0 else 1.0
    bucket: dict[int, list[SeuEvent]] = defaultdict(list)
    for ev in events:
        frac = min(max(ev.t_days / T_days, 0.0), 1.0 - 1e-15)
        step = int(frac * n_steps)
        if step >= n_steps:
            step = n_steps - 1
        bucket[step].append(ev)
    return dict(bucket)


def self_test_bitflip() -> None:
    """Sanity: sign-bit XOR of 1.0 → -1.0; exponent hit is huge/non-finite."""
    t = torch.tensor([1.0], dtype=torch.float32)
    u = t.numpy().view(np.uint32)
    u[0] ^= np.uint32(1) << np.uint32(31)
    assert t.item() == -1.0, t.item()
    t2 = torch.tensor([1.0], dtype=torch.float32)
    u2 = t2.numpy().view(np.uint32)
    u2[0] ^= np.uint32(1) << np.uint32(30)
    assert (not np.isfinite(t2.item())) or abs(t2.item()) > 1e30
    # poisson λ bookkeeping
    rng = np.random.default_rng(0)
    n_bits = 50 * 32
    _ev, lam = sample_poisson_events(1e-6, n_bits, 1.0, rng)
    assert abs(lam - 1e-6 * n_bits) < 1e-18, lam
    assert abs(R_SAA_MULTIPLIER - 36.0) < 1e-9, R_SAA_MULTIPLIER
    # leo_saa: E[λ] must equal r*N_bits*T even for fractional days
    rng = np.random.default_rng(1)
    for T in (0.5, 1.5, 2.5):
        lams = [sample_leo_saa_events(1e-6, 3_252_224, T, rng)[1] for _ in range(400)]
        assert abs(np.mean(lams) / (1e-6 * 3_252_224 * T) - 1.0) < 0.05, (T, np.mean(lams))
    # allowed_bits / single_hit
    s = sample_schedule("stress", n_bits=3200, rng=rng, p=1.0, allowed_bits="30")
    assert s.events and all(e.bit == 30 for e in s.events)
    s = sample_schedule("leo_saa", n_bits=3200, rng=rng, single_hit=True, allowed_bits="exponent")
    assert s.k == 1 and 23 <= s.events[0].bit <= 30
