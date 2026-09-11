#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train / infer loops that inject true SEU bit-flips.

Inference
    Load a CLEAN checkpoint (trained once per model, cached). Each trial
    copies weights, applies the radiation schedule for exposure T, applies
    the defense, then evaluates.

Training
    From scratch (or from the pretrained backbone for resnet18_t10 transfer).
    Event times in [0, T] map onto optimizer steps:
        step = floor((t / T) * n_steps)
    so an event at 30% of the window is injected at the 30% step.
    Defense is applied after each injection. Evaluate at the end.

    ASSUMPTION (not Colab wall-clock): --train-hours 24 maps the whole
    training run to T = 1.0 LEO day so λ = r * N_bits * 1 is nontrivial.
    Stress mode ignores T.

Crash
    NaN/Inf loss, logits, grads, or weights → crashed=1. Record last finite
    accuracy, else chance-level 1/n_classes.
"""
from __future__ import annotations

import copy
import os
import time
import traceback
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from defenses import (
    CLIP_LO,
    CLIP_HI,
    ENSEMBLE_DEFENSES,
    ENSEMBLE_INFER_REPLICAS,
    ENSEMBLE_TRAIN_MODELS,
    ENSEMBLE_TRAIN_REPLICAS,
    TMR_REPLICAS,
    clip_all_params,
    combine_logits,
    ensemble_reduce_of,
    nan_to_num_logits,
    params_finite,
    parity_of_params,
    restore_params,
    revert_failed_parity,
    snapshot_params,
    write_median,
    zero_failed_parity,
)
from models import (
    chance_acc_of,
    n_classes_of,
)
from radiation import (
    SeuEvent,
    apply_events,
    bit_kind,
    collect_weight_params,
    events_by_step,
    n_bits_of,
    sample_schedule,
    weight_bank_meta,
    xor_one_bit,
)

CKPT_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "ckpts")


def _eval_batch_size(model_name: str) -> int:
    if model_name in ("resnet18_t10", "cifar_deep"):
        return 256
    if model_name == "mnist_cnn":
        return 512
    return 2048


def evaluate(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    model_name: str,
    clean_model: nn.Module | None = None,
) -> float | None:
    """Accuracy in percent, or None if non-finite.

    toy50 infer: sign-agreement vs clean predictions (course metric).
    toy50 train: sign match vs binary labels.
    others: top-1 vs labels.
    """
    model.eval()
    if not params_finite(model):
        return None
    bs = _eval_batch_size(model_name)
    n = int(x.shape[0])
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, n, bs):
            xb = x[start : start + bs]
            yb = y[start : start + bs]
            try:
                logits = model(xb)
            except Exception:
                traceback.print_exc()  # a real bug must not hide behind "crash"
                return None
            if not torch.isfinite(logits).all():
                return None
            if model_name == "toy50" and clean_model is not None:
                clean_pred = clean_model(xb)
                if not torch.isfinite(clean_pred).all():
                    return None
                # sign agreement of scalar scores
                a = logits.reshape(-1)
                b = clean_pred.reshape(-1)
                finite = torch.isfinite(a) & torch.isfinite(b)
                match = (torch.sign(a) == torch.sign(b)) & finite
                correct += int(match.sum().item())
                total += int(a.numel())
            elif model_name == "toy50":
                pred = (logits.reshape(-1) >= 0).long()
                correct += int((pred == yb.long()).sum().item())
                total += int(yb.numel())
            else:
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += int(yb.numel())
    if total == 0:
        return None
    acc = 100.0 * correct / total
    if not np.isfinite(acc):
        return None
    return float(acc)


def evaluate_ensemble_logits(
    replicas: list[nn.Module],
    x: torch.Tensor,
    y: torch.Tensor,
    model_name: str,
    clean_model: nn.Module | None = None,
    how: str = "mean",
) -> float | None:
    """Combine logits across independently corrupted replicas ('mean' or 'median')."""
    for m in replicas:
        m.eval()
        if not params_finite(m):
            # still try: nan_to_num on logits later
            pass
    bs = _eval_batch_size(model_name)
    n = int(x.shape[0])
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, n, bs):
            xb = x[start : start + bs]
            yb = y[start : start + bs]
            outs = []
            for m in replicas:
                try:
                    logits = m(xb)
                except Exception:
                    traceback.print_exc()
                    continue
                outs.append(nan_to_num_logits(logits))
            if not outs:
                return None
            logits = combine_logits(torch.stack(outs, dim=0), how)
            if model_name == "toy50" and clean_model is not None:
                clean_pred = clean_model(xb).reshape(-1)
                a = logits.reshape(-1)
                finite = torch.isfinite(a) & torch.isfinite(clean_pred)
                match = (torch.sign(a) == torch.sign(clean_pred)) & finite
                correct += int(match.sum().item())
                total += int(a.numel())
            elif model_name == "toy50":
                pred = (logits.reshape(-1) >= 0).long()
                correct += int((pred == yb.long()).sum().item())
                total += int(yb.numel())
            else:
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += int(yb.numel())
    if total == 0:
        return None
    return float(100.0 * correct / total)


def _count_bit_kinds(events: list[SeuEvent]) -> tuple[int, int]:
    n_exp = n_sign = 0
    for ev in events:
        k = bit_kind(ev.bit)
        if k == "exponent":
            n_exp += 1
        elif k == "sign":
            n_sign += 1
    return n_exp, n_sign


def _fresh_schedule(mode, n_bits, rng, r, T_days, p, allowed_bits=None, single_hit=False):
    return sample_schedule(mode, n_bits=n_bits, rng=rng, r=r, T_days=T_days, p=p,
                           allowed_bits=allowed_bits, single_hit=single_hit)


def _independent_k_flips(params, cum, n_weights, k, rng, allowed_bits=None) -> list[SeuEvent]:
    """K independent (weight, bit) flips, used by training TMR replicas."""
    from radiation import _sample_bits
    k = int(k)
    if k <= 0 or n_weights <= 0:
        return []
    widx = rng.integers(0, n_weights, size=k, endpoint=False)
    bits = _sample_bits(k, rng, allowed_bits)
    evs = [SeuEvent(t_days=0.0, weight_index=int(w), bit=int(b)) for w, b in zip(widx.tolist(), bits.tolist())]
    apply_events(params, cum, evs)
    return evs


def apply_defense_after_injection(
    *,
    defense: str,
    model: nn.Module,
    params,
    cum,
    n_weights: int,
    pending: list[SeuEvent],
    rng: np.random.Generator,
    parity_table,
    pre_snap,
    mode: str,
    p: float,
    allowed_bits=None,
) -> int:
    """Apply `defense` to live weights AFTER `pending` events have been XOR'd.

    Returns extra flip count introduced by TMR's independent replicas
    (already applied; the live W is the median, so they are not "on" W).
    """
    extra = 0
    if defense == "none":
        return extra
    if defense == "clip":
        clip_all_params(model)
        return extra
    if defense == "parity":
        if parity_table is not None and pre_snap is not None:
            revert_failed_parity(params, parity_table, pre_snap)
        return extra
    if defense == "tmr":
        # 3 independent corruptions of the *pre-injection* snapshot, median → W.
        # Same event count K (including stress hits assigned to this step),
        # independent locations/bits — not a second full p-pass.
        k = len(pending)
        snaps = []
        for _ in range(TMR_REPLICAS):
            restore_params(params, pre_snap)
            extra += len(_independent_k_flips(params, cum, n_weights, k, rng, allowed_bits))
            snaps.append(snapshot_params(params))
        write_median(params, snaps)
        return extra
    if defense in ENSEMBLE_DEFENSES:
        # Training ensemble is handled at the replica-loop level, not here.
        return extra
    raise ValueError(f"unknown defense {defense!r}")


# ---------------------------------------------------------------------------
# Inference trial
# ---------------------------------------------------------------------------
def run_infer_trial(
    *,
    model_factory,
    clean_state: dict,
    data,
    model_name: str,
    rad: str,
    defense: str,
    seed: int,
    r: float,
    T_days: float,
    p: float,
    flip_bias: bool,
    allowed_bits=None,
    single_hit: bool = False,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    rng = np.random.default_rng(int(seed))
    torch.manual_seed(int(seed))
    note = ""

    clean_model = model_factory()
    clean_model.load_state_dict(clean_state)
    clean_model.eval()

    def _one_irradiated(rng_i: np.random.Generator) -> tuple[nn.Module, object]:
        m = model_factory()
        m.load_state_dict(copy.deepcopy(clean_state))
        params = collect_weight_params(m, flip_bias=flip_bias)
        cum, n_w = weight_bank_meta(params)
        n_bits = n_w * 32
        sched = _fresh_schedule(rad, n_bits, rng_i, r, T_days, p, allowed_bits, single_hit)
        parity_table = parity_of_params(params) if defense == "parity" else None
        apply_events(params, cum, sched.events)
        if defense == "clip":
            clip_all_params(m)
        elif defense == "parity" and parity_table is not None:
            zero_failed_parity(params, parity_table)
        elif defense == "none":
            pass
        m.eval()
        return m, sched

    n_bits = n_bits_of(collect_weight_params(clean_model, flip_bias=flip_bias))
    crashed = 0
    k_mean = 0.0
    n_exp = n_sign = 0
    acc: float | None = None

    try:
        if defense == "tmr":
            snaps = []
            ks = []
            for i in range(TMR_REPLICAS):
                m, sched = _one_irradiated(np.random.default_rng(int(seed) * 1009 + i + 1))
                snaps.append(snapshot_params(collect_weight_params(m, flip_bias=flip_bias)))
                ks.append(sched.k)
                e, s = _count_bit_kinds(sched.events)
                n_exp += e
                n_sign += s
            voted = model_factory()
            voted.load_state_dict(copy.deepcopy(clean_state))
            write_median(collect_weight_params(voted, flip_bias=flip_bias), snaps)
            if not params_finite(voted):
                clip_all_params(voted)  # last-ditch so eval can run; still a crash if originally nonfinite
            k_mean = float(np.mean(ks)) if ks else 0.0
            n_exp = int(round(n_exp / max(len(ks), 1)))
            n_sign = int(round(n_sign / max(len(ks), 1)))
            acc = evaluate(voted, data.x_test, data.y_test, model_name, clean_model=clean_model)
        elif defense in ENSEMBLE_DEFENSES:
            replicas = []
            ks = []
            for i in range(ENSEMBLE_INFER_REPLICAS):
                m, sched = _one_irradiated(np.random.default_rng(int(seed) * 1009 + i + 1))
                replicas.append(m)
                ks.append(sched.k)
                e, s = _count_bit_kinds(sched.events)
                n_exp += e
                n_sign += s
            k_mean = float(np.mean(ks)) if ks else 0.0
            n_exp = int(round(n_exp / max(len(ks), 1)))
            n_sign = int(round(n_sign / max(len(ks), 1)))
            acc = evaluate_ensemble_logits(
                replicas, data.x_test, data.y_test, model_name, clean_model=clean_model,
                how=ensemble_reduce_of(defense),
            )
        else:
            m, sched = _one_irradiated(rng)
            k_mean = float(sched.k)
            n_exp, n_sign = _count_bit_kinds(sched.events)
            acc = evaluate(m, data.x_test, data.y_test, model_name, clean_model=clean_model)
    except Exception as exc:
        traceback.print_exc()
        note = f"exception: {exc!r}"
        crashed = 1
        acc = None

    if acc is None:
        crashed = 1
        acc = chance_acc_of(model_name)

    return {
        "acc": float(acc),
        "crashed": int(crashed),
        "k": float(k_mean),
        "n_bits": int(n_bits),
        "n_exponent_hits": int(n_exp),
        "n_sign_hits": int(n_sign),
        "seconds": float(time.perf_counter() - t0),
        "last_finite_acc": float(acc),
        "note": note,
    }


def clean_reference_acc(model_factory, clean_state: dict, data, model_name: str) -> float:
    """Clean accuracy measured with the SAME metric the irradiated trials use.

    toy50 infer scores sign-agreement with the clean model, so its reference is 100.
    Other models: top-1 of the clean checkpoint on this test set.
    """
    m = model_factory()
    m.load_state_dict(copy.deepcopy(clean_state))
    m.eval()
    ref = evaluate(m, data.x_test, data.y_test, model_name, clean_model=m if model_name == "toy50" else None)
    return float(ref) if ref is not None else float("nan")


# ---------------------------------------------------------------------------
# Training trial
# ---------------------------------------------------------------------------
def _n_batches(n_train: int, batch_size: int) -> int:
    n = n_train // batch_size
    rem = n_train % batch_size
    if rem >= 2:
        n += 1
    return max(n, 1)


def _make_opt(model: nn.Module, lr: float, momentum: float):
    trainable = [p for p in model.parameters() if p.requires_grad]
    if momentum and momentum > 0:
        return torch.optim.SGD(trainable, lr=lr, momentum=momentum)
    return torch.optim.SGD(trainable, lr=lr)


def _train_one_model(
    *,
    model: nn.Module,
    data,
    model_name: str,
    rad: str,
    defense: str,
    seed: int,
    rng: np.random.Generator,
    r: float,
    T_days: float,
    p: float,
    flip_bias: bool,
    epochs: int,
    batch_size: int,
    lr: float,
    momentum: float,
    replica_tag: int = 0,
    allowed_bits=None,
    single_hit: bool = False,
) -> dict[str, Any]:
    """Single-network training with in-graph injection. Used for none/clip/tmr/parity
    and as one replica of a training ensemble."""
    note = ""
    opt = _make_opt(model, lr, momentum)
    # toy50: BCE-with-logits on a scalar score; others: CE
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()

    params = collect_weight_params(model, flip_bias=flip_bias)
    cum, n_weights = weight_bank_meta(params)
    n_bits = n_weights * 32
    sched = _fresh_schedule(rad, n_bits, rng, r, T_days, p, allowed_bits, single_hit)

    n_train = int(data.x_train.shape[0])
    steps_per_epoch = _n_batches(n_train, batch_size)
    total_steps = max(epochs * steps_per_epoch, 1)
    by_step = events_by_step(sched.events, total_steps, sched.T_days if sched.T_days > 0 else 1.0)

    parity_table = parity_of_params(params) if defense == "parity" else None
    n_exp, n_sign = _count_bit_kinds(sched.events)

    crashed = 0
    n_flips_applied = 0
    last_finite_acc: float | None = None
    global_step = 0
    x_train, y_train = data.x_train, data.y_train

    try:
        for _epoch in range(int(epochs)):
            model.train()
            perm = torch.randperm(n_train)
            for start in range(0, n_train, batch_size):
                idx = perm[start : start + batch_size]
                if idx.numel() < 2:
                    continue
                pending = by_step.get(global_step, [])
                pre_snap = snapshot_params(params) if pending else None
                if pending and defense != "tmr":
                    apply_events(params, cum, pending)
                    n_flips_applied += len(pending)
                if pending:
                    extra = apply_defense_after_injection(
                        defense=defense if defense not in ENSEMBLE_DEFENSES else "none",
                        model=model,
                        params=params,
                        cum=cum,
                        n_weights=n_weights,
                        pending=pending,
                        rng=rng,
                        parity_table=parity_table,
                        pre_snap=pre_snap,
                        mode=rad,
                        p=p,
                        allowed_bits=allowed_bits,
                    )
                    if defense == "tmr" and pending:
                        n_flips_applied += len(pending)  # nominal K; live W is the median
                    n_flips_applied += extra

                if not params_finite(model):
                    crashed = 1
                    break

                opt.zero_grad(set_to_none=True)
                logits = model(x_train[idx])
                if not torch.isfinite(logits).all():
                    crashed = 1
                    break
                if model_name == "toy50":
                    loss = bce(logits.reshape(-1), y_train[idx].float())
                else:
                    loss = ce(logits, y_train[idx])
                if not torch.isfinite(loss):
                    crashed = 1
                    break
                loss.backward()
                grads_ok = True
                for p_ in model.parameters():
                    if p_.grad is not None and not torch.isfinite(p_.grad).all():
                        grads_ok = False
                        break
                if not grads_ok:
                    crashed = 1
                    break
                opt.step()
                if not params_finite(model):
                    crashed = 1
                    break
                if defense == "parity":
                    parity_table = parity_of_params(params)
                global_step += 1
            if crashed:
                break
            acc_e = evaluate(model, data.x_test, data.y_test, model_name, clean_model=None)
            if acc_e is None:
                crashed = 1
                break
            last_finite_acc = acc_e
    except Exception as exc:
        traceback.print_exc()
        note = f"exception: {exc!r}"
        crashed = 1

    return {
        "model": model,
        "crashed": int(crashed),
        "k": float(n_flips_applied if defense != "tmr" else sched.k),
        "n_bits": int(n_bits),
        "n_exponent_hits": int(n_exp),
        "n_sign_hits": int(n_sign),
        "last_finite_acc": last_finite_acc,
        "sched_k": int(sched.k),
        "note": note,
    }


def run_train_trial(
    *,
    model_factory,
    data,
    model_name: str,
    rad: str,
    defense: str,
    seed: int,
    r: float,
    T_days: float,
    p: float,
    flip_bias: bool,
    epochs: int,
    batch_size: int,
    lr: float,
    momentum: float,
    allowed_bits=None,
    single_hit: bool = False,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    chance = chance_acc_of(model_name)

    if defense in ENSEMBLE_DEFENSES and model_name not in ENSEMBLE_TRAIN_MODELS:
        # Documented: CNN/ResNet ensemble is inference-only. Fall back to none
        # during train so the cell still produces a number, and tag it.
        defense_eff = "none"
        ensemble_skipped = 1
    else:
        defense_eff = defense
        ensemble_skipped = 0

    if defense_eff in ENSEMBLE_DEFENSES:
        replicas = []
        infos = []
        for i in range(ENSEMBLE_TRAIN_REPLICAS):
            m = model_factory()
            info = _train_one_model(
                model=m,
                data=data,
                model_name=model_name,
                rad=rad,
                defense="none",  # each replica is independently irradiated
                seed=int(seed),
                rng=np.random.default_rng(int(seed) * 1009 + i + 7),
                r=r,
                T_days=T_days,
                p=p,
                flip_bias=flip_bias,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                momentum=momentum,
                replica_tag=i,
                allowed_bits=allowed_bits,
                single_hit=single_hit,
            )
            replicas.append(info["model"])
            infos.append(info)
        crashed = int(any(i["crashed"] for i in infos))  # "crash" = any replica died
        acc = evaluate_ensemble_logits(replicas, data.x_test, data.y_test, model_name, clean_model=None,
                                       how=ensemble_reduce_of(defense_eff))
        last = None
        for i in infos:
            if i["last_finite_acc"] is not None:
                last = i["last_finite_acc"]
        if acc is None:
            crashed = 1
            acc = last if last is not None else chance
        k_mean = float(np.mean([i["sched_k"] for i in infos]))
        n_bits = int(infos[0]["n_bits"])
        n_exp = int(round(np.mean([i["n_exponent_hits"] for i in infos])))
        n_sign = int(round(np.mean([i["n_sign_hits"] for i in infos])))
        return {
            "acc": float(acc),
            "crashed": int(crashed),
            "k": k_mean,
            "n_bits": n_bits,
            "n_exponent_hits": n_exp,
            "n_sign_hits": n_sign,
            "seconds": float(time.perf_counter() - t0),
            "last_finite_acc": float(acc),
            "ensemble_skipped": int(ensemble_skipped),
            "note": "; ".join(i["note"] for i in infos if i["note"]),
        }

    model = model_factory()
    info = _train_one_model(
        model=model,
        data=data,
        model_name=model_name,
        rad=rad,
        defense=defense_eff,
        seed=int(seed),
        rng=rng,
        r=r,
        T_days=T_days,
        p=p,
        flip_bias=flip_bias,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        momentum=momentum,
        allowed_bits=allowed_bits,
        single_hit=single_hit,
    )
    crashed = int(info["crashed"])
    if defense_eff == "clip":
        clip_all_params(model)
    acc = evaluate(model, data.x_test, data.y_test, model_name, clean_model=None)
    if acc is None:
        crashed = 1
        acc = info["last_finite_acc"] if info["last_finite_acc"] is not None else chance
    return {
        "acc": float(acc),
        "crashed": int(crashed),
        "k": float(info["k"]),
        "n_bits": int(info["n_bits"]),
        "n_exponent_hits": int(info["n_exponent_hits"]),
        "n_sign_hits": int(info["n_sign_hits"]),
        "seconds": float(time.perf_counter() - t0),
        "last_finite_acc": float(acc),
        "ensemble_skipped": int(ensemble_skipped),
        "note": info["note"],
    }


# ---------------------------------------------------------------------------
# Clean checkpoint (cached, used by infer)
# ---------------------------------------------------------------------------
def ckpt_path(ckpt_dir: str, model_name: str, n_train: int | None = None, epochs: int | None = None) -> str:
    """Checkpoint name carries the training budget so a changed --n-train/--epochs
    never silently reuses an old file."""
    if n_train is None or epochs is None:
        return os.path.join(ckpt_dir, f"{model_name}_clean.pt")
    return os.path.join(ckpt_dir, f"{model_name}_clean_n{int(n_train)}_e{int(epochs)}.pt")


def train_clean_checkpoint(
    *,
    model_factory,
    data,
    model_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    momentum: float,
    ckpt_dir: str,
    seed: int = 2026,
) -> tuple[dict, float, float]:
    """Train a clean model, save state_dict, return (state, acc, seconds)."""
    os.makedirs(ckpt_dir, exist_ok=True)
    path = ckpt_path(ckpt_dir, model_name, int(data.x_train.shape[0]), int(epochs))
    t0 = time.perf_counter()
    torch.manual_seed(int(seed))
    model = model_factory()
    opt = _make_opt(model, lr, momentum)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    n_train = int(data.x_train.shape[0])
    for _epoch in range(int(epochs)):
        model.train()
        perm = torch.randperm(n_train)
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            if idx.numel() < 2:
                continue
            opt.zero_grad(set_to_none=True)
            logits = model(data.x_train[idx])
            if model_name == "toy50":
                loss = bce(logits.reshape(-1), data.y_train[idx].float())
            else:
                loss = ce(logits, data.y_train[idx])
            loss.backward()
            opt.step()
    acc = evaluate(model, data.x_test, data.y_test, model_name, clean_model=None)
    if acc is None:
        acc = chance_acc_of(model_name)
    state = copy.deepcopy(model.state_dict())
    torch.save({"state_dict": state, "acc": acc, "model_name": model_name}, path)
    seconds = time.perf_counter() - t0
    print(f"[ckpt] {model_name} clean acc={acc:.2f}%  {seconds:.2f}s  → {path}")
    return state, float(acc), float(seconds)


def load_or_train_clean(
    *,
    model_factory,
    data,
    model_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    momentum: float,
    ckpt_dir: str,
    seed: int = 2026,
    force: bool = False,
) -> tuple[dict, float]:
    path = ckpt_path(ckpt_dir, model_name, int(data.x_train.shape[0]), int(epochs))
    if (not force) and os.path.isfile(path):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        print(f"[ckpt] loaded {path}  stored_acc={blob.get('acc', float('nan')):.2f}%")
        return blob["state_dict"], float(blob.get("acc", float("nan")))
    state, acc, _s = train_clean_checkpoint(
        model_factory=model_factory,
        data=data,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        momentum=momentum,
        ckpt_dir=ckpt_dir,
        seed=seed,
    )
    return state, acc
