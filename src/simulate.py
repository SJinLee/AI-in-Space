#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI for the reusable multi-model LEO bit-flip suite.

Examples
--------
    python simulate.py --model toy50 --mode infer --rad leo_saa --defense clip --trials 20
    python simulate.py --model mnist_cnn --mode train --rad poisson --train-hours 24 --defense tmr --trials 8
    python simulate.py --preset paper_sweep

ASSUMPTION
    --train-hours 24 maps the whole training run to T = 1.0 LEO day.
    That is a satellite-equivalent exposure, NOT Colab / laptop wall-clock.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from defenses import DEFENSE_NAMES, ENSEMBLE_TRAIN_MODELS
from engine import (
    CKPT_DIR_DEFAULT,
    clean_reference_acc,
    load_or_train_clean,
    run_infer_trial,
    run_train_trial,
)
from models import (
    DATA_DIR,
    MODEL_NAMES,
    RESNET_VARIANT,
    build_model,
    chance_acc_of,
    default_budget,
    describe_model,
    load_data_for,
)
from presets import Cell, apply_pilot_decisions, paper_sweep_cells
from radiation import (
    DEFAULT_R,
    DEFAULT_STRESS_P,
    SAA_PASS_DAYS,
    collect_weight_params,
    n_bits_of,
    parse_allowed_bits,
    self_test_bitflip,
)

MASTER_SEED = 2026
N_THREADS = 4


def parse_T(s: str) -> float:
    s = str(s).strip().lower()
    if s in ("saa_pass", "saa", "one_saa", "pass"):
        return float(SAA_PASS_DAYS)  # 12/1440
    return float(s)


def trial_seed(model: str, mode: str, rad: str, defense: str, trial: int, master: int = MASTER_SEED) -> int:
    """Distinct recorded seed per cell × trial. Derived from MASTER_SEED=2026.

    2026-09-11: stride per cell raised 20 → 1000 so up to 1000 trials per cell
    cannot collide with a neighbouring cell. Seeds therefore differ from the
    2026-08-19 pilot sweep (that sweep stays reproducible from its recorded seeds).
    """
    import hashlib
    key = f"{model}|{mode}|{rad}|{defense}".encode("utf-8")
    h = int(hashlib.md5(key).hexdigest()[:6], 16) % 100_000
    return int(master) * 100_000_000 + h * 1000 + int(trial)


def factory_for(model_name: str):
    def _f():
        return build_model(model_name, try_pretrained=True)

    return _f


def configure_torch() -> None:
    torch.set_num_threads(N_THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


# ---------------------------------------------------------------------------
# Single cell
# ---------------------------------------------------------------------------
def run_cell(
    cell: Cell,
    *,
    r: float,
    flip_bias: bool,
    ckpt_dir: str,
    master_seed: int,
    force_ckpt: bool = False,
    data_cache: dict,
    ckpt_cache: dict,
    budget_overrides: dict,
) -> list[dict]:
    name = cell.model
    bud = dict(default_budget(name))
    bud.update({k: v for k, v in budget_overrides.get(name, {}).items() if v is not None})
    if cell.epochs is not None:
        bud["epochs"] = int(cell.epochs)
    if cell.n_train is not None:
        bud["n_train"] = int(cell.n_train)
    if cell.n_test is not None:
        bud["n_test"] = int(cell.n_test)

    cache_key = (name, bud["n_train"], bud["n_test"])
    if cache_key not in data_cache:
        print(f"[data] loading {name} n_train={bud['n_train']} n_test={bud['n_test']} ...")
        data_cache[cache_key] = load_data_for(name, bud["n_train"], bud["n_test"], toy_seed=master_seed)
    data = data_cache[cache_key]
    fac = factory_for(name)

    # N_bits from a fresh model (and print once)
    probe = fac()
    meta = describe_model(probe, flip_bias=flip_bias)
    n_bits = int(meta["n_bits"])
    print(
        f"[model] {name} n_weight_elements={meta['n_weight_elements']}  "
        f"N_bits={n_bits}  n_bias={meta['n_bias_elements']}  all_params={meta['n_all_params']}"
    )
    del probe

    T_days = float(cell.T_days)
    if cell.mode == "train" and cell.rad != "stress":
        # ASSUMPTION: map the whole training run to train_hours of satellite time.
        T_days = float(cell.train_hours) / 24.0

    rows: list[dict] = []
    clean_state = None
    clean_acc = float("nan")
    if cell.mode == "infer":
        ck = (name, bud["n_train"], int(bud.get("ckpt_epochs", bud["epochs"])))
        if ck not in ckpt_cache:
            state, acc = load_or_train_clean(
                model_factory=fac,
                data=data,
                model_name=name,
                epochs=int(bud.get("ckpt_epochs", bud["epochs"])),
                batch_size=int(bud["batch_size"]),
                lr=float(bud["lr"]),
                momentum=float(bud["momentum"]),
                ckpt_dir=ckpt_dir,
                seed=master_seed,
                force=force_ckpt,
            )
            ckpt_cache[ck] = (state, acc)
        clean_state, clean_acc = ckpt_cache[ck]
        # Reference measured with the same metric as the trials (toy50 → 100 = self-agreement).
        clean_acc = clean_reference_acc(fac, clean_state, data, name)

    n_trials = int(cell.trials)
    for trial in range(n_trials):
        seed = trial_seed(name, cell.mode, cell.rad, cell.defense, trial, master=master_seed)
        if cell.mode == "infer":
            out = run_infer_trial(
                model_factory=fac,
                clean_state=clean_state,
                data=data,
                model_name=name,
                rad=cell.rad,
                defense=cell.defense,
                seed=seed,
                r=r,
                T_days=T_days,
                p=cell.p,
                flip_bias=flip_bias,
                allowed_bits=cell.allowed_bits,
                single_hit=cell.single_hit,
            )
        else:
            out = run_train_trial(
                model_factory=fac,
                data=data,
                model_name=name,
                rad=cell.rad,
                defense=cell.defense,
                seed=seed,
                r=r,
                T_days=T_days,
                p=cell.p,
                flip_bias=flip_bias,
                epochs=int(bud["epochs"]),
                batch_size=int(bud["batch_size"]),
                lr=float(bud["lr"]),
                momentum=float(bud["momentum"]),
                allowed_bits=cell.allowed_bits,
                single_hit=cell.single_hit,
            )
        row = {
            "model": name,
            "mode": cell.mode,
            "rad": cell.rad,
            "defense": cell.defense,
            "trial": int(trial),
            "seed": int(seed),
            "acc": float(out["acc"]),
            "crashed": int(out["crashed"]),
            "k": float(out["k"]),
            "n_bits": int(out["n_bits"]),
            "n_exponent_hits": int(out["n_exponent_hits"]),
            "n_sign_hits": int(out["n_sign_hits"]),
            "T_days": float(T_days),
            "r": float(r),
            "p": float(cell.p) if cell.rad == "stress" else float("nan"),
            "train_hours": float(cell.train_hours) if cell.mode == "train" else float("nan"),
            "epochs": int(bud["epochs"]) if cell.mode == "train" else int(bud.get("ckpt_epochs", bud["epochs"])),
            "allowed_bits": "" if cell.allowed_bits is None else str(cell.allowed_bits),
            "single_hit": int(cell.single_hit),
            "n_train": int(bud["n_train"]),
            "n_test": int(data.x_test.shape[0]),
            "seconds": float(out["seconds"]),
            "clean_acc": float(clean_acc) if cell.mode == "infer" else float("nan"),
            "last_finite_acc": float(out["last_finite_acc"]),
            "ensemble_skipped": int(out.get("ensemble_skipped", 0)),
            "note": "; ".join(s for s in (cell.note, out.get("note", "")) if s),
        }
        rows.append(row)
        if trial == 0 or trial + 1 == n_trials or (trial + 1) % 5 == 0:
            print(
                f"  [{name} {cell.mode} {cell.rad} {cell.defense}  "
                f"{trial+1}/{n_trials}] acc={row['acc']:.2f} crash={row['crashed']} "
                f"k={row['k']:.1f} {row['seconds']:.2f}s"
            )
    return rows


# ---------------------------------------------------------------------------
# Summary + plots
# ---------------------------------------------------------------------------
SURVIVAL_FRACTION = 0.9  # a trial "survives" if acc >= 0.9 * clean reference and it did not crash


def add_survival_column(df: pd.DataFrame) -> pd.DataFrame:
    """survived = not crashed and acc >= SURVIVAL_FRACTION * clean reference.

    Outcomes are bimodal (fine or dead), so mean ± std is misleading; report
    survival_rate and median instead. Train rows have no clean_acc: borrow the
    model's infer reference if the sweep contains one, else NaN.
    """
    df = df.copy()
    ref = pd.to_numeric(df.get("clean_acc"), errors="coerce")
    if "mode" in df:
        infer_ref = df[df["mode"] == "infer"].groupby("model")["clean_acc"].first()
        ref = ref.where(ref.notna(), df["model"].map(infer_ref))
    ok = (df["crashed"] == 0) & (df["acc"] >= SURVIVAL_FRACTION * ref)
    df["survived"] = ok.astype(float).where(ref.notna(), np.nan)
    return df


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "mode", "rad", "defense"]
    if "single_hit" in df:
        keys += ["allowed_bits", "single_hit"]
    df = add_survival_column(df)
    rows = []
    for key, g in df.groupby(keys, sort=True, dropna=False):
        acc = g["acc"]
        q25, q75 = float(acc.quantile(0.25)), float(acc.quantile(0.75))
        row = dict(zip(keys, key))
        row.update(
            {
                "n": int(len(g)),
                "mean_acc": float(acc.mean()),
                "std_acc": float(acc.std(ddof=1)) if len(g) > 1 else 0.0,
                "median_acc": float(acc.median()),
                "iqr_acc": q75 - q25,
                "crash_rate": float(g["crashed"].mean()),
                "survival_rate": float(g["survived"].mean()) if g["survived"].notna().any() else float("nan"),
                "mean_k": float(g["k"].mean()),
                "n_bits": int(g["n_bits"].iloc[0]),
                "mean_seconds": float(g["seconds"].mean()),
                "T_days": float(g["T_days"].mean()),
                "p": float(g["p"].mean()) if "p" in g else float("nan"),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _boxplot(ax, data, labels, **kw):
    """matplotlib >= 3.9 renamed labels= to tick_labels=; support both."""
    try:
        return ax.boxplot(data, tick_labels=labels, **kw)
    except TypeError:
        return ax.boxplot(data, labels=labels, **kw)


def _style(ax):
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_infer_leo_by_defense(df: pd.DataFrame, path: str) -> None:
    sub = df[(df["mode"] == "infer") & (df["rad"] == "leo_saa")]
    if sub.empty:
        return
    models = list(sub["model"].unique())
    n = len(models)
    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows), squeeze=False)
    colors = {
        "none": "#c0392b",
        "clip": "#2471a3",
        "tmr": "#1e8449",
        "ensemble": "#6c3483",
        "ensemble_median": "#a569bd",
        "parity": "#d68910",
    }
    for i, m in enumerate(models):
        ax = axes[i // ncols][i % ncols]
        g = sub[sub["model"] == m]
        defs = list(g["defense"].unique())
        data, labs, cols = [], [], []
        for d in defs:
            data.append(g[g["defense"] == d]["acc"].to_numpy())
            labs.append(d)
            cols.append(colors.get(d, "#7f8c8d"))
        bp = _boxplot(ax, data, labs, patch_artist=True, showfliers=True)
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
        ax.set_title(m)
        ax.set_ylabel("Accuracy (%)")
        ax.set_xlabel("Defense")
        ax.set_ylim(0, 105)
        _style(ax)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle("Inference accuracy under leo_saa (T = 1 LEO day) by defense", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[write] {path}")


def plot_stress_vs_leo(df: pd.DataFrame, path: str) -> None:
    sub = df[(df["mode"] == "infer") & (df["rad"].isin(["leo_saa", "stress"]))]
    if sub.empty:
        return
    models = list(sub["model"].unique())
    n = len(models)
    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.6 * nrows), squeeze=False)
    for i, m in enumerate(models):
        ax = axes[i // ncols][i % ncols]
        g = sub[sub["model"] == m]
        series, labels, cols = [], [], []
        for rad, col in (("leo_saa", "#1f6aa5"), ("stress", "#c0392b")):
            for d in g["defense"].unique():
                vals = g[(g["rad"] == rad) & (g["defense"] == d)]["acc"].to_numpy()
                if vals.size == 0:
                    continue
                series.append(vals)
                labels.append(f"{rad}\n{d}")
                cols.append(col if rad == "leo_saa" else "#c0392b")
        if not series:
            ax.axis("off")
            continue
        bp = _boxplot(ax, series, labels, patch_artist=True, showfliers=True)
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.set_title(m)
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 105)
        ax.tick_params(axis="x", labelsize=7)
        _style(ax)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle("Inference accuracy: physical leo_saa (T=1 day) vs course-style stress (p=0.01)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[write] {path}")


def plot_crash_rate(df: pd.DataFrame, path: str) -> None:
    if df.empty or float(df["crashed"].max()) <= 0:
        # still write a "no crashes" figure so the artifact exists
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.text(0.5, 0.5, "No NaN/Inf crashes recorded in this sweep.", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[write] {path} (no crashes)")
        return
    g = (
        df.groupby(["model", "mode", "rad", "defense"], as_index=False)["crashed"]
        .mean()
        .rename(columns={"crashed": "crash_rate"})
    )
    g = g[g["crash_rate"] > 0].copy()
    if g.empty:
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.text(0.5, 0.5, "No NaN/Inf crashes recorded in this sweep.", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[write] {path} (no crashes)")
        return
    labels = [f"{r.model}\n{r.mode}/{r.rad}/{r.defense}" for r in g.itertuples()]
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(labels)), 4.5))
    ax.bar(np.arange(len(g)), g["crash_rate"].to_numpy(), color="#c0392b", edgecolor="black", linewidth=0.4)
    ax.set_xticks(np.arange(len(g)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Crash rate (NaN/Inf)")
    ax.set_title("Training/inference crash rate where > 0")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[write] {path}")


def write_notes(
    path: str,
    *,
    summary: pd.DataFrame,
    raw: pd.DataFrame,
    nbits: dict[str, int],
    timings: dict,
    resnet_variant: str,
    decisions: list[str],
    wall_s: float,
) -> None:
    lines = []
    lines.append("# SEU suite — paper_sweep notes\n")
    lines.append("## Physics / time ASSUMPTIONS (not wall-clock)\n")
    lines.append("- True SEU = XOR of one random bit of a float32 weight (uint32 view). **Not** Gaussian noise.\n")
    lines.append("- Default rate `r = 1e-6` SEU/bit/day. `λ = r * N_bits * Δt_days`.\n")
    lines.append("- Inference exposure **T = 1.0 LEO day** (also `--T saa_pass` = 12/1440 day).\n")
    lines.append("- Training uses `--train-hours 24` so the **entire training run is mapped to 1.0 satellite day**.")
    lines.append("  This is an ASSUMPTION so λ is nontrivial. A 2-second Colab train is **not** a real satellite wall-clock.\n")
    lines.append("- `leo_saa`: 3 SAA passes × 12 min / day; ~90% of expected daily SEUs in those passes")
    lines.append("  (`r_saa ≈ 36 r_avg`), remainder in quiet time.\n")
    lines.append("- `stress`: ignore physical r; each weight independently hit with probability p=0.01 (course accelerator).\n")
    lines.append("- Only weight tensors with dim ≥ 2 are flipped (no biases unless `--flip-bias`).\n")
    lines.append("\n## N_bits per model (weight tensors only)\n")
    for m, nb in nbits.items():
        n_w = nb // 32
        lam = 1e-6 * nb * 1.0
        lines.append(f"- **{m}**: N_weights={n_w:,}  N_bits={nb:,}  λ(1 day, r=1e-6)≈{lam:.4g}\n")
    lines.append(f"\n## resnet18_t10 variant\n\n`{resnet_variant}`\n")
    lines.append("Input is 32×32; images are **not** upscaled to 224.\n")
    lines.append("\n## What ran / pilot decisions\n")
    for d in decisions:
        lines.append(f"- {d}\n")
    lines.append("\n## Timings (seconds)\n")
    for k, v in timings.items():
        lines.append(f"- {k}: {v:.3f}s\n")
    lines.append(f"\n- **total wall**: {wall_s:.1f}s ({wall_s/60:.1f} min)\n")
    lines.append("\n## Ensemble training\n")
    lines.append(f"Logit-average ensemble during training is implemented only for {sorted(ENSEMBLE_TRAIN_MODELS)}.")
    lines.append(" CNN / ResNet ensemble is inference-only.\n")
    lines.append("\n## Summary table\n\n")
    if not summary.empty:
        lines.append(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        lines.append("\n")
    lines.append(f"\nRaw trials: {len(raw)}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[write] {path}")


# ---------------------------------------------------------------------------
# Pilot
# ---------------------------------------------------------------------------
def pilot_model(name: str, data, fac, bud, ckpt_dir, master_seed, r, flip_bias) -> dict:
    """Time a clean train (1 epoch) and a clean infer eval; return seconds + acc."""
    info = {"model": name}
    t0 = time.perf_counter()
    # 1-epoch train timing on current subset
    out = run_train_trial(
        model_factory=fac,
        data=data,
        model_name=name,
        rad="poisson",
        defense="none",
        seed=master_seed,
        r=r,
        T_days=1.0,
        p=0.01,
        flip_bias=flip_bias,
        epochs=1,
        batch_size=int(bud["batch_size"]),
        lr=float(bud["lr"]),
        momentum=float(bud["momentum"]),
    )
    info["train_1ep_s"] = float(out["seconds"])
    info["train_1ep_acc"] = float(out["acc"])
    # infer: need a ckpt; train a short clean one if missing
    state, acc = load_or_train_clean(
        model_factory=fac,
        data=data,
        model_name=name,
        epochs=max(1, int(bud.get("ckpt_epochs", bud["epochs"]))),
        batch_size=int(bud["batch_size"]),
        lr=float(bud["lr"]),
        momentum=float(bud["momentum"]),
        ckpt_dir=ckpt_dir,
        seed=master_seed,
        force=False,
    )
    t1 = time.perf_counter()
    inf = run_infer_trial(
        model_factory=fac,
        clean_state=state,
        data=data,
        model_name=name,
        rad="leo_saa",
        defense="none",
        seed=master_seed + 1,
        r=r,
        T_days=1.0,
        p=0.01,
        flip_bias=flip_bias,
    )
    info["infer_s"] = float(inf["seconds"])
    info["infer_acc"] = float(inf["acc"])
    info["clean_acc"] = float(acc)
    info["ckpt_s"] = float(t1 - t0)  # includes the 1ep train above roughly
    print(
        f"[pilot] {name}: train_1ep={info['train_1ep_s']:.2f}s acc={info['train_1ep_acc']:.1f}%  "
        f"infer={info['infer_s']:.2f}s acc={info['infer_acc']:.1f}%  clean_ckpt_acc={acc:.1f}%"
    )
    return info


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Multi-model LEO bit-flip SEU suite")
    p.add_argument("--model", choices=MODEL_NAMES, default="toy50")
    p.add_argument("--mode", choices=("infer", "train"), default="infer")
    p.add_argument("--rad", choices=("poisson", "leo_saa", "stress"), default="leo_saa")
    p.add_argument("--defense", choices=DEFENSE_NAMES, default="none")
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--r", type=float, default=DEFAULT_R, help="SEU/bit/day (ignored by stress)")
    p.add_argument("--T", type=str, default="1.0", help="Exposure in days, or 'saa_pass' (=12/1440)")
    p.add_argument("--p", type=float, default=DEFAULT_STRESS_P, help="stress per-weight hit probability")
    p.add_argument("--train-hours", type=float, default=24.0,
                   help="ASSUMPTION: map the training run to this many hours of satellite time (24 → 1 LEO day)")
    p.add_argument("--flip-bias", action="store_true", help="Also flip dim-1 bias tensors")
    p.add_argument("--allowed-bits", type=str, default=None,
                   help="Restrict flips to these bit positions: '30', '23-30', '31,30', or sign|exponent|mantissa")
    p.add_argument("--single-hit", action="store_true",
                   help="Exactly ONE (weight, bit) event per trial (bit-criticality experiment); ignores r/T/p")
    p.add_argument("--tag", type=str, default="", help="Extra tag for the output CSV name")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--n-test", type=int, default=None)
    p.add_argument("--preset", choices=("paper_sweep",), default=None)
    p.add_argument("--out-dir", type=str, default=os.path.join(HERE, "results"))
    p.add_argument("--ckpt-dir", type=str, default=CKPT_DIR_DEFAULT)
    p.add_argument("--seed-master", type=int, default=MASTER_SEED)
    p.add_argument("--force-ckpt", action="store_true")
    return p


def run_paper_sweep(args) -> None:
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.abspath(args.ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    t_all = time.perf_counter()
    self_test_bitflip()
    print("[self-test] bit-flip OK")

    cells = paper_sweep_cells()
    data_cache: dict = {}
    ckpt_cache: dict = {}
    budget_overrides: dict[str, dict] = {}
    timings: dict[str, float] = {}
    nbits: dict[str, int] = {}
    decisions: list[str] = []

    # Probe N_bits for every model
    for name in MODEL_NAMES:
        m = build_model(name, try_pretrained=True)
        meta = describe_model(m, flip_bias=args.flip_bias)
        nbits[name] = int(meta["n_bits"])
        print(f"[N_bits] {name}: weights={meta['n_weight_elements']:,}  bits={meta['n_bits']:,}  "
              f"λ_day={1e-6 * meta['n_bits']:.4g}")
        del m

    # Pilot each model that appears
    models_needed = sorted({c.model for c in cells})
    train_s: dict[str, float] = {}
    infer_s: dict[str, float] = {}
    epoch_caps: dict[str, int] = {}
    n_train_caps: dict[str, int] = {}

    for name in models_needed:
        bud = dict(default_budget(name))
        # shrink CIFAR train subset up front (CPU)
        if name in ("resnet18_t10", "cifar_deep"):
            bud["n_train"] = min(bud["n_train"], 10_000)
            bud["n_test"] = min(bud["n_test"], 2_000)
        data = load_data_for(name, bud["n_train"], bud["n_test"], toy_seed=args.seed_master)
        data_cache[(name, bud["n_train"], bud["n_test"])] = data
        fac = factory_for(name)
        info = pilot_model(name, data, fac, bud, ckpt_dir, args.seed_master, args.r, args.flip_bias)
        timings[f"pilot_train_1ep_{name}"] = info["train_1ep_s"]
        timings[f"pilot_infer_{name}"] = info["infer_s"]
        train_s[name] = info["train_1ep_s"]
        infer_s[name] = info["infer_s"]

        # Epoch / subset policy from the spec
        cap_ep = int(bud["epochs"])
        cap_n = int(bud["n_train"])
        if name == "mnist_linear":
            # 2–3 epochs full MNIST if <5s/run else 1 epoch
            if info["train_1ep_s"] < 5.0:
                cap_ep = 2
            else:
                cap_ep = 1
                decisions.append(f"mnist_linear: train_1ep={info['train_1ep_s']:.2f}s ≥5s → 1 epoch")
        elif name == "mnist_cnn":
            if info["train_1ep_s"] > 25.0:
                cap_ep = 1
                decisions.append(f"mnist_cnn: train_1ep={info['train_1ep_s']:.2f}s >25s → 1 epoch")
            else:
                cap_ep = 2
        elif name == "resnet18_t10":
            cap_ep = 1
            if info["train_1ep_s"] >= 180.0:
                decisions.append(f"resnet18_t10: train_1ep={info['train_1ep_s']:.2f}s ≥180s → infer only")
        elif name == "cifar_deep":
            cap_ep = 1
            if info["train_1ep_s"] >= 180.0:
                decisions.append(f"cifar_deep: train_1ep={info['train_1ep_s']:.2f}s ≥180s → infer only")
        elif name == "toy50":
            cap_ep = int(bud["epochs"])
        epoch_caps[name] = cap_ep
        n_train_caps[name] = cap_n
        budget_overrides[name] = {
            "epochs": cap_ep,
            "n_train": cap_n,
            "n_test": bud["n_test"],
            "ckpt_epochs": int(bud.get("ckpt_epochs", cap_ep)),
        }
        decisions.append(
            f"{name}: epochs={cap_ep} n_train={cap_n} n_test={bud['n_test']} "
            f"train_1ep={info['train_1ep_s']:.2f}s infer={info['infer_s']:.2f}s "
            f"clean_acc={info['clean_acc']:.2f}%"
        )

    cells = apply_pilot_decisions(
        cells,
        train_seconds=train_s,
        infer_seconds=infer_s,
        epoch_caps=epoch_caps,
        n_train_caps=n_train_caps,
    )
    print(f"[plan] {len(cells)} cells, total trials={sum(c.trials for c in cells)}")

    all_rows: list[dict] = []
    for i, cell in enumerate(cells, 1):
        # push pilot epoch into the cell
        ov = budget_overrides.get(cell.model, {})
        if cell.epochs is None and "epochs" in ov:
            cell.epochs = ov["epochs"]
        print(f"\n==== cell {i}/{len(cells)}  {cell.model} {cell.mode} {cell.rad} {cell.defense} "
              f"trials={cell.trials} epochs={cell.epochs} ====")
        rows = run_cell(
            cell,
            r=args.r,
            flip_bias=args.flip_bias,
            ckpt_dir=ckpt_dir,
            master_seed=args.seed_master,
            force_ckpt=args.force_ckpt,
            data_cache=data_cache,
            ckpt_cache=ckpt_cache,
            budget_overrides=budget_overrides,
        )
        all_rows.extend(rows)

    raw = pd.DataFrame(all_rows)
    raw_path = os.path.join(out_dir, "sweep_raw.csv")
    raw.to_csv(raw_path, index=False)
    print(f"[write] {raw_path}  n={len(raw)}")

    summary = make_summary(raw)
    sum_path = os.path.join(out_dir, "sweep_summary.csv")
    summary.to_csv(sum_path, index=False, float_format="%.4f")
    print(f"[write] {sum_path}")
    print("\n===== SWEEP SUMMARY =====")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    plot_infer_leo_by_defense(raw, os.path.join(out_dir, "acc_infer_leo_saa_by_defense.png"))
    plot_stress_vs_leo(raw, os.path.join(out_dir, "acc_stress_vs_leo.png"))
    plot_crash_rate(raw, os.path.join(out_dir, "crash_rate.png"))

    wall = time.perf_counter() - t_all
    import models as _models
    rv = _models.RESNET_VARIANT

    write_notes(
        os.path.join(out_dir, "NOTES.md"),
        summary=summary,
        raw=raw,
        nbits=nbits,
        timings=timings,
        resnet_variant=rv,
        decisions=decisions,
        wall_s=wall,
    )
    print(f"[done] wall={wall:.1f}s  ({wall/60:.1f} min)")


def run_single(args) -> None:
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    self_test_bitflip()
    print("[self-test] bit-flip OK")
    T_days = parse_T(args.T)
    cell = Cell(
        model=args.model,
        mode=args.mode,
        rad=args.rad,
        defense=args.defense,
        trials=int(args.trials),
        p=float(args.p),
        T_days=T_days,
        train_hours=float(args.train_hours),
        epochs=args.epochs,
        n_train=args.n_train,
        n_test=args.n_test,
        allowed_bits=(None if args.allowed_bits is None else ",".join(str(b) for b in parse_allowed_bits(args.allowed_bits))),
        single_hit=bool(args.single_hit),
    )
    rows = run_cell(
        cell,
        r=args.r,
        flip_bias=args.flip_bias,
        ckpt_dir=os.path.abspath(args.ckpt_dir),
        master_seed=args.seed_master,
        force_ckpt=args.force_ckpt,
        data_cache={},
        ckpt_cache={},
        budget_overrides={},
    )
    df = pd.DataFrame(rows)
    tag = f"{args.model}_{args.mode}_{args.rad}_{args.defense}"
    if args.single_hit:
        tag += "_1hit"
    elif args.rad == "stress":
        tag += f"_p{args.p:g}"
    else:
        tag += f"_T{T_days:g}_r{args.r:g}"
    if args.allowed_bits:
        tag += "_bits" + str(args.allowed_bits).replace(",", "+")
    if args.tag:
        tag += "_" + args.tag
    path = os.path.join(out_dir, f"single_{tag}.csv")
    df.to_csv(path, index=False)
    print(f"[write] {path}")
    print(df[["trial", "seed", "acc", "crashed", "k", "n_bits", "seconds"]].to_string(index=False))
    summ = make_summary(df).iloc[0]
    print(
        f"mean_acc={summ['mean_acc']:.2f} ± {summ['std_acc']:.2f}  median={summ['median_acc']:.2f}  "
        f"survival={summ['survival_rate']:.2f}  crash={summ['crash_rate']:.2f}  "
        f"mean_k={summ['mean_k']:.2f}  N_bits={int(summ['n_bits'])}  clean_ref={df['clean_acc'].iloc[0]:.2f}"
    )


def main() -> None:
    configure_torch()
    args = build_parser().parse_args()
    print(f"[env] torch={torch.__version__} threads={torch.get_num_threads()}  data={DATA_DIR}")
    print(f"[env] r={args.r:g}  train-hours={args.train_hours} (ASSUMPTION, not wall-clock)")
    if args.preset == "paper_sweep":
        run_paper_sweep(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
