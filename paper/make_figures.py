#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
논문용 그림·표 생성 스크립트.

    python paper/make_figures.py

src/results/single_*.csv 를 읽어
    paper/figures/fig_e1_dose_response.pdf (.png)
    paper/figures/fig_e1_survival_crash.pdf (.png)
    paper/figures/fig_e1_trials.pdf (.png)
    paper/tables/table_e1.tex
    paper/tables/numbers.tex      (본문에서 \\num... 매크로로 쓰는 핵심 수치)
를 만든다. 수치는 전부 CSV에서 계산하므로 손으로 옮겨 적지 않는다.
"""
from __future__ import annotations

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, NullFormatter

PLAIN = FuncFormatter(lambda v, _: f"{v:g}")  # 10^{-3} 대신 0.001 (한글 글꼴에 − 글리프가 없음)


def log_axis(ax, which="x"):
    axis = ax.xaxis if which == "x" else ax.yaxis
    axis.set_major_formatter(PLAIN)
    axis.set_minor_formatter(NullFormatter())

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "src", "results")
FIG_DIR = os.path.join(HERE, "figures")
TAB_DIR = os.path.join(HERE, "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

R_SEU = 1e-6  # SEU / bit / day (README 기본값, 가정)
COLORS = {"none": "#C94A24", "clip": "#245EA6"}  # dataviz 검증 통과 (ΔE 20.5 protan)
MARKERS = {"none": "o", "clip": "s"}
LABELS = {"none": "방어 없음 (none)", "clip": "클리핑 (clip)"}

plt.rcParams.update({
    "font.family": ["Malgun Gothic", "NanumGothic", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "legend.frameon": False,
})


def load() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(RESULTS, "single_mnist_linear_infer_stress_*.csv")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[(df.single_hit == 0) & (df.allowed_bits.isna()) & (df.defense.isin(["none", "clip"]))].copy()
    df["survived"] = (df.acc >= 0.9 * df.clean_acc).astype(float)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["defense", "p"])
        .agg(
            n=("acc", "size"),
            median_acc=("acc", "median"),
            q25=("acc", lambda s: s.quantile(0.25)),
            q75=("acc", lambda s: s.quantile(0.75)),
            min_acc=("acc", "min"),
            mean_acc=("acc", "mean"),
            survival=("survived", "mean"),
            crash=("crashed", "mean"),
            mean_k=("k", "mean"),
            mean_exp=("n_exponent_hits", "mean"),
        )
        .reset_index()
        .sort_values(["defense", "p"])
    )
    n_bits = int(df.n_bits.iloc[0])
    agg["leo_days"] = agg.p / (32.0 * R_SEU)  # E[k] 가 같아지는 LEO 노출 일수
    agg.attrs["n_bits"] = n_bits
    agg.attrs["n_weights"] = n_bits // 32
    agg.attrs["clean_acc"] = float(df.clean_acc.iloc[0])
    return agg


def p_at_survival_half(g: pd.DataFrame) -> float:
    """생존률이 0.5를 처음 가로지르는 p (log p 선형 보간)."""
    g = g.sort_values("p")
    x, y = np.log10(g.p.values), g.survival.values
    for i in range(1, len(y)):
        if y[i - 1] > 0.5 >= y[i]:
            t = (y[i - 1] - 0.5) / (y[i - 1] - y[i])
            return float(10 ** (x[i - 1] + t * (x[i] - x[i - 1])))
    return float("nan")


def fig_dose_response(agg: pd.DataFrame):
    clean = agg.attrs["clean_acc"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for d, g in agg.groupby("defense"):
        c = COLORS[d]
        x = g.p * 100
        ax.fill_between(x, g.q25, g.q75, color=c, alpha=0.15, linewidth=0)
        ax.plot(x, g.median_acc, marker=MARKERS[d], ms=5, lw=2, color=c, label=f"{LABELS[d]} 중앙값")
    ax.axhline(clean, ls="--", lw=1, color="#555", label=f"무손상 정확도 {clean:.1f}%")
    ax.axhline(0.9 * clean, ls=":", lw=1, color="#555", label="생존 기준선 (무손상의 90%)")
    ax.set_xscale("log")
    log_axis(ax)
    ax.set_xlabel("가중치당 비트플립 확률 $p$ (%)")
    ax.set_ylabel("MNIST 시험 정확도 (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, loc="lower left")
    # 위쪽 보조축: 같은 기대 명중 수를 주는 LEO 노출 일수
    sec = ax.secondary_xaxis("top", functions=(lambda x: x / 100 / (32 * R_SEU), lambda d: d * 32 * R_SEU * 100))
    sec.xaxis.set_major_formatter(PLAIN)
    sec.xaxis.set_minor_formatter(NullFormatter())
    sec.set_xlabel("같은 기대 명중 수를 주는 LEO 노출 일수 (r = 1e-6 /bit/day 가정)", fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig_e1_dose_response.{ext}"), dpi=200)
    plt.close(fig)


def fig_survival_crash(agg: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3), sharex=True)
    for d, g in agg.groupby("defense"):
        c, m = COLORS[d], MARKERS[d]
        axes[0].plot(g.p * 100, g.survival, marker=m, ms=5, lw=2, color=c, label=LABELS[d])
        axes[1].plot(g.p * 100, g.crash, marker=m, ms=5, lw=2, color=c, label=LABELS[d])
    for ax, title, yl in zip(axes, ["(a) 생존률", "(b) 붕괴율 (NaN/∞ 발생)"], ["생존률", "붕괴율"]):
        ax.set_xscale("log")
        log_axis(ax)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("가중치당 비트플립 확률 $p$ (%)")
        ax.set_ylabel(yl)
        ax.set_title(title, fontsize=10)
    axes[0].axhline(0.5, ls=":", lw=1, color="#555")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig_e1_survival_crash.{ext}"), dpi=200)
    plt.close(fig)


def fig_trials(df: pd.DataFrame, agg: pd.DataFrame):
    """시행 하나하나를 점으로: 중간 선량대의 이봉 분포를 보여준다."""
    clean = agg.attrs["clean_acc"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    rng = np.random.default_rng(0)
    for d in ["none", "clip"]:
        g = df[df.defense == d]
        jitter = 10 ** (rng.uniform(-0.04, 0.04, len(g)))
        ax.scatter(g.p * 100 * jitter, g.acc, s=16, marker=MARKERS[d], color=COLORS[d],
                   alpha=0.75, edgecolors="white", linewidths=0.4, label=LABELS[d])
    ax.axhline(0.9 * clean, ls=":", lw=1, color="#555", label="생존 기준선 (무손상의 90%)")
    ax.axhline(10, ls="--", lw=1, color="#999", label="찍기 수준 (10%)")
    ax.set_xscale("log")
    log_axis(ax)
    ax.set_xlabel("가중치당 비트플립 확률 $p$ (%)")
    ax.set_ylabel("MNIST 시험 정확도 (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig_e1_trials.{ext}"), dpi=200)
    plt.close(fig)


def fmt_p(p: float) -> str:
    return f"{p * 100:g}"


def table_e1(agg: pd.DataFrame):
    none = agg[agg.defense == "none"].set_index("p")
    clip = agg[agg.defense == "clip"].set_index("p")
    lines = [
        r"\begin{tabular}{rrr|rrrr|rrrr}",
        r"\toprule",
        r"$p$ (\%) & $\bar{k}$ & LEO 일수 & \multicolumn{4}{c|}{방어 없음} & \multicolumn{4}{c}{클리핑} \\",
        r" & & & 중앙값 & IQR & 생존률 & 붕괴율 & 중앙값 & IQR & 생존률 & 붕괴율 \\",
        r"\midrule",
    ]
    for p in none.index:
        a, b = none.loc[p], clip.loc[p]
        lines.append(
            f"{fmt_p(p)} & {a.mean_k:,.0f} & {a.leo_days:,.1f} & "
            f"{a.median_acc:.1f} & {a.q25:.1f}--{a.q75:.1f} & {a.survival:.1f} & {a.crash:.1f} & "
            f"{b.median_acc:.1f} & {b.q25:.1f}--{b.q75:.1f} & {b.survival:.1f} & {b.crash:.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TAB_DIR, "table_e1.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def numbers(df: pd.DataFrame, agg: pd.DataFrame):
    none = agg[agg.defense == "none"]
    clip = agg[agg.defense == "clip"]
    p50_none = p_at_survival_half(none)
    p50_clip = p_at_survival_half(clip)
    n_bits = agg.attrs["n_bits"]
    vals = {
        "NumTrials": f"{len(df)}",
        "NumP": f"{df.p.nunique()}",
        "NumWeights": f"{agg.attrs['n_weights']:,}",
        "NumBits": f"{n_bits:,}",
        "CleanAcc": f"{agg.attrs['clean_acc']:.2f}",
        "LeoFlipsPerDay": f"{R_SEU * n_bits:.2f}",
        "PhalfNone": f"{p50_none * 100:.3f}",
        "PhalfClip": f"{p50_clip * 100:.2f}",
        "PhalfRatio": f"{p50_clip / p50_none:.0f}",
        "PhalfNoneDays": f"{p50_none / (32 * R_SEU):.0f}",
        "PhalfClipDays": f"{p50_clip / (32 * R_SEU):.0f}",
        "ExpHitFrac": f"{100 * df.n_exponent_hits.sum() / df.k.sum():.1f}",
        "SignHitFrac": f"{100 * df.n_sign_hits.sum() / df.k.sum():.1f}",
        "MaxSeconds": f"{df.seconds.max():.2f}",
    }
    # 특정 p 의 값들 (본문 서술용)
    # LaTeX 매크로 이름에는 숫자를 쓸 수 없으므로 영문으로 적는다 (EmFive = e-5)
    for p, tag in [(3e-5, "pThreeEmFive"), (1e-4, "pOneEmFour"), (3e-4, "pThreeEmFour"), (1e-3, "pOneEmThree"),
                   (3e-3, "pThreeEmThree"), (1e-2, "pOneEmTwo"), (5e-2, "pFiveEmTwo"), (1e-1, "pOneEmOne")]:
        for d, g in [("None", none), ("Clip", clip)]:
            r = g[np.isclose(g.p, p)].iloc[0]
            vals[f"{tag}{d}Med"] = f"{r.median_acc:.1f}"
            vals[f"{tag}{d}Surv"] = f"{r.survival:.1f}"
            vals[f"{tag}{d}Crash"] = f"{r.crash:.1f}"
            vals[f"{tag}{d}Min"] = f"{r.min_acc:.1f}"
            vals[f"{tag}{d}K"] = f"{r.mean_k:.0f}"
    with open(os.path.join(TAB_DIR, "numbers.tex"), "w", encoding="utf-8") as f:
        f.write("% 자동 생성: paper/make_figures.py — 손으로 고치지 말 것\n")
        for k, v in vals.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    return vals


# ---------------------------------------------------------------------------
# E1-M: 모델 크기 축 (mnist_linear / mnist_cnn / cifar_deep / resnet18_t10)
# ---------------------------------------------------------------------------
MODEL_ORDER = ["mnist_linear", "mnist_cnn", "cifar_deep", "resnet18_t10"]
MODEL_LABEL = {
    "mnist_linear": "mnist_linear (FC 2층)",
    "mnist_cnn": "mnist_cnn (Conv 2 + FC 2)",
    "cifar_deep": "cifar_deep (Conv 6 + FC 1)",
    "resnet18_t10": "resnet18_t10 (ResNet-18)",
}
MODEL_TAG = {"mnist_linear": "Linear", "mnist_cnn": "Cnn", "cifar_deep": "Cifar", "resnet18_t10": "Resnet"}
# dataviz 검증 통과 (worst adjacent ΔE 9.1 protan; 대비 경고는 표 4·마커로 보완)
MODEL_COLORS = {"mnist_linear": "#2a78d6", "mnist_cnn": "#eb6834", "cifar_deep": "#1baf7a", "resnet18_t10": "#eda100"}
MODEL_MARKERS = {"mnist_linear": "o", "mnist_cnn": "s", "cifar_deep": "^", "resnet18_t10": "D"}


def load_models() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(RESULTS, "single_*_infer_stress_*.csv")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[(df.single_hit == 0) & (df.allowed_bits.isna()) & (df.model.isin(MODEL_ORDER)) & (df.defense.isin(["none", "clip"]))].copy()
    df["survived"] = (df.acc >= 0.9 * df.clean_acc).astype(float)
    return df


def summarize_models(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["model", "defense", "p"])
        .agg(n=("acc", "size"), median_acc=("acc", "median"), survival=("survived", "mean"),
             crash=("crashed", "mean"), mean_k=("k", "mean"), n_bits=("n_bits", "first"),
             clean=("clean_acc", "first"))
        .reset_index()
    )
    agg["rel_median"] = 100 * agg.median_acc / agg.clean
    agg["n_w"] = agg.n_bits // 32
    agg["model"] = pd.Categorical(agg.model, MODEL_ORDER, ordered=True)
    return agg.sort_values(["model", "defense", "p"])


def fig_models(aggm: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.0), sharex=True)
    for j, d in enumerate(["none", "clip"]):
        for m in MODEL_ORDER:
            g = aggm[(aggm.model == m) & (aggm.defense == d)]
            if g.empty:
                continue
            c, mk = MODEL_COLORS[m], MODEL_MARKERS[m]
            axes[0, j].plot(g.p * 100, g.rel_median, marker=mk, ms=5, lw=2, color=c, label=MODEL_LABEL[m])
            axes[1, j].plot(g.p * 100, g.survival, marker=mk, ms=5, lw=2, color=c, label=MODEL_LABEL[m])
        axes[0, j].set_title(f"({'ab'[j]}) {LABELS[d]}: 중앙값 정확도 (무손상 대비 %)", fontsize=9)
        axes[1, j].set_title(f"({'cd'[j]}) {LABELS[d]}: 생존률", fontsize=9)
        axes[0, j].axhline(90, ls=":", lw=1, color="#555")
        axes[1, j].axhline(0.5, ls=":", lw=1, color="#555")
        axes[0, j].set_ylim(0, 105)
        axes[1, j].set_ylim(-0.05, 1.05)
        axes[1, j].set_xlabel("가중치당 비트플립 확률 $p$ (%)")
        for ax in axes[:, j]:
            ax.set_xscale("log")
            log_axis(ax)
    axes[0, 0].set_ylabel("무손상 대비 정확도 (%)")
    axes[1, 0].set_ylabel("생존률")
    axes[1, 0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig_e1_models.{ext}"), dpi=200)
    plt.close(fig)


def fig_models_hits(aggm: pd.DataFrame):
    """가로축을 '기대 명중 수 k' 로 바꾸면 모델 크기 효과가 사라지는지 본다."""
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3), sharey=True)
    for j, d in enumerate(["none", "clip"]):
        ax = axes[j]
        for m in MODEL_ORDER:
            g = aggm[(aggm.model == m) & (aggm.defense == d)]
            if g.empty:
                continue
            ax.plot(g.mean_k, g.survival, marker=MODEL_MARKERS[m], ms=5, lw=2, color=MODEL_COLORS[m], label=MODEL_LABEL[m])
        ax.axhline(0.5, ls=":", lw=1, color="#555")
        ax.set_xscale("log")
        log_axis(ax)
        ax.set_xlabel("평균 명중 수 $\\bar{k}$ (뒤집힌 비트 수)")
        ax.set_title(f"({'ab'[j]}) {LABELS[d]}", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("생존률")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig_e1_models_hits.{ext}"), dpi=200)
    plt.close(fig)


def table_models(aggm: pd.DataFrame) -> dict:
    """표 4: 모델별 N_w, N_bits, 무손상 정확도, 하루 명중 수, 생존률 50% 지점 (p, k, LEO 일수)."""
    vals = {}
    lines = [
        r"\begin{tabular}{lrrrr|rrr|rr}",
        r"\toprule",
        r"모델 & $N_w$ & $N_{\mathrm{bits}}$ & 무손상 & 명중/일 & \multicolumn{3}{c|}{방어 없음, 생존률 50\%} & \multicolumn{2}{c}{클리핑, 생존률 50\%} \\",
        r" & & & 정확도 & ($r$=$10^{-6}$) & $p_{50}$ (\%) & $k_{50}$ & LEO 일수 & $p_{50}$ (\%) & LEO 일수 \\",
        r"\midrule",
    ]
    short = {"mnist_linear": r"\code{mnist\_linear}", "mnist_cnn": r"\code{mnist\_cnn}",
             "cifar_deep": r"\code{cifar\_deep}", "resnet18_t10": r"\code{resnet18\_t10}"}

    def fmt_days(dys):
        if np.isnan(dys):
            return "---"
        if dys < 1:
            return f"{dys * 24:.1f}시간"
        return f"{dys:,.0f}일"

    for m in MODEL_ORDER:
        g = aggm[aggm.model == m]
        if g.empty:
            continue
        n_w, n_bits, clean = int(g.n_w.iloc[0]), int(g.n_bits.iloc[0]), float(g.clean.iloc[0])
        hits_day = R_SEU * n_bits
        row = {}
        for d in ["none", "clip"]:
            gd = g[g.defense == d]
            p50 = p_at_survival_half(gd) if len(gd) else float("nan")
            row[d] = (p50, p50 * n_w, p50 / (32 * R_SEU))
        pn, kn, dn = row["none"]
        pc, kc, dc = row["clip"]
        pc_s = "---" if np.isnan(pc) else f"{pc * 100:.2g}"
        lines.append(
            f"{short[m]} & {n_w:,} & {n_bits:,} & {clean:.1f} & {hits_day:,.1f} & "
            f"{pn * 100:.3g} & {kn:,.0f} & {fmt_days(dn)} & {pc_s} & {fmt_days(dc)} \\\\"
        )
        tag = MODEL_TAG[m]
        vals[f"Nw{tag}"] = f"{n_w:,}"
        vals[f"HitsDay{tag}"] = f"{hits_day:,.1f}"
        vals[f"Clean{tag}"] = f"{clean:.1f}"
        vals[f"PhalfNone{tag}"] = f"{pn * 100:.3g}"
        vals[f"KhalfNone{tag}"] = f"{kn:,.0f}"
        vals[f"DaysNone{tag}"] = fmt_days(dn)
        vals[f"PhalfClip{tag}"] = pc_s
        vals[f"DaysClip{tag}"] = fmt_days(dc)
        vals[f"ClipGain{tag}"] = "---" if np.isnan(pc) else f"{pc / pn:.0f}"
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TAB_DIR, "table_models.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return vals


# ---------------------------------------------------------------------------
# 수학 모델 (model_fit.py), E3 궤도 모드 검증, E5 방어 비교
# ---------------------------------------------------------------------------
import model_fit as MF  # noqa: E402

POS_FRAC = {"mnist_linear": 0.492, "mnist_cnn": 0.470, "cifar_deep": 0.421, "resnet18_t10": 0.459}  # 클린 체크포인트에서 잰 양수 가중치 비율
DEF_ORDER = ["none", "clip", "parity", "tmr", "ensemble_median", "ensemble"]
DEF_LABEL = {"none": "방어 없음", "clip": "클리핑", "parity": "패리티", "tmr": "TMR(가중치 중앙값)",
             "ensemble_median": "앙상블(로짓 중앙값)", "ensemble": "앙상블(로짓 평균)"}
# dataviz 팔레트 6슬롯 (blue, orange, aqua, yellow, magenta, green)
DEF_COLORS = {"none": "#C94A24", "clip": "#245EA6", "parity": "#1baf7a", "tmr": "#eda100",
              "ensemble_median": "#e87ba4", "ensemble": "#008300"}
DEF_SHORT = {"none": "없음", "clip": "클리핑", "parity": "패리티", "tmr": "TMR",
             "ensemble_median": "앙상블(중앙값)", "ensemble": "앙상블(평균)"}
DEF_MARKERS = {"none": "o", "clip": "s", "parity": "^", "tmr": "D", "ensemble_median": "v", "ensemble": "x"}


def load_leo() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(RESULTS, "single_*_infer_leo_saa_*.csv")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df.model.isin(MODEL_ORDER)].copy()
    df["survived"] = (df.acc >= 0.9 * df.clean_acc).astype(float)
    agg = (df.groupby(["model", "defense", "T_days"])
             .agg(n=("acc", "size"), median_acc=("acc", "median"), survival=("survived", "mean"),
                  crash=("crashed", "mean"), mean_k=("k", "mean"), n_bits=("n_bits", "first"), clean=("clean_acc", "first"))
             .reset_index())
    agg["model"] = pd.Categorical(agg.model, MODEL_ORDER, ordered=True)
    return agg.sort_values(["model", "defense", "T_days"])


def load_e5() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(RESULTS, "single_mnist_linear_infer_stress_*.csv")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[(df.single_hit == 0) & (df.allowed_bits.isna())].copy()
    df["survived"] = (df.acc >= 0.9 * df.clean_acc).astype(float)
    agg = (df.groupby(["defense", "p"])
             .agg(n=("acc", "size"), median_acc=("acc", "median"), survival=("survived", "mean"),
                  crash=("crashed", "mean"), mean_k=("k", "mean"), n_bits=("n_bits", "first"), clean=("clean_acc", "first"))
             .reset_index())
    agg["n_w"] = agg.n_bits // 32
    agg["model"] = "mnist_linear"
    return agg


def fig_model_fit(aggm: pd.DataFrame, fit: pd.DataFrame):
    """생존률 vs k 실측(점)과 모델 A(방어 없음, 지수)·모델 B(클리핑, 로그-로지스틱) 곡선."""
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3), sharey=True)
    kk = np.logspace(0, 5.3, 200)
    for j, d in enumerate(["none", "clip"]):
        ax = axes[j]
        for m in MODEL_ORDER:
            g = aggm[(aggm.model == m) & (aggm.defense == d)]
            if g.empty:
                continue
            c = MODEL_COLORS[m]
            ax.scatter(g.mean_k, g.survival, s=28, marker=MODEL_MARKERS[m], color=c, label=MODEL_LABEL[m], zorder=3)
            ax.plot(kk, MF.predict_survival(fit, m, d, kk), lw=1.6, color=c, alpha=0.8)
        ax.axhline(0.5, ls=":", lw=1, color="#555")
        ax.set_xscale("log")
        log_axis(ax)
        ax.set_xlabel("평균 명중 수 $\\bar{k}$")
        ax.set_title(f"({'ab'[j]}) {LABELS[d]}: 점 = 실측, 선 = 모델 {'AB'[j]}", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("생존률")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig_model_fit.{ext}"), dpi=200)
    plt.close(fig)


def fig_e3(leo: pd.DataFrame, fit: pd.DataFrame):
    """E3: leo_saa 모드 실측(점) vs stress 로 맞춘 모델의 예측(선). 가로축 = 노출 일수."""
    models = [m for m in MODEL_ORDER if m in set(leo.model.astype(str))]
    fig, axes = plt.subplots(1, len(models), figsize=(2.6 * len(models) + 1.2, 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        n_bits = int(leo[leo.model == m].n_bits.iloc[0])
        tt = np.logspace(np.log10(0.01), np.log10(200), 200)
        for d in ["none", "clip"]:
            g = leo[(leo.model == m) & (leo.defense == d)]
            if g.empty:
                continue
            c = COLORS[d]
            ax.plot(tt, MF.predict_survival(fit, m, d, R_SEU * n_bits * tt), lw=1.6, color=c, alpha=0.8,
                    label=f"{LABELS[d]} 예측")
            ax.scatter(g.T_days, g.survival, s=30, marker=MARKERS[d], color=c, zorder=3, label=f"{LABELS[d]} 실측")
        ax.axhline(0.5, ls=":", lw=1, color="#555")
        ax.set_xscale("log")
        log_axis(ax)
        ax.set_xlabel("LEO 노출 일수 $T$")
        ax.set_title(MODEL_LABEL[m].split(" ")[0], fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0.01, 200)
    axes[0].set_ylabel("생존률")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig_e3_leo.{ext}"), dpi=200)
    plt.close(fig)


def fig_e5(e5: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3))
    for d in DEF_ORDER:
        g = e5[e5.defense == d].sort_values("p")
        if g.empty:
            continue
        axes[0].plot(g.p * 100, g.median_acc, marker=DEF_MARKERS[d], ms=5, lw=2, color=DEF_COLORS[d], label=DEF_LABEL[d])
        axes[1].plot(g.p * 100, g.survival, marker=DEF_MARKERS[d], ms=5, lw=2, color=DEF_COLORS[d], label=DEF_LABEL[d])
    clean = float(e5.clean.iloc[0])
    axes[0].axhline(0.9 * clean, ls=":", lw=1, color="#555")
    axes[1].axhline(0.5, ls=":", lw=1, color="#555")
    for ax, t, yl, yl2 in zip(axes, ["(a) 중앙값 정확도", "(b) 생존률"], ["정확도 (%)", "생존률"], [(0, 100), (-0.05, 1.05)]):
        ax.set_xscale("log")
        log_axis(ax)
        ax.set_xlabel("가중치당 비트플립 확률 $p$ (%)")
        ax.set_ylabel(yl)
        ax.set_title(t, fontsize=10)
        ax.set_ylim(*yl2)
    axes[1].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig_e5_defenses.{ext}"), dpi=200)
    plt.close(fig)


def table_fit(fit: pd.DataFrame) -> dict:
    """표: 모델 A 파라미터 (q, α, k50) 와 모델 B 파라미터 (k50, β), 예측 T50."""
    vals = {}
    lines = [r"\begin{tabular}{llrrrr|rrr|r}", r"\toprule",
             r"모델 & 방어 & $\pi_+$ & $q$ & $\alpha$ & $k_{50}^{A}$ & $k_{50}^{B}$ & $\beta$ & MAE & $T_{50}$ 예측 \\",
             r"\midrule"]
    short = {"mnist_linear": r"\code{mnist\_linear}", "mnist_cnn": r"\code{mnist\_cnn}",
             "cifar_deep": r"\code{cifar\_deep}", "resnet18_t10": r"\code{resnet18\_t10}"}

    def fmt_days(dys):
        return f"{dys * 24:.1f}시간" if dys < 1 else f"{dys:,.0f}일"

    for m in MODEL_ORDER:
        for d in ["none", "clip"]:
            r = fit[(fit.model == m) & (fit.defense == d)]
            if r.empty:
                continue
            r = r.iloc[0]
            t50 = MF.t50_days(fit, m, d, R_SEU)
            tag = MODEL_TAG[m]
            if d == "none":
                lines.append(f"{short[m]} & 없음 & {r.pi_pos:.2f} & {r.q:.4f} & {r.alpha:.2f} & {r.k50_exp:,.0f} & "
                             f"{r.k50:,.0f} & {r.beta:.1f} & {r.mae_exp:.2f} & {fmt_days(t50)} \\\\")
                vals[f"Q{tag}"] = f"{r.q:.4f}"
                vals[f"Alpha{tag}"] = f"{r.alpha:.2f}"
                vals[f"KfiftyA{tag}"] = f"{r.k50_exp:,.0f}"
                vals[f"MaeA{tag}"] = f"{r.mae_exp:.2f}"
            else:
                lines.append(f" & 클리핑 & & & & & {r.k50:,.0f} & {r.beta:.1f} & {r.mae_ll:.2f} & {fmt_days(t50)} \\\\")
                vals[f"KfiftyB{tag}"] = f"{r.k50:,.0f}"
                vals[f"Beta{tag}"] = f"{r.beta:.1f}"
                vals[f"Gain{tag}"] = f"{r.gain:.0f}"
            vals[f"TfiftyPred{'None' if d == 'none' else 'Clip'}{tag}"] = fmt_days(t50)
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TAB_DIR, "table_fit.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    qs = fit[fit.defense == "none"].q
    vals["QMin"], vals["QMax"] = f"{qs.min():.4f}", f"{qs.max():.4f}"
    al = fit[fit.defense == "none"].alpha
    vals["AlphaMin"], vals["AlphaMax"] = f"{al.min():.2f}", f"{al.max():.2f}"
    return vals


def table_e3(leo: pd.DataFrame, fit: pd.DataFrame) -> dict:
    """표: E3 실측 생존률 vs 예측 (모델별 T)."""
    vals = {}
    lines = [r"\begin{tabular}{lr|rrr|rrr}", r"\toprule",
             r"모델 & $T$ (일) & \multicolumn{3}{c|}{방어 없음} & \multicolumn{3}{c}{클리핑} \\",
             r" & & $\bar{k}$ & 실측 & 예측 & $\bar{k}$ & 실측 & 예측 \\", r"\midrule"]
    short = {"mnist_linear": r"\code{mnist\_linear}", "mnist_cnn": r"\code{mnist\_cnn}",
             "cifar_deep": r"\code{cifar\_deep}", "resnet18_t10": r"\code{resnet18\_t10}"}
    errs = []
    for m in MODEL_ORDER:
        g = leo[leo.model == m]
        if g.empty:
            continue
        n_bits = int(g.n_bits.iloc[0])
        for T in sorted(g.T_days.unique()):
            cells = []
            for d in ["none", "clip"]:
                r = g[(g.defense == d) & (g.T_days == T)]
                if r.empty:
                    cells += ["---", "---", "---"]
                    continue
                r = r.iloc[0]
                pred = float(MF.predict_survival(fit, m, d, r.mean_k))  # 실제 전달된 선량(평균 명중 수)으로 예측
                errs.append(abs(pred - r.survival))
                cells += [f"{r.mean_k:,.0f}", f"{r.survival:.1f}", f"{pred:.2f}"]
            lines.append(f"{short[m]} & {T:g} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TAB_DIR, "table_e3.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    vals["EthreeMAE"] = f"{np.mean(errs):.2f}"
    vals["EthreeMaxErr"] = f"{np.max(errs):.2f}"
    vals["EthreeN"] = f"{len(errs)}"
    return vals


def table_e5(e5: pd.DataFrame, fit_e5: pd.DataFrame) -> dict:
    vals = {}
    ps = [p for p in sorted(e5[e5.defense == "tmr"].p.unique()) if p <= 0.03]  # E5 본 실험의 p 5단계만 열로 (고선량은 별도 표)
    lines = [r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{l" + "r" * len(ps) + r"|rr}", r"\toprule",
             r"방어 & " + " & ".join(f"$p$={fmt_p(p)}\\%" for p in ps) + r" & $k_{50}$ & 이득 \\",
             r"\midrule"]
    base = float(fit_e5[fit_e5.defense == "none"].k50.iloc[0])
    for d in DEF_ORDER:
        g = e5[e5.defense == d].set_index("p")
        if g.empty:
            continue
        cells = [f"{g.loc[p].median_acc:.1f} ({g.loc[p].survival:.1f})" if p in g.index else "---" for p in ps]
        fr = fit_e5[fit_e5.defense == d]
        if fr.empty:
            k50s, gain = "---", "---"
        else:
            k50 = float(fr.k50.iloc[0])
            kmax = float(e5[e5.defense == d].mean_k.max())
            if k50 > 3 * kmax:
                k50s, gain = f"$>${kmax:,.0f}", f"$>${kmax / base:.0f}"
            else:
                k50s, gain = f"{k50:,.0f}", f"{k50 / base:.1f}"
        lines.append(f"{DEF_SHORT[d]} & " + " & ".join(cells) + f" & {k50s} & {gain} \\\\")
        tag = {"none": "None", "clip": "Clip", "parity": "Parity", "tmr": "Tmr", "ensemble_median": "EnsMed", "ensemble": "Ens"}[d]
        vals[f"GainE{tag}"] = gain.replace("$", "")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TAB_DIR, "table_e5.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return vals


def clopper_pearson(succ: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    from scipy.stats import beta
    lo = 0.0 if succ == 0 else float(beta.ppf(alpha / 2, succ, n - succ + 1))
    hi = 1.0 if succ == n else float(beta.ppf(1 - alpha / 2, succ + 1, n - succ))
    return lo, hi


def load_e2() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(RESULTS, "single_*_infer_stress_*_1hit_bits*.csv")))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[(df.single_hit == 1) & (df.model.isin(MODEL_ORDER))].copy()
    df["bit"] = df.allowed_bits.astype(int)
    df["survived"] = (df.acc >= 0.9 * df.clean_acc).astype(float)
    agg = (df.groupby(["model", "defense", "bit"])
             .agg(n=("acc", "size"), succ=("survived", "sum"), survival=("survived", "mean"),
                  median_acc=("acc", "median"), min_acc=("acc", "min"), crash=("crashed", "mean"), clean=("clean_acc", "first"))
             .reset_index())
    ci = [clopper_pearson(int(s), int(n)) for s, n in zip(agg.succ, agg.n)]
    agg["ci_lo"] = [c[0] for c in ci]
    agg["ci_hi"] = [c[1] for c in ci]
    agg["model"] = pd.Categorical(agg.model, MODEL_ORDER, ordered=True)
    return agg.sort_values(["model", "defense", "bit"], ascending=[True, True, False])


def bit_kind(b: int) -> str:
    return "sign" if b == 31 else ("exponent" if 23 <= b <= 30 else "mantissa")


def fig_e2(e2: pd.DataFrame):
    """비트 지도: 모델별 단일 명중 생존률 (방어 없음), 95% 신뢰구간."""
    models = [m for m in MODEL_ORDER if m in set(e2.model.astype(str))]
    fig, axes = plt.subplots(1, len(models), figsize=(2.3 * len(models) + 1.0, 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    kind_col = {"sign": "#245EA6", "exponent": "#C94A24", "mantissa": "#8A97A8"}
    for ax, m in zip(axes, models):
        g = e2[(e2.model == m) & (e2.defense == "none")].sort_values("bit", ascending=False)
        x = np.arange(len(g))
        cols = [kind_col[bit_kind(int(b))] for b in g.bit]
        ax.bar(x, g.survival, color=cols, width=0.7)
        ax.errorbar(x, g.survival, yerr=[g.survival - g.ci_lo, g.ci_hi - g.survival], fmt="none", ecolor="#333", elinewidth=0.8, capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(b)) for b in g.bit], fontsize=7)
        ax.set_xlabel("뒤집은 비트")
        ax.set_title(MODEL_LABEL[m].split(" ")[0], fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, ls=":", lw=1, color="#555")
    axes[0].set_ylabel("단일 명중 생존률")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=kind_col[k], label=l) for k, l in [("sign", "부호(31)"), ("exponent", "지수(30–23)"), ("mantissa", "가수(22–0)")]],
               fontsize=8, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig_e2_bits.{ext}"), dpi=200)
    plt.close(fig)


def table_e2(e2: pd.DataFrame, fit: pd.DataFrame) -> dict:
    """표: 비트별 생존률 [95% CI] (방어 없음) + 클리핑(mnist_linear), 그리고 E2 로부터 독립 추정한 q."""
    vals = {}
    models = [m for m in MODEL_ORDER if m in set(e2.model.astype(str))]
    bits = sorted(e2.bit.unique(), reverse=True)
    short = {"mnist_linear": r"\code{mnist\_linear}", "mnist_cnn": r"\code{mnist\_cnn}",
             "cifar_deep": r"\code{cifar\_deep}", "resnet18_t10": r"\code{resnet18\_t10}"}
    has_clip = not e2[(e2.model == "mnist_linear") & (e2.defense == "clip")].empty
    cols = models + (["mnist_linear 클리핑"] if has_clip else [])
    lines = [r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
             "비트 & " + " & ".join([short[m] for m in models] + ([r"\code{mnist\_linear} 클리핑"] if has_clip else [])) + r" \\", r"\midrule"]
    for b in bits:
        cells = []
        for m in models:
            r = e2[(e2.model == m) & (e2.defense == "none") & (e2.bit == b)]
            cells.append("---" if r.empty else f"{r.survival.iloc[0]:.2f} [{r.ci_lo.iloc[0]:.2f}, {r.ci_hi.iloc[0]:.2f}]")
        if has_clip:
            r = e2[(e2.model == "mnist_linear") & (e2.defense == "clip") & (e2.bit == b)]
            cells.append("---" if r.empty else f"{r.survival.iloc[0]:.2f}")
        lab = {31: "31 (부호)", 30: "30 (지수 MSB)"}.get(int(b), f"{int(b)} ({'지수' if 23 <= b <= 30 else '가수'})")
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    lines += [r"\midrule"]
    # E2 기반 q 추정: q_E2 = (1/32) Σ_b (1 - S_b), 측정하지 않은 가수 비트는 측정한 가수 비트 평균으로
    qrow, qfit_row, qpred_row = [], [], []
    for m in models:
        g = e2[(e2.model == m) & (e2.defense == "none")].set_index("bit")
        meas = {int(b): float(g.loc[b].survival) for b in g.index}
        mant = [1 - meas[b] for b in meas if b <= 22]
        mant_mean = float(np.mean(mant)) if mant else 0.0
        tot = 0.0
        for b in range(32):
            tot += (1 - meas[b]) if b in meas else (mant_mean if b <= 22 else 0.0)
        q_e2 = tot / 32
        fr = fit[(fit.model == m) & (fit.defense == "none")]
        q_fit = float(fr.q.iloc[0]) if not fr.empty else np.nan
        s30 = meas.get(30, np.nan)
        qrow.append(f"{q_e2:.4f}")
        qfit_row.append(f"{q_fit:.4f}")
        qpred_row.append(f"{1 - 32 * q_fit:.2f} / {s30:.2f}")
        tag = MODEL_TAG[m]
        vals[f"QEtwo{tag}"] = f"{q_e2:.4f}"
        vals[f"SthirtyEtwo{tag}"] = f"{s30:.2f}"
        vals[f"SthirtyPred{tag}"] = f"{1 - 32 * q_fit:.2f}"
        if 30 in g.index:
            r30 = g.loc[30]
            vals[f"SthirtyCI{tag}"] = f"[{float(r30.ci_lo):.2f}, {float(r30.ci_hi):.2f}]"
        else:
            vals[f"SthirtyCI{tag}"] = "[---]"
    pad = [""] if has_clip else []
    lines.append(r"$q_{E2}=\frac{1}{32}\sum_b(1-S_b)$ & " + " & ".join(qrow + pad) + r" \\")
    lines.append(r"$q$ (E1 적합) & " + " & ".join(qfit_row + pad) + r" \\")
    lines.append(r"bit 30 예측 $1-32q$ / 실측 & " + " & ".join(qpred_row + pad) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TAB_DIR, "table_e2.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return vals


def table_e5_high(e5: pd.DataFrame, fit_e5: pd.DataFrame) -> dict:
    """고선량 TMR·패리티: 측정 vs TMR 해석적 예측 S = exp(-3 k^2 q / (32 N_w))."""
    vals = {}
    q = float(fit_e5[fit_e5.defense == "none"].q.iloc[0])
    n_w = int(e5.n_w.iloc[0])
    k50_tmr = float(np.sqrt(32 * n_w * np.log(2) / (3 * q)))
    vals["KfiftyTmrPred"] = f"{k50_tmr:,.0f}"
    vals["PfiftyTmrPred"] = f"{100 * k50_tmr / n_w:.0f}"
    ps = sorted(e5[e5.defense == "tmr"].p.unique())
    lines = [r"\begin{tabular}{lrr|rr|rr}", r"\toprule",
             r"$p$ (\%) & $\bar{k}$ & $k/N_w$ & TMR 실측 & TMR 예측 & 패리티 실측 & 패리티 중앙값 \\", r"\midrule"]
    for p in ps:
        t = e5[(e5.defense == "tmr") & (e5.p == p)].iloc[0]
        pa = e5[(e5.defense == "parity") & (e5.p == p)]
        pred = float(np.exp(-3 * t.mean_k ** 2 * q / (32 * n_w)))
        pa_s = "---" if pa.empty else f"{pa.survival.iloc[0]:.1f}"
        pa_m = "---" if pa.empty else f"{pa.median_acc.iloc[0]:.1f}"
        lines.append(f"{fmt_p(p)} & {t.mean_k:,.0f} & {t.mean_k / n_w:.2f} & {t.survival:.1f} & {pred:.2f} & {pa_s} & {pa_m} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TAB_DIR, "table_e5_high.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return vals


def worked_example(fit: pd.DataFrame) -> dict:
    """본문 예시: 보편 q 로 임의 모델의 T50 예측 (ResNet-50 2,560만, 1억 파라미터)."""
    q_med = float(fit[fit.defense == "none"].q.median())
    vals = {"QMedian": f"{q_med:.4f}"}
    for name, n_w in [("ResnetFifty", 25_600_000), ("HundredM", 100_000_000)]:
        t50 = np.log(2) / (q_med * R_SEU * 32 * n_w)
        vals[f"Tfifty{name}"] = f"{t50 * 24:.1f}시간" if t50 < 1 else f"{t50:.1f}일"
        vals[f"Tfifty{name}Min"] = f"{t50 * 24 * 60:.0f}분"
    return vals


if __name__ == "__main__":
    df = load()
    agg = summarize(df)
    pd.set_option("display.width", 200)
    print(agg.round(3).to_string(index=False))
    fig_dose_response(agg)
    fig_survival_crash(agg)
    fig_trials(df, agg)
    table_e1(agg)
    vals = numbers(df, agg)
    # E1-M
    dfm = load_models()
    aggm = summarize_models(dfm)
    print(aggm.round(3).to_string(index=False))
    fig_models(aggm)
    fig_models_hits(aggm)
    vals_m = table_models(aggm)
    # 수학 모델 피팅 (stress 데이터만) + E3 검증 + E5
    fit = MF.fit_all(aggm, POS_FRAC)
    print(fit.round(4).to_string(index=False))
    fig_model_fit(aggm, fit)
    vals_m.update(table_fit(fit))
    lrt = MF.lrt_common_q(aggm)
    print("LRT common q:", lrt)
    vals_m.update({"QCommon": f"{lrt['q_common']:.4f}", "LrtD": f"{lrt['D']:.1f}", "LrtDf": f"{lrt['df']}",
                   "LrtP": (f"{lrt['p_value']:.3f}" if lrt['p_value'] >= 0.001 else "<0.001")})
    vals_m.update(worked_example(fit))
    leo = load_leo()
    print(leo.round(3).to_string(index=False))
    fig_e3(leo, fit)
    vals_m.update(table_e3(leo, fit))
    e5 = load_e5()
    fit_e5 = MF.fit_all(e5.assign(model="mnist_linear"), POS_FRAC)
    print(fit_e5.round(3).to_string(index=False))
    fig_e5(e5[e5.p <= 0.03])
    vals_m.update(table_e5(e5, fit_e5))
    vals_m.update(table_e5_high(e5, fit_e5))
    e2 = load_e2()
    if not e2.empty:
        print(e2.round(3).to_string(index=False))
        fig_e2(e2)
        vals_m.update(table_e2(e2, fit))
    with open(os.path.join(TAB_DIR, "numbers.tex"), "a", encoding="utf-8") as f:
        f.write("% --- E1-M 모델 크기 축 / 수학 모델 / E3 / E5 ---\n")
        for k, v in vals_m.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    vals.update(vals_m)
    for k, v in vals.items():
        print(f"{k:>18} = {v}")
    print("wrote ->", FIG_DIR, TAB_DIR)
