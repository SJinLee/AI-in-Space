#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
수학적 예측 모델 (make_figures.py 에서 호출).

모델 A — 치명 명중 모형 (방어 없음, 기계론적):
    명중 하나가 '치명'일 확률 q.  k번 명중 후 생존률 S = (1-q)^k ≈ exp(-q k).
    q = (1/32) · π₊ · α      (지수 최상위 비트 선택 1/32, 양수 가중치 비율 π₊, 유효 전파 비율 α)
    LEO:  k = r N_bits T  →  S(T) = exp(-q r N_bits T),   T50 = ln2 / (q r N_bits)

모델 B — 누적 손상 모형 (모든 방어, 경험적):
    S(k) = 1 / (1 + (k / k50)^β)     (로그-로지스틱)
    방어 이득 G_d = k50_d / k50_none  (같은 생존률을 주는 선량의 배수)

둘 다 각 (모델, 방어)의 stress 데이터(생존 횟수 / 시행 수)에 이항 최대가능도로 맞춘다.
E3(leo_saa) 데이터에는 맞추지 않고, stress 로 맞춘 모델의 예측과 비교한다(검증).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, minimize

EPS = 1e-9


def _binom_nll(s_pred: np.ndarray, surv: np.ndarray, n: np.ndarray) -> float:
    s_pred = np.clip(s_pred, EPS, 1 - EPS)
    succ = surv * n
    return float(-np.sum(succ * np.log(s_pred) + (n - succ) * np.log(1 - s_pred)))


def fit_exponential(k: np.ndarray, surv: np.ndarray, n: np.ndarray) -> float:
    """모델 A: S = exp(-q k).  q 의 MLE."""
    f = lambda lq: _binom_nll(np.exp(-np.exp(lq) * k), surv, n)
    res = minimize_scalar(f, bounds=(np.log(1e-6), np.log(1.0)), method="bounded")
    return float(np.exp(res.x))


def fit_loglogistic(k: np.ndarray, surv: np.ndarray, n: np.ndarray) -> tuple[float, float]:
    """모델 B: S = 1/(1+(k/k50)^β).  (k50, β) 의 MLE."""
    def f(x):
        lk50, lbeta = x
        s = 1.0 / (1.0 + (k / np.exp(lk50)) ** np.exp(lbeta))
        return _binom_nll(s, surv, n)
    best = None
    for lk0 in np.log([10, 100, 1000, 10000]):
        for lb0 in np.log([0.7, 1.5, 3.0]):
            r = minimize(f, x0=[lk0, lb0], method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000})
            if best is None or r.fun < best.fun:
                best = r
    return float(np.exp(best.x[0])), float(np.exp(best.x[1]))


def s_exp(k, q):
    return np.exp(-q * np.asarray(k, dtype=float))


def s_ll(k, k50, beta):
    return 1.0 / (1.0 + (np.asarray(k, dtype=float) / k50) ** beta)


def fit_all(aggm: pd.DataFrame, pos_frac: dict) -> pd.DataFrame:
    """aggm: (model, defense, p) 별 survival, n, mean_k, n_bits.  stress 데이터만 넣을 것."""
    rows = []
    for (m, d), g in aggm.groupby(["model", "defense"], observed=True):
        g = g[g.mean_k > 0]
        k, s, n = g.mean_k.values, g.survival.values, g.n.values
        k50, beta = fit_loglogistic(k, s, n)
        row = dict(model=m, defense=d, n_points=len(g), n_w=int(g.n_w.iloc[0]), n_bits=int(g.n_bits.iloc[0]),
                   k50=k50, beta=beta)
        if d == "none":
            q = fit_exponential(k, s, n)
            row["q"] = q
            row["k50_exp"] = np.log(2) / q
            pi = pos_frac.get(str(m), np.nan)
            row["pi_pos"] = pi
            row["alpha"] = q / (pi / 32.0) if pi == pi else np.nan
            # 적합도: 각 점에서 |S_pred - S_obs| 평균
            row["mae_exp"] = float(np.mean(np.abs(s_exp(k, q) - s)))
        row["mae_ll"] = float(np.mean(np.abs(s_ll(k, k50, beta) - s)))
        rows.append(row)
    fit = pd.DataFrame(rows)
    # 방어 이득
    base = fit[fit.defense == "none"].set_index("model").k50
    fit["gain"] = [r.k50 / base.get(r.model, np.nan) for r in fit.itertuples()]
    return fit


def lrt_common_q(aggm: pd.DataFrame) -> dict:
    """H4 검정: 모델마다 다른 q (대립) vs 모든 모델에 공통 q (귀무). 방어 없음 stress 데이터.
    가능도비 통계량 D = 2(ℓ_sep - ℓ_common) ~ χ²(df = 모델 수 - 1)."""
    from scipy.stats import chi2
    g = aggm[(aggm.defense == "none") & (aggm.mean_k > 0)]
    models = list(g.model.astype(str).unique())
    ll_sep = 0.0
    for m in models:
        gm = g[g.model.astype(str) == m]
        q = fit_exponential(gm.mean_k.values, gm.survival.values, gm.n.values)
        ll_sep -= _binom_nll(s_exp(gm.mean_k.values, q), gm.survival.values, gm.n.values)
    q_c = fit_exponential(g.mean_k.values, g.survival.values, g.n.values)
    ll_com = -_binom_nll(s_exp(g.mean_k.values, q_c), g.survival.values, g.n.values)
    D = 2 * (ll_sep - ll_com)
    df = len(models) - 1
    return dict(q_common=q_c, D=float(D), df=df, p_value=float(chi2.sf(D, df)), n_models=len(models))


def predict_survival(fit: pd.DataFrame, model: str, defense: str, k) -> np.ndarray:
    r = fit[(fit.model == model) & (fit.defense == defense)].iloc[0]
    if defense == "none":
        return s_exp(k, r.q)
    return s_ll(k, r.k50, r.beta)


def t50_days(fit: pd.DataFrame, model: str, defense: str, r_seu: float) -> float:
    row = fit[(fit.model == model) & (fit.defense == defense)].iloc[0]
    k50 = row.k50_exp if defense == "none" else row.k50
    return float(k50 / (r_seu * row.n_bits))
