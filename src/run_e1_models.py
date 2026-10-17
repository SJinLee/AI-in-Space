#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E1 (용량-반응) 을 여러 모델에 반복: 모델 크기 축.

    python src/run_e1_models.py --models mnist_cnn cifar_deep --trials 10
    python src/run_e1_models.py --models resnet18_t10 --trials 10 --p-list 1e-5 3e-5 1e-4 3e-4 1e-3

노트북 3-1 확장 셀과 같은 방식으로 simulate.run_cell() 을 직접 불러 데이터·체크포인트를
한 번만 읽고, 결과를 simulate.py 와 같은 이름(results/single_<model>_infer_stress_<defense>_p<p>.csv)
으로 저장한다. 그래서 paper/make_figures.py 가 그대로 읽는다.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import pandas as pd

import simulate as SIM
from presets import Cell

DEFAULT_P = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["mnist_cnn", "cifar_deep"])
    ap.add_argument("--defenses", nargs="+", default=["none", "clip"])
    ap.add_argument("--rad", default="stress", choices=["stress", "leo_saa", "poisson"],
                    help="stress: --p-list 를 돈다 / leo_saa·poisson: --T-list(일)를 돈다 (E3)")
    ap.add_argument("--p-list", nargs="+", type=float, default=DEFAULT_P)
    ap.add_argument("--T-list", nargs="+", type=float, default=[1, 7, 30, 90])
    ap.add_argument("--r", type=float, default=SIM.DEFAULT_R)
    ap.add_argument("--bits", nargs="+", type=int, default=None,
                    help="E2: 지정한 비트 하나만, 정확히 한 번 뒤집기 (--rad stress 와 함께; p 무시)")
    ap.add_argument("--mode", default="infer", choices=["infer", "train"],
                    help="E4: train 이면 학습 도중에 주입 (stress: 총 p·N_w 번을 전체 스텝에 분산; leo_saa: --train-hours 를 궤도 시간으로 가정)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--train-hours", nargs="+", type=float, default=[24.0],
                    help="E4 leo_saa: 학습 전체를 이 시간(h)의 궤도 노출로 가정; 여러 값이면 각각 돈다")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--skip-existing", action="store_true", help="이미 있는 CSV는 건너뜀")
    args = ap.parse_args()

    SIM.configure_torch()
    os.makedirs(args.out_dir, exist_ok=True)
    data_cache, ckpt_cache = {}, {}
    t0 = time.time()
    train = args.mode == "train"
    if args.bits is not None:
        doses = args.bits
    elif train and args.rad != "stress":
        doses = args.train_hours
    else:
        doses = args.p_list if args.rad == "stress" else args.T_list
    tag_ep = f"_e{args.epochs}" if (train and args.epochs is not None) else ""
    for model in args.models:
        for dose in doses:
            for d in args.defenses:
                bits, single, hours = None, False, 24.0
                if args.bits is not None:
                    p, T, bits, single = SIM.DEFAULT_STRESS_P, 1.0, str(int(dose)), True
                    path = os.path.join(args.out_dir, f"single_{model}_infer_stress_{d}_1hit_bits{bits}.csv")
                elif args.rad == "stress":
                    p, T = dose, 1.0
                    path = os.path.join(args.out_dir, f"single_{model}_{args.mode}_stress_{d}_p{p:g}{tag_ep}.csv")
                elif train:
                    p, T, hours = SIM.DEFAULT_STRESS_P, dose / 24.0, dose
                    path = os.path.join(args.out_dir, f"single_{model}_train_{args.rad}_{d}_h{hours:g}_r{args.r:g}{tag_ep}.csv")
                else:
                    p, T = SIM.DEFAULT_STRESS_P, dose
                    path = os.path.join(args.out_dir, f"single_{model}_infer_{args.rad}_{d}_T{T:g}_r{args.r:g}.csv")
                if args.skip_existing and os.path.exists(path):
                    print(f"skip {path}")
                    continue
                cell = Cell(model=model, mode=args.mode, rad=args.rad, defense=d, trials=args.trials, p=p, T_days=T,
                            train_hours=hours, epochs=args.epochs, n_train=args.n_train,
                            allowed_bits=bits, single_hit=single)
                t1 = time.time()
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rows = SIM.run_cell(
                        cell, r=args.r, flip_bias=False, ckpt_dir=SIM.CKPT_DIR_DEFAULT,
                        master_seed=SIM.MASTER_SEED, data_cache=data_cache, ckpt_cache=ckpt_cache,
                        budget_overrides={},
                    )
                log = buf.getvalue()
                if "[model]" in log and p == args.p_list[0] and d == args.defenses[0]:
                    print([l for l in log.splitlines() if l.startswith("[model]")][0])
                df = pd.DataFrame(rows)
                df.to_csv(path, index=False)
                s = SIM.make_summary(df).iloc[0]
                dose_s = f"bit={bits:<7}" if single else (f"p={p:<8g}" if args.rad == "stress" else (f"h={hours:<7g}" if train else f"T={T:<6g}d"))
                clean_s = f"clean={df.clean_acc.iloc[0]:6.2f}" if "clean_acc" in df and df.clean_acc.notna().any() else f"last_fin={df.last_finite_acc.median():6.2f}"
                print(f"{model:<13} {args.mode[:5]} {dose_s} {d:<15} n={args.trials}  {clean_s}  "
                      f"median={s['median_acc']:6.2f}  survival={s['survival_rate']:.2f}  "
                      f"crash={s['crash_rate']:.2f}  k={s['mean_k']:7.1f}  "
                      f"[{time.time()-t1:5.1f}s / {time.time()-t0:6.1f}s]", flush=True)
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
