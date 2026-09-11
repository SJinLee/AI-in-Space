#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Named experiment matrices.

paper_sweep (CPU-feasible default)
----------------------------------
Must complete, shrinking epochs/subset rather than dropping a model:

- toy50:        infer+train; leo_saa, poisson, stress(p=0.01);
                none, clip, tmr, ensemble, parity; trials=15
- mnist_linear: infer+train; leo_saa, poisson, stress 0.01;
                none, clip, tmr; trials=8
- mnist_cnn:    infer+train; leo_saa, stress 0.01; none, clip; trials=5
- resnet18_t10: infer; leo_saa, stress 0.01; none, clip; trials=3
                + train transfer 1 epoch, leo_saa+stress, none+clip, trials=2
                IF a run is <3 min; else infer only
- cifar_deep:   infer; leo_saa, stress 0.01; none, clip; trials=3
                + train 1 epoch none+clip × leo_saa if time

Always leave at least infer results for all 5 models.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterator


@dataclass
class Cell:
    model: str
    mode: str          # infer | train
    rad: str           # poisson | leo_saa | stress
    defense: str
    trials: int
    p: float = 0.01
    T_days: float = 1.0
    train_hours: float = 24.0
    epochs: int | None = None          # None → model default / pilot
    n_train: int | None = None
    n_test: int | None = None
    optional: bool = False             # may be dropped after pilot
    note: str = ""
    allowed_bits: str | None = None    # e.g. "30", "23-30", "exponent"; None = all 32 bits
    single_hit: bool = False           # exactly one event (bit-criticality experiment)


def _grid(model, mode, rads, defenses, trials, **kw) -> list[Cell]:
    cells = []
    for rad in rads:
        for defense in defenses:
            cells.append(Cell(model=model, mode=mode, rad=rad, defense=defense, trials=trials, **kw))
    return cells


def paper_sweep_cells() -> list[Cell]:
    cells: list[Cell] = []
    # --- toy50 (required, cheap) ---
    for mode in ("infer", "train"):
        cells += _grid(
            "toy50", mode,
            rads=("leo_saa", "poisson", "stress"),
            defenses=("none", "clip", "tmr", "ensemble", "parity"),
            trials=15,
        )
    # --- mnist_linear ---
    for mode in ("infer", "train"):
        cells += _grid(
            "mnist_linear", mode,
            rads=("leo_saa", "poisson", "stress"),
            defenses=("none", "clip", "tmr"),
            trials=8,
        )
    # --- mnist_cnn ---
    for mode in ("infer", "train"):
        cells += _grid(
            "mnist_cnn", mode,
            rads=("leo_saa", "stress"),
            defenses=("none", "clip"),
            trials=5,
        )
    # --- resnet18_t10 infer (required) ---
    cells += _grid(
        "resnet18_t10", "infer",
        rads=("leo_saa", "stress"),
        defenses=("none", "clip"),
        trials=3,
        n_train=10_000,
        n_test=2_000,
    )
    # resnet train: optional, keep if a run is < 3 min
    cells += _grid(
        "resnet18_t10", "train",
        rads=("leo_saa", "stress"),
        defenses=("none", "clip"),
        trials=2,
        n_train=10_000,
        n_test=2_000,
        epochs=1,
        optional=True,
        note="train only if one run < 180s",
    )
    # --- cifar_deep infer (required) ---
    cells += _grid(
        "cifar_deep", "infer",
        rads=("leo_saa", "stress"),
        defenses=("none", "clip"),
        trials=3,
        n_train=10_000,
        n_test=2_000,
    )
    # cifar_deep train: 1 epoch none+clip × leo_saa if time
    cells += _grid(
        "cifar_deep", "train",
        rads=("leo_saa",),
        defenses=("none", "clip"),
        trials=2,
        n_train=10_000,
        n_test=2_000,
        epochs=1,
        optional=True,
        note="train 1 epoch if a run is not too slow (< 180s)",
    )
    return cells


def apply_pilot_decisions(
    cells: list[Cell],
    *,
    train_seconds: dict[str, float],
    infer_seconds: dict[str, float],
    epoch_caps: dict[str, int],
    n_train_caps: dict[str, int],
) -> list[Cell]:
    """Drop optional train cells that are too slow; shrink epochs/n_train."""
    out: list[Cell] = []
    for c in cells:
        c = replace(c)
        if c.model in epoch_caps and c.mode == "train":
            cap = epoch_caps[c.model]
            c.epochs = cap if c.epochs is None else min(int(c.epochs), cap)
        if c.model in n_train_caps:
            cap = n_train_caps[c.model]
            c.n_train = cap if c.n_train is None else min(int(c.n_train), cap)
        if c.optional and c.mode == "train":
            t = float(train_seconds.get(c.model, 9999.0))
            if t >= 180.0:
                print(f"[preset] drop optional train cell {c.model}/{c.rad}/{c.defense} "
                      f"(pilot {t:.1f}s >= 180s) — infer-only for this model")
                continue
        out.append(c)
    return out


def cell_key(c: Cell) -> str:
    return f"{c.model}|{c.mode}|{c.rad}|{c.defense}|p={c.p}|T={c.T_days}|bits={c.allowed_bits}|1hit={int(c.single_hit)}"
