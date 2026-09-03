#!/usr/bin/env python3
"""Estimator, mitigation, and harder-real-data robustness experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import coopt
from coopt import ContextModel, Workload, evaluate, utility_from_components
from search import METHODS_MAIN, joint_bo, run_method


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
WINDOWS = [0, 1, 2]
TEST_WINDOWS = list(range(9))


def sha256(path: Path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def deployment(wl, cfg, cm, seed, draws=20):
    comps = []
    for draw in range(draws):
        rng_ctx = np.random.default_rng(301_000_000 + seed * 1000 + draw)
        context = cm.sample(rng_ctx, TEST_WINDOWS[draw % len(TEST_WINDOWS)])
        comps.append(evaluate(wl, cfg, context, 3,
                              np.random.default_rng(302_000_000 + seed * 1000 + draw)))
    utility = np.array([utility_from_components(comp) for comp in comps])
    return {
        "utility": float(utility.mean()), "utility_p10": float(np.percentile(utility, 10)),
        "service_violation": float(np.mean([max(c["deadline_miss"], c["infeasible"], c["realized_fail"])
                                                   for c in comps])),
        "quality_loss": float(np.mean([c["quality_loss"] for c in comps])),
        "delay_s": float(np.mean([c["delay_s"] for c in comps])),
    }


def winsor_experiment():
    rows = []
    labels = [(-2.0, "-2"), (-4.0, "-4"), (-6.0, "-6"), (-10.0, "-10"), (None, "none")]
    workloads = ["qaoa", "qaoa_channel", "qaoa_place", "vqc", "vqc_cancer"]
    for workload in workloads:
        for seed in range(210, 216):
            wl = Workload.make(workload, seed); cm = ContextModel(seed=500 + seed)
            for floor, label in labels:
                rng = np.random.default_rng(303_000_000 + seed)
                cfg, ledger, _ = joint_bo(wl, cm, WINDOWS, 20.0, rng,
                                          multi_fidelity=True, winsor_floor=floor)
                rows.append({"workload": workload, "seed": seed, "winsor_floor": label,
                             "search_cost_units": ledger.spent,
                             **deployment(wl, cfg, cm, seed)})
                print(f"winsor {workload} {seed} {label}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(RESULTS / "fable_winsor_sensitivity.csv", index=False)
    return result


def config_from_row(row):
    if row.workload.startswith("qaoa"):
        keys = ("p", "opt_iters", "warm_start", "shots", "mitigation", "compile_level")
    else:
        keys = ("n_qubits", "layers", "reupload", "train_iters", "shots", "mitigation", "compile_level")
    cfg = {key: int(getattr(row, f"cfg_{key}")) for key in keys}
    if not row.workload.startswith("qaoa"):
        cfg["lr"] = float(row.cfg_lr)
    cfg["placement"] = row.cfg_placement
    return cfg


def mitigation_experiment():
    source = pd.read_csv(RESULTS / "clearaccept_confirmatory.csv")
    source = source[source.method.isin(["joint_mf", "capo_chance", "sequential"])
                    & source.seed.isin([100, 101])].copy()
    rows = []
    original_shots = coopt.MITIGATION_SHOT_MULTIPLIER
    original_error = coopt.MITIGATION_ERROR_MULTIPLIER
    try:
        for shot_factor in (2.0, 3.0, 5.0):
            for error_factor in (0.25, 0.50, 0.75):
                coopt.MITIGATION_SHOT_MULTIPLIER = shot_factor
                coopt.MITIGATION_ERROR_MULTIPLIER = error_factor
                for row in source.itertuples():
                    wl = Workload.make(row.workload, int(row.seed)); cm = ContextModel(seed=500 + int(row.seed))
                    cfg = config_from_row(row)
                    metrics = deployment(wl, cfg, cm, int(row.seed), draws=10)
                    rows.append({"workload": row.workload, "seed": int(row.seed),
                                 "method": row.method, "selected_mitigation": cfg["mitigation"],
                                 "shot_multiplier": shot_factor,
                                 "residual_error_multiplier": error_factor, **metrics})
                print(f"mitigation shots={shot_factor} residual={error_factor}", flush=True)
    finally:
        coopt.MITIGATION_SHOT_MULTIPLIER = original_shots
        coopt.MITIGATION_ERROR_MULTIPLIER = original_error
    result = pd.DataFrame(rows)
    result.to_csv(RESULTS / "fable_mitigation_sensitivity.csv", index=False)
    return result


def digits_experiment():
    methods = ["joint_mf", "capo_chance", "sequential",
               "device_aware_sequential", "fixed_design_scheduler", "classical_only"]
    rows = []
    for seed in range(300, 308):
        wl = Workload.make("vqc_digits", seed); cm = ContextModel(seed=500 + seed)
        for method in methods:
            rng = np.random.default_rng(304_000_000 + seed * 100 + METHODS_MAIN.index(method))
            cfg, ledger = run_method(method, wl, cm, WINDOWS, 30.0, rng)
            rows.append({"workload": "vqc_digits", "seed": seed, "method": method,
                         "search_cost_units": ledger.spent,
                         "search_cost_modeled_s": ledger.modeled_cost_s,
                         "cfg_placement": cfg["placement"], **deployment(wl, cfg, cm, seed)})
            print(f"digits {seed} {method}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(RESULTS / "fable_digits.csv", index=False)
    return result


def main():
    winsor = winsor_experiment()
    mitigation = mitigation_experiment()
    digits = digits_experiment()
    manifest = {
        "script_sha256": sha256(Path(__file__)),
        "winsor": {"rows": len(winsor), "fresh_seeds": [210, 215], "budget": 20,
                    "deployment_draws": 20},
        "mitigation": {"rows": len(mitigation), "selected_config_seeds": [100, 101],
                       "deployment_draws": 10,
                       "interpretation": "fixed-selection deployment sensitivity, not re-optimization"},
        "digits": {"rows": len(digits), "fresh_seeds": [300, 307], "budget": 30,
                   "deployment_draws": 20, "task": "sklearn digits 3-versus-8"},
        "evidence": "exact statevector plus explicitly modeled noise; no QPU execution",
    }
    (RESULTS / "fable_robustness_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
