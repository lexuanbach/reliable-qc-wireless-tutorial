#!/usr/bin/env python3
"""Run the co-optimization experiments (plan section 9, Experiments 1-6).

Experiment 6 is renamed 'fidelity consistency': with no hardware access the
deployment fidelity (fid3) is a held-out, perturbed noisy simulation and is
labeled as such everywhere.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from coopt import (
    ContextModel,
    Workload,
    evaluate,
    mean_utility,
    sample_config,
    space_for,
    utility_from_components,
)
from search import METHODS_MAIN, fixed_policy, joint_bo, oracle_placement, run_method

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"

TRAIN_WINDOWS = [0]
MULTI_WINDOWS = [0, 1, 2]
TEST_WINDOWS = list(range(9))


def final_eval(wl, cfg, cm, windows, seed, draws=20, drop_terms=(),
               fidelity=3):
    """Paired deployment evaluation at fid3 with a fixed evaluation seed."""
    utils, comps = [], []
    for i in range(draws):
        w = windows[i % len(windows)]
        ctx = cm.sample(np.random.default_rng(777_000 + seed * 1000 + i), w)
        eval_rng = np.random.default_rng(877_000 + seed * 1000 + i)
        comp = evaluate(wl, cfg, ctx, fidelity, eval_rng)
        utils.append(utility_from_components(comp, drop_terms))
        comps.append(comp)
    keys = ("quality_loss", "delay_s", "deadline_miss", "infeasible", "energy_j",
            "money", "cost_scalar", "realized_fail")
    agg = {k: float(np.mean([c[k] for c in comps])) for k in keys}
    agg["utility"] = float(np.mean(utils))
    agg["utility_p10"] = float(np.percentile(utils, 10))
    k = max(1, int(np.ceil(0.10 * len(utils))))
    agg["utility_cvar10"] = float(np.sort(utils)[:k].mean())
    agg["utility_min"] = float(np.min(utils))
    agg["service_violation"] = float(np.mean([
        max(c["deadline_miss"], c["infeasible"], c["realized_fail"])
        for c in comps]))
    return agg


def exp1_search_efficiency(workloads, seeds, budget, rows, offset=0,
                           methods=None, oracle=True):
    for kind in workloads:
        for seed in range(offset, offset + seeds):
            wl = Workload.make(kind, seed)
            cm = ContextModel(seed=500 + seed)
            for method in (methods or METHODS_MAIN):
                rng = np.random.default_rng(1_000_000 + seed * 100
                                            + METHODS_MAIN.index(method))
                t0 = time.perf_counter()
                cfg, ledger = run_method(method, wl, cm, MULTI_WINDOWS, budget, rng)
                agg = final_eval(wl, cfg, cm, TEST_WINDOWS, seed)
                rows.append({
                    "experiment": "exp1", "workload": kind, "seed": seed,
                    "method": method, "budget": budget,
                    "search_cost_units": ledger.spent,
                    "search_cost_modeled_s": ledger.modeled_cost_s,
                    "search_evaluations": ledger.evaluations,
                    "unique_candidates": len(ledger.unique_configs),
                    "fidelity_counts": json.dumps(ledger.fidelity_counts, sort_keys=True),
                    "wall_s": time.perf_counter() - t0,
                    **{f"cfg_{k}": v for k, v in cfg.items()}, **agg,
                })
                print(f"exp1 {kind} seed={seed} {method}: u={agg['utility']:.3f}",
                      flush=True)
            # oracle placement on top of the joint pick (upper bound only)
            if not oracle:
                continue
            joint_cfg = next(r for r in rows if r["experiment"] == "exp1"
                             and r["workload"] == kind and r["seed"] == seed
                             and r["method"] == "joint_bo")
            base = {k[4:]: v for k, v in joint_cfg.items() if k.startswith("cfg_")}
            ocfg = oracle_placement(wl, base, cm, TEST_WINDOWS,
                                    np.random.default_rng(2_000_000 + seed))
            agg = final_eval(wl, ocfg, cm, TEST_WINDOWS, seed)
            rows.append({"experiment": "exp1", "workload": kind, "seed": seed,
                         "method": "oracle_placement", "budget": budget,
                         "search_cost_units": 0.0, "search_cost_modeled_s": 0.0,
                         "wall_s": 0.0,
                         **{f"cfg_{k}": v for k, v in ocfg.items()}, **agg})


ABLATION_DROPS = {
    "full": (),
    "no_latency": ("latency",),
    "no_energy": ("energy",),
    "no_money": ("money",),
    "no_failure": ("failure",),
    "no_staleness": ("staleness",),
}


def exp2_objective_ablation(workloads, seeds, budget, rows):
    for kind in workloads:
        for seed in range(seeds):
            wl = Workload.make(kind, seed)
            cm = ContextModel(seed=500 + seed)
            for tag, drops in ABLATION_DROPS.items():
                rng = np.random.default_rng(3_000_000 + seed * 100
                                            + list(ABLATION_DROPS).index(tag))
                cfg, ledger, _ = joint_bo(wl, cm, MULTI_WINDOWS, budget, rng,
                                          drop_terms=drops)
                agg = final_eval(wl, cfg, cm, TEST_WINDOWS, seed)  # full objective
                rows.append({"experiment": "exp2", "workload": kind, "seed": seed,
                             "method": f"ablate_{tag}",
                             "search_cost_units": ledger.spent,
                             **{f"cfg_{k}": v for k, v in cfg.items()}, **agg})
                print(f"exp2 {kind} seed={seed} {tag}: u={agg['utility']:.3f}",
                      flush=True)


def exp3_drift(workloads, seeds, budget, rows):
    variants = {
        "single_snapshot": dict(windows=TRAIN_WINDOWS),
        "multi_env": dict(windows=MULTI_WINDOWS),
        "cvar": dict(windows=MULTI_WINDOWS, cvar_alpha=0.3),
    }
    for kind in workloads:
        for seed in range(seeds):
            wl = Workload.make(kind, seed)
            cm = ContextModel(seed=500 + seed)
            picks = {}
            for tag, opts in variants.items():
                rng = np.random.default_rng(4_000_000 + seed * 100
                                            + list(variants).index(tag))
                kw = {k: v for k, v in opts.items() if k != "windows"}
                cfg, _, _ = joint_bo(wl, cm, opts["windows"], budget, rng, **kw)
                picks[tag] = cfg
            # hindsight oracle pool per window
            pool = [sample_config(space_for(kind), np.random.default_rng(5_000_000 + i))
                    for i in range(24)] + list(picks.values()) \
                + [fixed_policy("classical_only", wl), fixed_policy("always_simulator", wl)]
            for w in TEST_WINDOWS:
                oracle_u = max(final_eval(wl, c, cm, [w], seed, draws=6)["utility"]
                               for c in pool)
                for tag, cfg in picks.items():
                    agg = final_eval(wl, cfg, cm, [w], seed, draws=6)
                    rows.append({"experiment": "exp3", "workload": kind,
                                 "seed": seed, "method": tag, "window": w,
                                 "oracle_utility": oracle_u,
                                 "regret": oracle_u - agg["utility"], **agg})
            print(f"exp3 {kind} seed={seed} done", flush=True)


def exp4_transfer(workloads, seeds, budget, rows):
    """Backend transfer: search with qpu_A visible; deploy where only qpu_B
    exists (A's calibration degraded to unusable in test contexts)."""
    for kind in workloads:
        for seed in range(seeds):
            wl = Workload.make(kind, seed)
            cm_a = ContextModel(seed=500 + seed)
            cm_b = ContextModel(seed=9_500 + seed)  # different drift realization
            rng = np.random.default_rng(6_000_000 + seed)
            cfg_a, _, data = joint_bo(wl, cm_a, MULTI_WINDOWS, budget, rng)
            # zero-shot: deploy A-selected config under B's context model
            agg = final_eval(wl, cfg_a, cm_b, TEST_WINDOWS, seed)
            rows.append({"experiment": "exp4", "workload": kind, "seed": seed,
                         "method": "zero_shot", **agg})
            # warm-start: continue BO on B initialized with A's observations
            rng2 = np.random.default_rng(6_100_000 + seed)
            cfg_w, _, _ = joint_bo(wl, cm_b, MULTI_WINDOWS, budget / 2, rng2,
                                   init_data=data)
            agg = final_eval(wl, cfg_w, cm_b, TEST_WINDOWS, seed)
            rows.append({"experiment": "exp4", "workload": kind, "seed": seed,
                         "method": "warm_start_half_budget", **agg})
            # fresh search on B with full budget
            rng3 = np.random.default_rng(6_200_000 + seed)
            cfg_f, _, _ = joint_bo(wl, cm_b, MULTI_WINDOWS, budget, rng3)
            agg = final_eval(wl, cfg_f, cm_b, TEST_WINDOWS, seed)
            rows.append({"experiment": "exp4", "workload": kind, "seed": seed,
                         "method": "fresh_full_budget", **agg})
            print(f"exp4 {kind} seed={seed} done", flush=True)


def exp6_fidelity_consistency(workloads, seeds, rows):
    for kind in workloads:
        for seed in range(min(seeds, 4)):
            wl = Workload.make(kind, seed)
            cm = ContextModel(seed=500 + seed)
            rng = np.random.default_rng(8_000_000 + seed)
            pool = [sample_config(space_for(kind), rng) for _ in range(24)]
            for i, cfg in enumerate(pool):
                rec = {"experiment": "exp6", "workload": kind, "seed": seed,
                       "config_id": i, **{f"cfg_{k}": v for k, v in cfg.items()}}
                for fid in (0, 1, 2, 3):
                    u = mean_utility(wl, cfg, cm, TEST_WINDOWS,
                                     np.random.default_rng(8_500_000 + seed * 1000 + i),
                                     fid, 6 if fid else 1)
                    rec[f"u_fid{fid}"] = u
                rows.append(rec)
            print(f"exp6 {kind} seed={seed} done", flush=True)




def exp7_budget_sweep(workloads, seeds, rows, offset=0):
    """Search-efficiency curve: final deployment utility vs search budget."""
    methods = ["random", "joint_bo", "joint_mf"]
    for kind in workloads:
        for seed in range(offset, offset + seeds):
            wl = Workload.make(kind, seed)
            cm = ContextModel(seed=500 + seed)
            for budget in (10.0, 60.0):
                for method in methods:
                    rng = np.random.default_rng(9_000_000 + seed * 1000
                                                + int(budget) * 10
                                                + methods.index(method))
                    cfg, ledger = run_method(method, wl, cm, MULTI_WINDOWS,
                                             budget, rng)
                    agg = final_eval(wl, cfg, cm, TEST_WINDOWS, seed)
                    rows.append({"experiment": "exp7", "workload": kind,
                                 "seed": seed, "method": method,
                                 "budget": budget,
                                 "search_cost_units": ledger.spent,
                                 "search_cost_modeled_s": ledger.modeled_cost_s,
                                 **{f"cfg_{k}": v for k, v in cfg.items()},
                                 **agg})
            print(f"exp7 {kind} seed={seed} done", flush=True)


def exp8_scaling(seeds, rows):
    """Empirical per-evaluation and per-search cost vs problem size."""
    from coopt import evaluate, hand_designed
    for size in (6, 8, 9, 12):
        for seed in range(min(seeds, 4)):
            wl = Workload.make("qaoa", seed, size=size)
            cm = ContextModel(seed=500 + seed)
            cfg = hand_designed("qaoa")
            cfg["placement"] = "remote_sim"
            rng = np.random.default_rng(9_500_000 + seed)
            t0 = time.perf_counter()
            n_evals = 12
            for i in range(n_evals):
                ctx = cm.sample(rng, i % 3)
                evaluate(wl, cfg, ctx, 2, rng)
            per_eval = (time.perf_counter() - t0) / n_evals
            t1 = time.perf_counter()
            _, ledger = run_method("joint_bo", wl, cm, MULTI_WINDOWS, 10.0,
                                   np.random.default_rng(9_600_000 + seed))
            rows.append({"experiment": "exp8", "workload": "qaoa",
                         "size": size, "seed": seed,
                         "eval_wall_s": per_eval,
                         "search_wall_s": time.perf_counter() - t1,
                         "search_cost_units": ledger.spent})
            print(f"exp8 n={size} seed={seed} eval={per_eval*1e3:.1f}ms",
                  flush=True)




def exp10_mismatch(rows):
    """Re-score every exp1-selected configuration under a structurally
    mismatched deployment model (fid4: heavier calibration tail, misspecified
    gate-count exponent, readout corruption) to probe noise-model
    misspecification -- a simulation-side proxy for the hardware gap."""
    import pandas as pd
    src = pd.concat([pd.read_csv(OUT / f"raw_main_{t}.csv")
                     for t in ("qaoa", "vqc", "qaoa_s2", "vqc_s2")],
                    ignore_index=True)
    src = src[(src.experiment == "exp1")]
    cfg_cols = [c for c in src.columns if c.startswith("cfg_")]
    for _, r in src.iterrows():
        wl = Workload.make(r.workload, int(r.seed))
        cm = ContextModel(seed=500 + int(r.seed))
        cfg = {c[4:]: r[c] for c in cfg_cols if pd.notna(r[c])}
        for k in cfg:
            if k != "placement":
                cfg[k] = int(cfg[k]) if float(cfg[k]).is_integer() else float(cfg[k])
        for fid in (3, 4):
            agg = final_eval(wl, cfg, cm, TEST_WINDOWS, int(r.seed),
                             fidelity=fid)
            rows.append({"experiment": "exp10", "workload": r.workload,
                         "seed": int(r.seed), "method": r.method,
                         "fidelity": fid, **agg})
    print("exp10 done", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--budget", type=float, default=40.0)
    ap.add_argument("--workloads", default="qaoa,vqc")
    ap.add_argument("--experiments", default="1,2,3,4,6")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--methods", default="")
    ap.add_argument("--no-oracle", action="store_true")
    args = ap.parse_args()
    workloads = args.workloads.split(",")
    exps = set(args.experiments.split(","))
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    t0 = time.time()
    if "1" in exps:
        exp1_search_efficiency(workloads, args.seeds, args.budget, rows,
                               offset=args.seed_offset,
                               methods=args.methods.split(",") if args.methods
                               else None, oracle=not args.no_oracle)
    if "2" in exps:
        exp2_objective_ablation(workloads, args.seeds, args.budget, rows)
    if "3" in exps:
        exp3_drift(workloads, args.seeds, args.budget, rows)
    if "4" in exps:
        exp4_transfer(workloads, args.seeds, args.budget, rows)
    if "6" in exps:
        exp6_fidelity_consistency(workloads, args.seeds, rows)
    if "7" in exps:
        exp7_budget_sweep(workloads, args.seeds, rows, offset=args.seed_offset)
    if "8" in exps:
        exp8_scaling(args.seeds, rows)
    if "10" in exps:
        exp10_mismatch(rows)
    df = pd.DataFrame(rows)
    out_csv = OUT / f"raw_{args.tag}.csv"
    df.to_csv(out_csv, index=False)
    manifest = {
        "created_unix": time.time(), "elapsed_s": time.time() - t0,
        "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "seeds": args.seeds, "budget": args.budget, "workloads": workloads,
        "experiments": sorted(exps),
        "quantum_execution": "exact statevector + modeled depolarizing/shot noise",
        "deployment_fidelity": "held-out perturbed noisy simulation (fid3); no hardware",
        "cost_figures": "modeled, not measured",
    }
    (OUT / f"manifest_{args.tag}.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(df)} rows to {out_csv} in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
