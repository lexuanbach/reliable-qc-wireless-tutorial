#!/usr/bin/env python3
"""Fresh-seed CAPO component, risk, and modular-baseline experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import resource
import time

import numpy as np
import pandas as pd

from coopt import ContextModel, Workload, evaluate, utility_from_components
from search import METHODS_MAIN, run_method


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
WORKLOADS = ["qaoa", "qaoa_channel", "qaoa_place", "vqc", "vqc_cancer"]
METHODS = [
    "sequential",
    "capo_joint_only",
    "random",
    "capo_mf_no_rescore",
    "joint_mf",
    "capo_chance",
    "device_aware_sequential",
    "fixed_design_scheduler",
    "classical_only",
]
MULTI_WINDOWS = [0, 1, 2]
TEST_WINDOWS = list(range(9))


def final_eval(wl, cfg, cm, windows, seed, draws=40, fidelity=3):
    """Paired held-out evaluation with mean, tail, and reliability outcomes."""
    utilities = []; comps = []
    for i in range(draws):
        ctx = cm.sample(np.random.default_rng(97_000_000 + seed * 1000 + i),
                        windows[i % len(windows)])
        eval_rng = np.random.default_rng(98_000_000 + seed * 1000 + i)
        comp = evaluate(wl, cfg, ctx, fidelity, eval_rng)
        comps.append(comp); utilities.append(utility_from_components(comp))
    keys = ("quality_loss", "delay_s", "deadline_miss", "infeasible",
            "energy_j", "money", "cost_scalar", "realized_fail")
    agg = {key: float(np.mean([c[key] for c in comps])) for key in keys}
    utilities = np.asarray(utilities)
    k = max(1, int(np.ceil(0.10 * len(utilities))))
    agg.update({
        "utility": float(utilities.mean()),
        "utility_p10": float(np.percentile(utilities, 10)),
        "utility_cvar10": float(np.sort(utilities)[:k].mean()),
        "utility_min": float(utilities.min()),
        "service_violation": float(np.mean([
            max(c["deadline_miss"], c["infeasible"], c["realized_fail"])
            for c in comps])),
    })
    return agg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--seed-offset", type=int, default=100)
    parser.add_argument("--budget", type=float, default=30.0)
    parser.add_argument("--draws", type=int, default=40)
    parser.add_argument("--workloads", default=",".join(WORKLOADS))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--out", default="clearaccept_confirmatory")
    args = parser.parse_args()
    workloads = args.workloads.split(","); methods = args.methods.split(",")
    rows = []; started = time.time()
    for workload in workloads:
        for seed in range(args.seed_offset, args.seed_offset + args.seeds):
            wl = Workload.make(workload, seed)
            cm = ContextModel(seed=500 + seed)
            for method in methods:
                rng = np.random.default_rng(91_000_000 + seed * 1000
                                            + METHODS_MAIN.index(method))
                t0 = time.perf_counter()
                cfg, ledger = run_method(method, wl, cm, MULTI_WINDOWS,
                                         args.budget, rng)
                search_wall = time.perf_counter() - t0
                agg = final_eval(wl, cfg, cm, TEST_WINDOWS, seed,
                                 draws=args.draws)
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                rows.append({
                    "experiment": "clearaccept_confirmatory",
                    "workload": workload,
                    "seed": seed,
                    "method": method,
                    "budget": args.budget,
                    "search_cost_units": ledger.spent,
                    "search_cost_modeled_s": ledger.modeled_cost_s,
                    "search_evaluations": ledger.evaluations,
                    "unique_candidates": len(ledger.unique_configs),
                    "fidelity_counts": json.dumps(ledger.fidelity_counts, sort_keys=True),
                    "search_wall_s": search_wall,
                    "peak_rss_mb": float(rss / (1024.0 if platform.system() == "Linux" else 1024.0**2)),
                    **{f"cfg_{key}": value for key, value in cfg.items()},
                    **agg,
                })
                print(f"confirm {workload} seed={seed} {method}: "
                      f"u={agg['utility']:.3f} risk={agg['service_violation']:.3f}",
                      flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / f"{args.out}.csv", index=False)
    (OUT / f"{args.out}_manifest.json").write_text(json.dumps({
        "created_unix": time.time(), "elapsed_s": time.time() - started,
        "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(), "workloads": workloads,
        "methods": methods, "seeds": args.seeds, "seed_offset": args.seed_offset,
        "budget": args.budget, "deployment_draws": args.draws,
        "analysis_status": "fresh confirmatory seeds, simulator-first; no QPU measurement",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
