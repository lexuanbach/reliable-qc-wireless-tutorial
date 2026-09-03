#!/usr/bin/env python3
"""Public IBM-job queue/runtime and calibration-drift replay for CAPO.

QuantumQueue supplies public job records (times are minutes in its capture
notebook); DAQEC supplies dated backend calibration snapshots.  Circuit
quality is still simulated.  Consequently every output is labeled
``public-trace-calibrated simulation``, never hardware execution.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import coopt
from coopt import (BACKEND_BASE, COMPILE_GATE_FACTOR, Context, Workload,
                   evaluate, utility_from_components)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
QUEUE = ROOT / "data" / "QuantumQueue"
DAQEC = ROOT / "data" / "daquec" / "master.parquet"
METHODS = ["joint_mf", "capo_chance", "sequential",
           "device_aware_sequential", "fixed_design_scheduler", "classical_only"]
FEATURES = ["batch", "shots", "depth", "width", "qubits", "gateops", "pulse"]


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_queue():
    frames, files = [], []
    for path in sorted(QUEUE.glob("*.csv")):
        frame = pd.read_csv(path)
        if {"run_time", "queue_time", "status", "machine", *FEATURES}.issubset(frame.columns):
            frames.append(frame); files.append({"file": path.name, "sha256": sha256(path), "rows": len(frame)})
    data = pd.concat(frames, ignore_index=True)
    data = data[data.status.astype(str).str.upper().str.contains("DONE", na=False)
                & ~data.machine.astype(str).str.lower().str.contains("simulator", na=False)].copy()
    for column in ["run_time", "queue_time", *FEATURES]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["run_time", "queue_time", *FEATURES])
    data = data[(data.run_time > 0) & (data.queue_time >= 0)].copy()
    data["machine"] = data.machine.astype(str).str.strip("'")
    data["run_s"] = 60.0 * data.run_time
    data["queue_s"] = 60.0 * data.queue_time
    return data, files


def runtime_model(data):
    X = np.log1p(data[FEATURES].clip(lower=0).to_numpy(dtype=float))
    y = np.log1p(data.run_s.to_numpy(dtype=float))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.20, random_state=20260901)
    model = ExtraTreesRegressor(n_estimators=256, min_samples_leaf=3,
                                random_state=20260901, n_jobs=-1).fit(Xtr, ytr)
    predicted = np.expm1(model.predict(Xte)); actual = np.expm1(yte)
    metrics = {
        "test_rows": len(actual), "r2_log_runtime": float(r2_score(yte, model.predict(Xte))),
        "mae_runtime_s": float(mean_absolute_error(actual, predicted)),
        "median_absolute_error_s": float(np.median(np.abs(actual - predicted))),
    }
    # Refit on every record after holding out the diagnostic split.
    model.fit(X, y)
    return model, metrics


def load_drift():
    data = pd.read_parquet(DAQEC)
    data["timestamp"] = pd.to_datetime(data["timestamp_utc"], errors="coerce", utc=True)
    data = data.dropna(subset=["backend", "timestamp", "avg_ecr_error"])
    daily = data.groupby(["backend", "timestamp"], as_index=False).agg(
        avg_ecr_error=("avg_ecr_error", "median"),
        avg_readout_error=("avg_readout_error", "median"))
    pools = {name: group.sort_values("timestamp").avg_ecr_error.to_numpy(dtype=float)
             for name, group in daily.groupby("backend")}
    drift = {}
    for name, values in pools.items():
        drift[name] = {
            "snapshots": len(values), "mean_ecr_error": float(values.mean()),
            "sd_ecr_error": float(values.std(ddof=1)), "minimum": float(values.min()),
            "maximum": float(values.max()),
            "log_increment_sd": float(np.diff(np.log(values)).std(ddof=1)),
        }
    return pools, drift


def config_from_row(row):
    if row.workload.startswith("qaoa"):
        ints = ("p", "opt_iters", "warm_start", "shots", "mitigation", "compile_level")
    else:
        ints = ("n_qubits", "layers", "reupload", "train_iters", "shots", "mitigation", "compile_level")
    cfg = {key: int(getattr(row, f"cfg_{key}")) for key in ints}
    if not row.workload.startswith("qaoa"):
        cfg["lr"] = float(row.cfg_lr)
    cfg["placement"] = row.cfg_placement
    return cfg


def job_features(wl, cfg):
    if wl.kind.startswith("qaoa"):
        n = wl.inst.n_vars; depth = int(cfg["p"] * (n + 1))
        interactions = wl.inst.metadata.get("edges", int(round(1.5 * n)))
        g2 = max(1, int(round(interactions * cfg["p"]
                            * COMPILE_GATE_FACTOR[cfg["compile_level"]])))
        batch = 1 if cfg["warm_start"] else cfg["opt_iters"]
        gateops = g2 + n * cfg["p"]
    else:
        n = cfg["n_qubits"]; layer_depth = cfg["layers"] * (2 if cfg["reupload"] else 1)
        depth = int(layer_depth * (n + 1)); batch = 32
        g2 = max(1, int(round(n * layer_depth * COMPILE_GATE_FACTOR[cfg["compile_level"]])))
        gateops = g2 + n * layer_depth
    return np.log1p(np.array([[batch, cfg["shots"], depth, n, n, gateops, 0.0]], dtype=float))


def replace_runtime(comp, wl, cfg, ctx, model, runtime_multiplier=1.0):
    placement = cfg["placement"]
    if placement not in ("qpu_A", "qpu_B"):
        return comp
    predicted_job_s = float(np.expm1(model.predict(job_features(wl, cfg))[0]))
    predicted_job_s *= runtime_multiplier
    shots_eff = cfg["shots"] * (coopt.MITIGATION_SHOT_MULTIPLIER if cfg["mitigation"] else 1)
    _, _, per_circuit, per_shot, _ = BACKEND_BASE[placement]
    if wl.kind.startswith("qaoa"):
        evaluations = 1 if cfg["warm_start"] else cfg["opt_iters"]
        modeled_core = evaluations * (per_circuit + shots_eff * per_shot)
        trace_core = predicted_job_s
    else:
        modeled_core = per_circuit + shots_eff * per_shot
        trace_core = predicted_job_s / 32.0
    adjusted = dict(comp)
    adjusted["delay_s"] = max(0.0, comp["delay_s"] - modeled_core + trace_core)
    adjusted["deadline_miss"] = float(adjusted["delay_s"] > ctx.deadline_s)
    adjusted["staleness"] = ctx.volatility * adjusted["delay_s"]
    adjusted["utility"] = utility_from_components(adjusted)
    adjusted["cost_scalar"] = coopt.cost_scalar(adjusted)
    adjusted["trace_runtime_s"] = trace_core
    return adjusted


def main():
    queue, queue_files = load_queue()
    runtime, runtime_metrics = runtime_model(queue)
    drift_pools, drift_stats = load_drift()

    backend_medians = queue.groupby("machine").queue_s.median().sort_values()
    split = float(backend_medians.median())
    fast_names = backend_medians[backend_medians <= split].index
    slow_names = backend_medians[backend_medians > split].index
    fast_queue = np.sort(queue[queue.machine.isin(fast_names)].queue_s.to_numpy())
    slow_queue = np.sort(queue[queue.machine.isin(slow_names)].queue_s.to_numpy())
    # Preserve the paper's quality/latency tension: A is lower-error but slower;
    # B is higher-error but faster.  These are trace strata, not backend aliases.
    eps_a = drift_pools["ibm_kyoto"]
    eps_b = drift_pools["ibm_osaka"]

    selected = pd.read_csv(RESULTS / "clearaccept_confirmatory.csv")
    selected = selected[selected.method.isin(METHODS)].copy()
    rows = []
    draws = 20
    for row in selected.itertuples():
        wl = Workload.make(row.workload, int(row.seed))
        cfg = config_from_row(row)
        comps = []
        for draw in range(draws):
            index = int(row.seed) * 37 + draw * 53
            context = Context(
                window=draw,
                eps={"qpu_A": float(eps_a[index % len(eps_a)]),
                     "qpu_B": float(eps_b[(index * 3) % len(eps_b)])},
                queue_s={"qpu_A": float(slow_queue[index % len(slow_queue)]),
                         "qpu_B": float(fast_queue[(index * 7) % len(fast_queue)])},
                rtt_s=0.03, fail_prob=0.001,
                deadline_s=float(np.exp(np.random.default_rng(70_000 + index).normal(math.log(1.5), .9))),
                energy_budget_j=400.0, money_budget=1.0,
                volatility=0.05, reuse_count=32,
            )
            comp = evaluate(wl, cfg, context, 2,
                            np.random.default_rng(80_000_000 + index))
            comps.append(replace_runtime(comp, wl, cfg, context, runtime))
        utilities = np.array([c["utility"] for c in comps])
        violation = [max(c["deadline_miss"], c["infeasible"]) for c in comps]
        rows.append({
            "experiment": "public_trace_calibrated_simulation",
            "workload": row.workload, "seed": int(row.seed), "method": row.method,
            "placement": cfg["placement"], "draws": draws,
            "utility": float(utilities.mean()), "utility_p10": float(np.percentile(utilities, 10)),
            "service_violation": float(np.mean(violation)),
            "deadline_miss": float(np.mean([c["deadline_miss"] for c in comps])),
            "quality_loss": float(np.mean([c["quality_loss"] for c in comps])),
            "delay_s": float(np.mean([c["delay_s"] for c in comps])),
            "trace_runtime_s": float(np.mean([c.get("trace_runtime_s", 0.0) for c in comps])),
        })
        print(f"trace {row.workload} {row.seed} {row.method}", flush=True)
    output = pd.DataFrame(rows)
    output.to_csv(RESULTS / "fable_trace_replay.csv", index=False)

    try:
        commit = subprocess.check_output(["git", "-C", str(QUEUE), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    manifest = {
        "evidence_label": "public-trace-calibrated simulation; no CAPO hardware execution",
        "quantumqueue": {
            "url": "https://github.com/rgokulsm/QuantumQueue", "commit": commit,
            "rows": len(queue), "files": queue_files,
            "minutes_to_seconds_evidence": "capture-the-data-share.ipynb divides total_seconds by 60",
            "queue_minutes": {f"q{q}": float(np.quantile(queue.queue_time, q / 100))
                              for q in (10, 25, 50, 75, 90, 95, 99)},
            "fraction_over_120_minutes": float((queue.queue_time > 120).mean()),
            "fraction_over_1440_minutes": float((queue.queue_time > 1440).mean()),
            "backend_median_split_s": split,
        },
        "runtime_model": runtime_metrics,
        "daquec": {"url": "https://zenodo.org/records/18045662",
                   "master_sha256": sha256(DAQEC), "backend_statistics": drift_stats,
                   "mapping": {"qpu_A_error": "ibm_kyoto", "qpu_B_error": "ibm_osaka"}},
        "replay": {"methods": METHODS, "draws_per_selected_configuration": draws,
                   "rows": len(output), "fidelity": 2},
    }
    (RESULTS / "fable_trace_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
