#!/usr/bin/env python3
"""Propagate held-out runtime-model residuals through the trace contrast."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split

from coopt import Context, Workload, evaluate
from run_fable_trace_replay import (
    DAQEC,
    FEATURES,
    config_from_row,
    job_features,
    load_drift,
    load_queue,
    replace_runtime,
)


ARTIFACT = Path(__file__).resolve().parent
RESULTS = ARTIFACT / "results"
GENERATED = ARTIFACT.parent / "submission" / "paper" / "generated"


def fit_runtime_with_residuals(data: pd.DataFrame):
    features = np.log1p(data[FEATURES].clip(lower=0).to_numpy(dtype=float))
    target = np.log1p(data.run_s.to_numpy(dtype=float))
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.20, random_state=20260901
    )
    model = ExtraTreesRegressor(
        n_estimators=256,
        min_samples_leaf=3,
        random_state=20260901,
        n_jobs=-1,
    ).fit(x_train, y_train)
    residuals = y_test - model.predict(x_test)
    model.fit(features, target)
    return model, residuals


def context_pools(queue: pd.DataFrame):
    backend_medians = queue.groupby("machine").queue_s.median().sort_values()
    split = float(backend_medians.median())
    fast_names = backend_medians[backend_medians <= split].index
    slow_names = backend_medians[backend_medians > split].index
    fast_queue = np.sort(queue[queue.machine.isin(fast_names)].queue_s.to_numpy())
    slow_queue = np.sort(queue[queue.machine.isin(slow_names)].queue_s.to_numpy())
    drift_pools, _ = load_drift()
    return fast_queue, slow_queue, drift_pools["ibm_kyoto"], drift_pools["ibm_osaka"]


def qpu_contexts(row, fast_queue, slow_queue, eps_a, eps_b):
    contexts = []
    for draw in range(20):
        index = int(row.seed) * 37 + draw * 53
        contexts.append((index, Context(
            window=draw,
            eps={"qpu_A": float(eps_a[index % len(eps_a)]),
                 "qpu_B": float(eps_b[(index * 3) % len(eps_b)])},
            queue_s={"qpu_A": float(slow_queue[index % len(slow_queue)]),
                     "qpu_B": float(fast_queue[(index * 7) % len(fast_queue)])},
            rtt_s=0.03,
            fail_prob=0.001,
            deadline_s=float(np.exp(np.random.default_rng(70_000 + index).normal(math.log(1.5), 0.9))),
            energy_budget_j=400.0,
            money_budget=1.0,
            volatility=0.05,
            reuse_count=32,
        )))
    return contexts


def main() -> int:
    queue, _ = load_queue()
    model, log_residuals = fit_runtime_with_residuals(queue)
    fast_queue, slow_queue, eps_a, eps_b = context_pools(queue)
    selected = pd.read_csv(RESULTS / "clearaccept_confirmatory.csv")
    selected = selected[selected.method.isin(("joint_mf", "sequential"))]
    selected = selected.set_index(["workload", "seed", "method"])
    replay = pd.read_csv(RESULTS / "fable_trace_replay.csv")
    replay = replay[replay.method.isin(("joint_mf", "sequential"))]
    pivot = replay.pivot(index=["workload", "seed"], columns="method", values="utility")
    base_delta = (pivot.joint_mf - pivot.sequential).to_dict()

    qpu_rows = replay[(replay.method == "sequential") & replay.placement.isin(("qpu_A", "qpu_B"))]
    repetitions = 10_000
    prepared = {}
    for row in qpu_rows.itertuples():
        source = selected.loc[(row.workload, row.seed, "sequential")]
        source = source.copy()
        source["workload"] = row.workload
        source["seed"] = row.seed
        wl = Workload.make(row.workload, int(row.seed))
        cfg = config_from_row(source)
        base_delay = []
        trace_runtime = []
        deadline = []
        constant_loss = []
        for index, context in qpu_contexts(source, fast_queue, slow_queue, eps_a, eps_b):
            comp = evaluate(wl, cfg, context, 2, np.random.default_rng(80_000_000 + index))
            zero = replace_runtime(comp, wl, cfg, context, model, runtime_multiplier=0.0)
            point = replace_runtime(comp, wl, cfg, context, model, runtime_multiplier=1.0)
            base_delay.append(zero["delay_s"])
            trace_runtime.append(point["trace_runtime_s"])
            deadline.append(context.deadline_s)
            constant_loss.append(
                -zero["utility"] - zero["staleness"] - 2.0 * zero["deadline_miss"]
            )
        multipliers = np.exp(
            np.random.default_rng(881_000 + int(row.seed)).choice(
                log_residuals, size=(repetitions, len(base_delay))
            )
        )
        delay = np.asarray(base_delay)[None, :] + np.asarray(trace_runtime)[None, :] * multipliers
        utility = -(
            np.asarray(constant_loss)[None, :]
            + 0.05 * delay
            + 2.0 * (delay > np.asarray(deadline)[None, :])
        )
        prepared[(row.workload, row.seed)] = utility.mean(axis=1)

    clusters = list(base_delta)
    rng = np.random.default_rng(880_000)
    contrasts = np.empty(repetitions)
    for repetition in range(repetitions):
        delta = dict(base_delta)
        for key, sequential_utility in prepared.items():
            joint = float(replay[
                (replay.workload == key[0])
                & (replay.seed == key[1])
                & (replay.method == "joint_mf")
            ].utility.iloc[0])
            delta[key] = joint - float(sequential_utility[repetition])
        sampled = rng.integers(0, len(clusters), len(clusters))
        contrasts[repetition] = np.mean([delta[clusters[index]] for index in sampled])

    report = {
        "runtime_model_heldout_log_residuals": len(log_residuals),
        "sequential_qpu_clusters": len(prepared),
        "total_clusters": len(clusters),
        "nested_bootstrap_repetitions": repetitions,
        "mean_utility_contrast": float(contrasts.mean()),
        "interval_95": [float(x) for x in np.percentile(contrasts, [2.5, 97.5])],
        "probability_positive": float((contrasts > 0).mean()),
        "scope": "fixed selected configurations with runtime residual propagation",
        "daquec_path": str(DAQEC.relative_to(ARTIFACT)),
    }
    (RESULTS / "trace_runtime_uncertainty.json").write_text(json.dumps(report, indent=2))
    macros = [
        f"\\newcommand{{\\RuntimeResiduals}}{{{len(log_residuals)}}}",
        f"\\newcommand{{\\RuntimeClusters}}{{{len(clusters)}}}",
        f"\\newcommand{{\\RuntimeUncertainMean}}{{{report['mean_utility_contrast']:+.2f}}}",
        f"\\newcommand{{\\RuntimeUncertainLo}}{{{report['interval_95'][0]:+.2f}}}",
        f"\\newcommand{{\\RuntimeUncertainHi}}{{{report['interval_95'][1]:+.2f}}}",
        f"\\newcommand{{\\RuntimeUncertainPositive}}{{{100*report['probability_positive']:.1f}\\%}}",
        f"\\newcommand{{\\RuntimeQpuClusters}}{{{len(prepared)}}}",
    ]
    (GENERATED / "trace_runtime_uncertainty_numbers.tex").write_text("\n".join(macros) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
