#!/usr/bin/env python3
"""Generate manuscript assets for the Fable-review experiments."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams.update({"pdf.fonttype": 42, "font.family": "serif",
                     "font.serif": ["Times New Roman", "Times",
                                    "Nimbus Roman", "STIXGeneral"],
                     "mathtext.fontset": "stix"})
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
if not RESULTS.is_dir():  # standalone artifact layout
    RESULTS = Path(__file__).resolve().parent / "results"
PAPER = ROOT / "submission" / "paper"
GEN = PAPER / "generated"
FIG = PAPER / "figures"
ROW_END = r"\\"


def paired_bootstrap(frame, a, b, value, seed=20260901):
    pivot = frame.pivot_table(index=["workload", "seed"], columns="method", values=value)
    difference = (pivot[a] - pivot[b]).dropna().to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    samples = difference[rng.integers(0, len(difference), (20_000, len(difference)))].mean(1)
    return float(difference.mean()), *map(float, np.percentile(samples, [2.5, 97.5]))


def write_table(name, header, rows, spec):
    (GEN / name).write_text("\n".join([
        f"\\begin{{tabular}}{{@{{}}{spec}@{{}}}}", r"\toprule", header, r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", ""]))


def main():
    GEN.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    trace = pd.read_csv(RESULTS / "fable_trace_replay.csv")
    trace_summary = trace.groupby("method", as_index=False).agg(
        utility=("utility", "mean"), service_violation=("service_violation", "mean"),
        deadline_miss=("deadline_miss", "mean"), delay_s=("delay_s", "mean"))
    order = ["joint_mf", "capo_chance", "sequential", "device_aware_sequential",
             "fixed_design_scheduler", "classical_only"]
    names = {"joint_mf": "CAPO-MF", "capo_chance": "CAPO-Chance",
             "sequential": "Sequential", "device_aware_sequential": "Device-aware seq.",
             "fixed_design_scheduler": "Fixed-design sched.", "classical_only": "Classical only"}
    indexed = trace_summary.set_index("method")
    rows = [f"{names[m]} & {indexed.loc[m, 'utility']:.3f} & "
            f"{100*indexed.loc[m, 'service_violation']:.1f}\\% & "
            f"{100*indexed.loc[m, 'deadline_miss']:.1f}\\% & {indexed.loc[m, 'delay_s']:.2f} " + ROW_END
            for m in order]
    write_table("tab_fable_trace.tex", "Method & Utility & Violation & Late & Delay s " + ROW_END,
                rows, "lrrrr")
    trace_u = paired_bootstrap(trace, "joint_mf", "sequential", "utility")
    trace_risk = paired_bootstrap(trace, "joint_mf", "sequential", "service_violation", 20260902)
    chance_risk = paired_bootstrap(trace, "capo_chance", "sequential", "service_violation", 20260903)

    winsor = pd.read_csv(RESULTS / "fable_winsor_sensitivity.csv")
    win_summary = winsor.groupby("winsor_floor", as_index=False).agg(
        utility=("utility", "mean"), violation=("service_violation", "mean"))
    win_summary["sort"] = win_summary.winsor_floor.map({"-2": 0, "-4": 1, "-6": 2, "-10": 3, "none": 4})
    rows = [f"{r.winsor_floor} & {r.utility:.4f} & {100*r.violation:.1f}\\% " + ROW_END
            for r in win_summary.sort_values("sort").itertuples()]
    write_table("tab_fable_winsor.tex", "Floor & Utility & Violation " + ROW_END, rows, "lrr")

    mitigation = pd.read_csv(RESULTS / "fable_mitigation_sensitivity.csv")
    mitigation_summary = mitigation.groupby(["shot_multiplier", "residual_error_multiplier"], as_index=False).agg(
        utility=("utility", "mean"), violation=("service_violation", "mean"), delay_s=("delay_s", "mean"))
    rows = [f"{r.shot_multiplier:.0f}$\\times$ & {r.residual_error_multiplier:.2f} & "
            f"{r.utility:.3f} & {100*r.violation:.1f}\\% & {r.delay_s:.3f} " + ROW_END
            for r in mitigation_summary.itertuples()]
    write_table("tab_fable_mitigation.tex", "Shot cost & Residual error & Utility & Violation & Delay s " + ROW_END,
                rows, "rrrrr")

    digits = pd.read_csv(RESULTS / "fable_digits.csv")
    digits_summary = digits.groupby("method", as_index=False).agg(
        utility=("utility", "mean"), violation=("service_violation", "mean"),
        quality_loss=("quality_loss", "mean"))
    indexed_digits = digits_summary.set_index("method")
    rows = [f"{names[m]} & {indexed_digits.loc[m, 'utility']:.4f} & "
            f"{indexed_digits.loc[m, 'quality_loss']:.4f} & {100*indexed_digits.loc[m, 'violation']:.1f}\\% " + ROW_END
            for m in order]
    write_table("tab_fable_digits.tex", "Method & Utility & Error & Violation " + ROW_END,
                rows, "lrrr")
    digits_u = paired_bootstrap(digits, "joint_mf", "sequential", "utility", 20260904)

    # Deployment-only money/energy rescaling for comparator-aware amortization.
    raw = pd.concat([pd.read_csv(path) for path in sorted(RESULTS.glob("raw_main_*.csv"))],
                    ignore_index=True)
    raw = raw[raw.experiment == "exp1"].drop_duplicates(
        ["workload", "seed", "method"], keep="last")
    raw = raw[raw.workload.isin(["qaoa", "qaoa_channel", "qaoa_place", "vqc", "vqc_cancer"])].copy()
    original_cost_terms = .5 * raw.money / .01 + .3 * raw.energy_j / 50.0
    amortization = []
    for scale in (.5, 1.0, 1.5):
        raw["rescaled_utility"] = raw.utility - (scale - 1.0) * original_cost_terms
        utility = raw.pivot_table(index=["workload", "seed"], columns="method", values="rescaled_utility")
        cost = raw.pivot_table(index=["workload", "seed"], columns="method",
                               values="search_cost_modeled_s").fillna(0.0)
        for rival in ("random", "feasible_random", "classical_only"):
            common = utility[["joint_bo", rival]].dropna().index.intersection(
                cost[["joint_bo", rival]].dropna().index)
            gain = (utility.loc[common, "joint_bo"] - utility.loc[common, rival]).to_numpy()
            incremental = np.maximum(0.0, (cost.loc[common, "joint_bo"] - cost.loc[common, rival]).to_numpy())
            finite = gain > 0
            break_even = incremental[finite] / gain[finite]
            amortization.append({"money_energy_scale": scale, "rival": rival,
                                 "cases": len(gain), "finite_fraction": float(finite.mean()),
                                 "median_break_even": float(np.median(break_even)) if len(break_even) else None})
    amortization_frame = pd.DataFrame(amortization)
    amortization_frame.to_csv(RESULTS / "fable_amortization_weight_sensitivity.csv", index=False)

    # Empirical CDF from precisely the complete-feature runtime calibration rows.
    queue_frames = []
    _qq = ROOT / "benchmark" / "data" / "QuantumQueue"
    if not _qq.is_dir():  # standalone artifact layout
        _qq = Path(__file__).resolve().parent / "data" / "QuantumQueue"
    for path in sorted(_qq.glob("*.csv")):
        frame = pd.read_csv(path)
        if {"run_time", "queue_time", "status", "machine", "batch", "shots", "depth",
            "width", "qubits", "gateops", "pulse"}.issubset(frame.columns):
            queue_frames.append(frame)
    queue = pd.concat(queue_frames, ignore_index=True)
    queue = queue[queue.status.astype(str).str.contains("DONE", na=False)
                  & ~queue.machine.astype(str).str.lower().str.contains("simulator", na=False)].copy()
    required = ["run_time", "queue_time", "batch", "shots", "depth", "width", "qubits", "gateops", "pulse"]
    for column in required:
        queue[column] = pd.to_numeric(queue[column], errors="coerce")
    queue = queue.dropna(subset=required)
    queue = queue[(queue.run_time > 0) & (queue.queue_time >= 0)]
    q = np.sort(queue.queue_time.to_numpy()); cdf = np.arange(1, len(q) + 1) / len(q)
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    ax.semilogx(np.maximum(q, 1e-3), cdf, color="#336699", linewidth=1.5)
    ax.axvline(120, color="#aa3333", linestyle="--", linewidth=1, label="2 hours")
    ax.set_xlabel("Queue time (minutes, log scale)"); ax.set_ylabel("Empirical CDF")
    ax.set_ylim(0, 1.01); ax.grid(alpha=.25); ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "fig_fable_queue_cdf.pdf", bbox_inches="tight"); plt.close(fig)

    manifest = json.loads((RESULTS / "fable_trace_manifest.json").read_text())
    macros = [
        f"\\newcommand{{\\FableTraceU}}{{{trace_u[0]:+.3f}}}",
        f"\\newcommand{{\\FableTraceULo}}{{{trace_u[1]:+.3f}}}",
        f"\\newcommand{{\\FableTraceUHi}}{{{trace_u[2]:+.3f}}}",
        f"\\newcommand{{\\FableTraceRisk}}{{{100*trace_risk[0]:+.2f}}}",
        f"\\newcommand{{\\FableTraceRiskLo}}{{{100*trace_risk[1]:+.2f}}}",
        f"\\newcommand{{\\FableTraceRiskHi}}{{{100*trace_risk[2]:+.2f}}}",
        f"\\newcommand{{\\FableTraceRiskReduction}}{{{-100*trace_risk[0]:.1f}}}",
        f"\\newcommand{{\\FableChanceTraceRisk}}{{{100*chance_risk[0]:+.2f}}}",
        f"\\newcommand{{\\FableChanceTraceRiskLo}}{{{100*chance_risk[1]:+.2f}}}",
        f"\\newcommand{{\\FableChanceTraceRiskHi}}{{{100*chance_risk[2]:+.2f}}}",
        f"\\newcommand{{\\FableDigitsU}}{{{digits_u[0]:+.4f}}}",
        f"\\newcommand{{\\FableDigitsULo}}{{{digits_u[1]:+.4f}}}",
        f"\\newcommand{{\\FableDigitsUHi}}{{{digits_u[2]:+.4f}}}",
        f"\\newcommand{{\\FableQueueRows}}{{{manifest['quantumqueue']['rows']}}}",
        f"\\newcommand{{\\FableQueueMedian}}{{{manifest['quantumqueue']['queue_minutes']['q50']:.1f}}}",
        f"\\newcommand{{\\FableRuntimeRtwo}}{{{manifest['runtime_model']['r2_log_runtime']:.2f}}}",
    ]
    (GEN / "fable_numbers.tex").write_text("\n".join(macros) + "\n")
    report = {"trace": trace_summary.to_dict("records"), "trace_joint_mf_vs_sequential": trace_u,
              "trace_risk_joint_mf_vs_sequential": trace_risk,
              "trace_risk_chance_vs_sequential": chance_risk,
              "winsor": win_summary.drop(columns="sort").to_dict("records"),
              "mitigation": mitigation_summary.to_dict("records"),
              "digits": digits_summary.to_dict("records"), "digits_joint_mf_vs_sequential": digits_u,
              "amortization_weight_sensitivity": amortization_frame.to_dict("records")}
    (RESULTS / "fable_analysis.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
