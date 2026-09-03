#!/usr/bin/env python3
"""Analyze CAPO clear-accept confirmatory and component experiments."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"pdf.fonttype": 42, "font.family": "serif",
                     "font.serif": ["Times New Roman", "Times",
                                    "Nimbus Roman", "STIXGeneral"],
                     "mathtext.fontset": "stix"})
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
if not RES.is_dir():  # standalone artifact layout
    RES = Path(__file__).resolve().parent / "results"
GEN = ROOT / "submission" / "paper" / "generated"
FIG = ROOT / "submission" / "paper" / "figures"
FAMILIES = ["qaoa", "qaoa_channel", "qaoa_place", "vqc", "vqc_cancer"]
ROW_END = r"\\"


def boot(values, seed=20260901):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), (20_000, len(values)))].mean(1)
    return float(values.mean()), *map(float, np.percentile(sampled, [2.5, 97.5]))


def stratified(pivot, a, b, seed=20260902):
    groups = []
    for family in FAMILIES:
        sub = pivot.xs(family)[[a, b]].dropna()
        groups.append((sub[a] - sub[b]).to_numpy())
    rng = np.random.default_rng(seed); draws = np.empty(20_000)
    for i in range(len(draws)):
        draws[i] = np.mean([g[rng.integers(0, len(g), len(g))].mean() for g in groups])
    return float(np.mean([g.mean() for g in groups])), *map(float, np.percentile(draws, [2.5, 97.5]))


def emit_table(name, spec, header, rows):
    (GEN / name).write_text("\n".join([
        f"\\begin{{tabular}}{{@{{}}{spec}@{{}}}}", r"\toprule", header,
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]) + "\n")


def main() -> int:
    GEN.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(RES / "clearaccept_confirmatory.csv")
    method_names = {
        "capo_joint_only": "CAPO-Joint",
        "random": "CAPO-Random",
        "capo_mf_no_rescore": "CAPO-MF/no-rescore",
        "joint_mf": "CAPO-MF",
        "capo_chance": "CAPO-Chance",
        "sequential": "Sequential",
        "device_aware_sequential": "Device-aware+schedule",
        "fixed_design_scheduler": "Fixed-design scheduler",
        "classical_only": "Classical only",
    }
    metrics = ["utility", "utility_cvar10", "service_violation", "deadline_miss",
               "infeasible", "quality_loss"]
    pivots = {metric: data.pivot_table(index=["workload", "seed"],
                                        columns="method", values=metric)
              for metric in metrics}
    primary = {
        "mean_utility": stratified(pivots["utility"], "joint_mf", "sequential"),
        "cvar10": stratified(pivots["utility_cvar10"], "joint_mf", "sequential"),
        "service_violation_reduction": stratified(
            pivots["service_violation"], "sequential", "joint_mf"),
        "chance_vs_sequential_violation_reduction": stratified(
            pivots["service_violation"], "sequential", "capo_chance"),
    }
    components = {}
    for a, b, tag in [
        ("random", "capo_joint_only", "fresh_rescore_random"),
        ("joint_mf", "capo_mf_no_rescore", "fresh_rescore_mf"),
        ("joint_mf", "random", "mf_vs_random"),
        ("joint_mf", "device_aware_sequential", "joint_vs_device_schedule"),
        ("joint_mf", "fixed_design_scheduler", "joint_vs_fixed_schedule"),
    ]:
        components[tag] = stratified(pivots["utility"], a, b)

    family = data.groupby(["workload", "method"], as_index=False).agg(
        utility=("utility", "mean"), cvar10=("utility_cvar10", "mean"),
        service_violation=("service_violation", "mean"),
        deadline_miss=("deadline_miss", "mean"), infeasible=("infeasible", "mean"),
        wall_s=("search_wall_s", "median"), evaluations=("search_evaluations", "mean"),
        unique_candidates=("unique_candidates", "mean"), peak_rss_mb=("peak_rss_mb", "max"))
    family.to_csv(RES / "clearaccept_confirmatory_summary.csv", index=False)

    seq_gain = (pivots["utility"].joint_mf - pivots["utility"].sequential).sort_values(ascending=False)
    influence = []
    for index, value in seq_gain.items():
        remainder = seq_gain.drop(index)
        influence.append({"workload": index[0], "seed": int(index[1]),
                          "gain": float(value), "mean_without_case": float(remainder.mean())})
    influence = pd.DataFrame(influence)
    influence.to_csv(RES / "clearaccept_influence.csv", index=False)
    trimmed = boot(seq_gain.sort_values().iloc[3:-3])

    # Deployment-frequency sensitivity: three declared mixtures.
    mixtures = {
        "equal": dict.fromkeys(FAMILIES, 0.20),
        "network_heavy": {"qaoa": .25, "qaoa_channel": .25, "qaoa_place": .25,
                          "vqc": .125, "vqc_cancer": .125},
        "learning_heavy": {"qaoa": .10, "qaoa_channel": .10, "qaoa_place": .10,
                           "vqc": .35, "vqc_cancer": .35},
    }
    mix_results = {}
    family_gain = seq_gain.groupby(level=0).mean()
    for name, weights in mixtures.items():
        mix_results[name] = float(sum(weights[f] * family_gain[f] for f in FAMILIES))

    # Pareto status over quality, service violation, and modeled cost.
    pareto_source = data.groupby("method", as_index=False).agg(
        quality_loss=("quality_loss", "mean"),
        service_violation=("service_violation", "mean"),
        money=("money", "mean"), energy_j=("energy_j", "mean"))
    dominated = []
    values = pareto_source[["quality_loss", "service_violation", "money", "energy_j"]].to_numpy()
    for i, row in enumerate(values):
        dominated.append(any(np.all(other <= row) and np.any(other < row)
                             for j, other in enumerate(values) if i != j))
    pareto_source["pareto"] = ~np.asarray(dominated)
    pareto_source.to_csv(RES / "clearaccept_pareto.csv", index=False)

    # Cost ledger with actual, not nominal, evaluation and wall-time totals.
    cost = data.groupby("method", as_index=False).agg(
        search_units=("search_cost_units", "mean"),
        modeled_search_s=("search_cost_modeled_s", "mean"),
        wall_s=("search_wall_s", "median"), evaluations=("search_evaluations", "mean"),
        unique_candidates=("unique_candidates", "mean"), peak_rss_mb=("peak_rss_mb", "max"))
    cost.to_csv(RES / "clearaccept_cost_ledger.csv", index=False)

    rows = []
    order = ["joint_mf", "capo_chance", "random", "capo_joint_only",
             "capo_mf_no_rescore", "sequential", "device_aware_sequential",
             "fixed_design_scheduler", "classical_only"]
    overall = data.groupby("method").agg(utility=("utility", "mean"),
                                          cvar=("utility_cvar10", "mean"),
                                          risk=("service_violation", "mean"))
    for method in order:
        row = overall.loc[method]
        rows.append(f"{method_names[method]} & {row.utility:.3f} & {row.cvar:.3f} & "
                    f"{100*row.risk:.1f}\\% " + ROW_END)
    emit_table("tab_clearaccept_confirmatory.tex", "lrrr",
               r"Method & Mean utility & CVaR$^{\mathrm{deploy}}_{0.1}$ & Violation " + ROW_END, rows)

    rows = []
    for method in order[:-1]:
        row = cost.set_index("method").loc[method]
        rows.append(f"{method_names[method]} & {row.search_units:.1f} & "
                    f"{row.evaluations:.1f} & {row.unique_candidates:.1f} & "
                    f"{row.wall_s:.2f} " + ROW_END)
    emit_table("tab_clearaccept_cost.tex", "lrrrr",
               "Method & Units & Evals & Unique & Wall s " + ROW_END, rows)

    rows = []
    for tag, value in components.items():
        rows.append(f"{tag.replace('_',' ')} & {value[0]:+.3f} & "
                    f"[{value[1]:+.3f},{value[2]:+.3f}] " + ROW_END)
    emit_table("tab_clearaccept_components.tex", "lrr",
               "Contrast & Mean gain & 95\\% CI " + ROW_END, rows)

    # Reliability/quality Pareto plot.
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    pareto_names = {
        "capo_chance": "CAPO-Chance", "capo_joint_only": "CAPO-Joint",
        "capo_mf_no_rescore": "CAPO-MF/no-rescore", "joint_mf": "CAPO-MF",
        "random": "CAPO-Random", "classical_only": "Classical",
        "fixed_design_scheduler": "Fixed-design", "sequential": "Sequential",
        "device_aware_sequential": "Device-aware+schedule",
    }
    for row in pareto_source.itertuples():
        ax.scatter(row.service_violation, row.quality_loss,
                   marker="o" if row.pareto else "x", s=22,
                   label=pareto_names[row.method])
    ax.set_xlabel("service-violation probability", fontsize=8)
    ax.set_ylabel("quality loss", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, lw=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=3,
              fontsize=6, frameon=False, columnspacing=0.7, handletextpad=0.3)
    fig.tight_layout(); fig.savefig(FIG / "fig_clearaccept_pareto.pdf", bbox_inches="tight")
    plt.close(fig)

    macros = []
    def macro(name, value): macros.append(f"\\newcommand{{\\{name}}}{{{value}}}")
    for key, prefix in [("mean_utility", "ClearMean"), ("cvar10", "ClearCvar"),
                        ("service_violation_reduction", "ClearRisk")]:
        value = primary[key]
        macro(prefix, f"{value[0]:+.3f}"); macro(prefix + "Lo", f"{value[1]:+.3f}")
        macro(prefix + "Hi", f"{value[2]:+.3f}")
    macro("ClearConfirmPairs", int(data.groupby(["workload", "seed"]).ngroups))
    macro("ClearTrimmedMean", f"{trimmed[0]:+.3f}")
    macro("ClearWorstFamily", f"{family_gain.min():+.3f}")
    macro("ClearNetworkMix", f"{mix_results['network_heavy']:+.3f}")
    macro("ClearLearningMix", f"{mix_results['learning_heavy']:+.3f}")
    for tag, prefix in [
        ("joint_vs_device_schedule", "ClearJointDevice"),
        ("joint_vs_fixed_schedule", "ClearJointFixed"),
    ]:
        value = components[tag]
        macro(prefix, f"{value[0]:+.3f}")
        macro(prefix + "Lo", f"{value[1]:+.3f}")
        macro(prefix + "Hi", f"{value[2]:+.3f}")
    (GEN / "clearaccept_numbers.tex").write_text("\n".join(macros) + "\n")

    report = {
        "primary": primary, "components": components,
        "trimmed_gain": trimmed, "family_gain": family_gain.to_dict(),
        "deployment_mixtures": mix_results,
        "top_influential": influence.head(10).to_dict("records"),
        "pareto": pareto_source.to_dict("records"),
        "cost": cost.to_dict("records"),
    }
    (RES / "clearaccept_analysis.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
