#!/usr/bin/env python3
"""Revision analyses: stratified inference, baselines, sensitivity, amortization."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Match the manuscript figures: IEEEtran body font (Times) incl. math.
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "pdf.fonttype": 42,
                     "font.family": "serif",
                     "font.serif": ["Times New Roman", "Times",
                                    "Nimbus Roman", "STIXGeneral"],
                     "mathtext.fontset": "stix"})
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
GEN = ROOT.parent / "submission" / "paper" / "generated"
FIG = ROOT.parent / "submission" / "paper" / "figures"
MAIN = ["qaoa", "qaoa_channel", "qaoa_place", "vqc", "vqc_cancer"]


def boot_ci(values, seed=73):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(20_000, len(values)))].mean(1)
    return float(values.mean()), *map(float, np.percentile(means, [2.5, 97.5]))


def load_exp1():
    df = pd.concat([pd.read_csv(path) for path in sorted(RES.glob("raw_main_*.csv"))],
                   ignore_index=True)
    df = df[df.experiment == "exp1"]
    return df.drop_duplicates(["workload", "seed", "method"], keep="last")


def stratified_bootstrap(paired: pd.DataFrame, a: str, b: str, seed=11):
    rng = np.random.default_rng(seed)
    families = list(paired.index.get_level_values(0).unique())
    family_values = []
    draws = []
    for family in families:
        sub = paired.xs(family)[[a, b]].dropna()
        values = (sub[a] - sub[b]).to_numpy()
        family_values.append(values.mean())
        draws.append(values)
    samples = np.empty(20_000)
    for i in range(len(samples)):
        samples[i] = np.mean([v[rng.integers(len(v), size=len(v))].mean()
                              for v in draws])
    return float(np.mean(family_values)), *map(float, np.percentile(samples, [2.5, 97.5]))


def main() -> int:
    GEN.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    all_exp1 = load_exp1()
    main = all_exp1[all_exp1.workload.isin(MAIN)].copy()
    piv = main.pivot_table(index=["workload", "seed"], columns="method",
                           values="utility")

    method_summary = main.groupby(["workload", "method"], as_index=False).agg(
        utility=("utility", "mean"), utility_sd=("utility", "std"),
        deadline_miss=("deadline_miss", "mean"), infeasible=("infeasible", "mean"))
    method_summary.to_csv(RES / "revision_method_summary.csv", index=False)

    comparisons = {}
    for rival in ("random", "feasible_random", "joint_rf", "sequential",
                  "classical_only", "always_simulator"):
        if rival not in piv or "joint_mf" not in piv:
            continue
        values = (piv.joint_mf - piv[rival]).dropna().to_numpy()
        comparisons[rival] = dict(zip(("mean", "lo", "hi"), boot_ci(values)))
        comparisons[rival]["win_rate"] = float((values > 0).mean())

    strat_mean, strat_lo, strat_hi = stratified_bootstrap(piv, "joint_bo", "sequential")
    leave_one_out = {}
    for omitted in MAIN:
        sub = piv[piv.index.get_level_values(0) != omitted]
        values = (sub.joint_bo - sub.sequential).dropna().to_numpy()
        leave_one_out[omitted] = dict(zip(("mean", "lo", "hi"), boot_ci(values)))
    seq_gain = (piv.joint_bo - piv.sequential).dropna().sort_values(ascending=False)
    positive_total = float(seq_gain.clip(lower=0).sum())
    top_n = max(1, int(np.ceil(0.10 * len(seq_gain))))
    top_share = float(seq_gain.iloc[:top_n].clip(lower=0).sum() / positive_total)

    # Comparator-aware amortization: incremental search cost and per-use gain.
    u = main.pivot_table(index=["workload", "seed"], columns="method", values="utility")
    c = main.pivot_table(index=["workload", "seed"], columns="method",
                         values="search_cost_modeled_s").fillna(0.0)
    amort = []
    for rival in ("hand_designed", "sequential", "random", "feasible_random",
                  "joint_rf", "classical_only", "always_simulator"):
        if rival not in u or rival not in c:
            continue
        common = u[["joint_bo", rival]].dropna().index.intersection(
            c[["joint_bo", rival]].dropna().index)
        for idx in common:
            du = float(u.loc[idx, "joint_bo"] - u.loc[idx, rival])
            dc = max(0.0, float(c.loc[idx, "joint_bo"] - c.loc[idx, rival]))
            nbe = dc / du if du > 0 else np.inf
            amort.append({"workload": idx[0], "seed": idx[1], "rival": rival,
                          "du": du, "incremental_search_s": dc, "n_be": nbe})
    amort = pd.DataFrame(amort)
    amort.to_csv(RES / "revision_amortization_all_baselines.csv", index=False)
    amort_summary = []
    for rival, group in amort.groupby("rival"):
        finite = group[np.isfinite(group.n_be)]
        amort_summary.append({
            "rival": rival, "cases": len(group),
            "finite_fraction": len(finite) / len(group),
            "median_n_be": float(finite.n_be.median()) if len(finite) else np.nan,
            "p90_n_be": float(finite.n_be.quantile(0.9)) if len(finite) else np.nan,
        })
    amort_summary = pd.DataFrame(amort_summary)
    amort_summary.to_csv(RES / "revision_amortization_summary.csv", index=False)

    # Sensitivity over the two terms that drive the headline comparison.
    deadline_weights = np.asarray([0.5, 1.0, 2.0, 4.0, 8.0])
    failure_weights = np.asarray([1.0, 2.5, 5.0, 7.5, 10.0])
    residual = -main.utility - 2.0 * main.deadline_miss - 5.0 * main.infeasible
    sens_source = main[["workload", "seed", "method", "deadline_miss",
                        "infeasible"]].copy()
    sens_source["residual"] = residual
    heat = np.empty((len(failure_weights), len(deadline_weights)))
    for i, wf in enumerate(failure_weights):
        for j, wd in enumerate(deadline_weights):
            tmp = sens_source.copy()
            tmp["rescored"] = -(tmp.residual + wd * tmp.deadline_miss
                                  + wf * tmp.infeasible)
            pp = tmp.pivot_table(index=["workload", "seed"], columns="method",
                                 values="rescored")
            heat[i, j] = (pp.joint_bo - pp.sequential).dropna().mean()
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    im = ax.imshow(heat, origin="lower", aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(deadline_weights)), deadline_weights)
    ax.set_yticks(range(len(failure_weights)), failure_weights)
    ax.set_xlabel("deadline-miss weight")
    ax.set_ylabel("failure/infeasibility weight")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("joint-SF minus sequential utility")
    fig.tight_layout()
    fig.savefig(FIG / "fig_revision_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)

    # New-baseline comparison by workload family.
    selected = ["joint_mf", "random", "feasible_random", "joint_rf"]
    labels = ["CAPO-MF", "Random", "Feas.-screened", "Extra-trees"]
    fam_means = method_summary[method_summary.method.isin(selected)].pivot(
        index="workload", columns="method", values="utility").reindex(MAIN)
    x = np.arange(len(MAIN)); width = 0.19
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    hatches = ["", "///", "xx", "..."]
    for j, (method, label) in enumerate(zip(selected, labels)):
        ax.bar(x + (j - 1.5) * width, fam_means[method], width, label=label,
               hatch=hatches[j], edgecolor="black", linewidth=0.45)
    ax.set_xticks(x, ["Path", "Channel", "Placement", "VQC-synth", "VQC-real"],
                  rotation=20, ha="right")
    ax.set_ylabel("deployment utility")
    ax.grid(axis="y", alpha=0.25)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, -0.04),
               columnspacing=0.8, handletextpad=0.3)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / "fig_revision_baselines.pdf", bbox_inches="tight")
    plt.close(fig)

    queue = pd.read_csv(RES / "raw_revision_queue_envelope.csv")
    queue_summary = queue.groupby("method", as_index=False).agg(
        utility=("utility", "mean"), deadline_miss=("deadline_miss", "mean"),
        infeasible=("infeasible", "mean"))
    queue_summary.to_csv(RES / "revision_queue_summary.csv", index=False)

    xl = all_exp1[all_exp1.workload == "qaoa_xl"]
    xl_summary = xl.groupby("method", as_index=False).agg(
        seeds=("seed", "nunique"), utility=("utility", "mean"),
        utility_sd=("utility", "std"))
    xl_summary.to_csv(RES / "revision_xl_summary.csv", index=False)

    # Compact LaTeX baseline table.
    lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"Family & \sys{}-MF & Random & Feas.-screen & Extra-trees \\",
             r"\midrule"]
    family_label = {"qaoa": "Path", "qaoa_channel": "Channel",
                    "qaoa_place": "Placement", "vqc": "VQC-synth.",
                    "vqc_cancer": "VQC-real"}
    for family in MAIN:
        row = fam_means.loc[family]
        lines.append(f"{family_label[family]} & {row.joint_mf:.3f} & {row.random:.3f} & "
                     f"{row.feasible_random:.3f} & {row.joint_rf:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_revision_baselines.tex").write_text("\n".join(lines) + "\n")

    # Emit macros used by the revision prose.
    macros = []
    def macro(name, value):
        macros.append(f"\\newcommand{{\\{name}}}{{{value}}}")
    macro("RevStratSeq", f"{strat_mean:+.2f}")
    macro("RevStratSeqLo", f"{strat_lo:+.2f}")
    macro("RevStratSeqHi", f"{strat_hi:+.2f}")
    macro("RevTopTailShare", f"{100*top_share:.0f}\\%")
    for rival, name in (("random", "Rand"), ("feasible_random", "Feas"),
                        ("joint_rf", "Rf")):
        item = comparisons[rival]
        macro(f"RevMf{name}", f"{item['mean']:+.2f}")
        macro(f"RevMf{name}Lo", f"{item['lo']:+.2f}")
        macro(f"RevMf{name}Hi", f"{item['hi']:+.2f}")
    for rival, name in (("hand_designed", "Hand"), ("random", "Rand"),
                        ("feasible_random", "Feas"), ("classical_only", "Class")):
        row = amort_summary.set_index("rival").loc[rival]
        macro(f"RevNbeFinite{name}", f"{100*row.finite_fraction:.0f}\\%")
        macro(f"RevNbeMedian{name}",
              "--" if pd.isna(row.median_n_be) else f"{row.median_n_be:.0f}")
    queue_idx = queue_summary.set_index("method")
    macro("RevQueueJointMiss", f"{100*queue_idx.loc['joint_mf','deadline_miss']:.1f}\\%")
    macro("RevQueueHandMiss", f"{100*queue_idx.loc['hand_designed','deadline_miss']:.1f}\\%")
    macro("RevXlSeeds", int(xl.seed.nunique()))
    (GEN / "revision_numbers.tex").write_text("\n".join(macros) + "\n")

    report = {
        "joint_mf_comparisons": comparisons,
        "stratified_joint_bo_vs_sequential": [strat_mean, strat_lo, strat_hi],
        "leave_one_family_out": leave_one_out,
        "top_10pct_positive_gain_share": top_share,
        "amortization": amort_summary.to_dict("records"),
        "queue_envelope": queue_summary.to_dict("records"),
        "n20": xl_summary.to_dict("records"),
    }
    (RES / "revision_analysis.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
