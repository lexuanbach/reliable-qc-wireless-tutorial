#!/usr/bin/env python3
"""Emit the fable.md revision assets: three-arm results, context
distributions, block reconciliation, and the full method-by-family matrix.
Reads stored CSVs only; runs no new searches except the arms CSV produced by
run_clear_accept_experiments.py --out clearaccept_newarms."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_clear_accept import boot, stratified, FAMILIES
from coopt import ContextModel

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
GEN = ROOT.parent / "submission" / "paper" / "generated"
FIG = ROOT.parent / "submission" / "paper" / "figures"
ROW = r"\\"

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "figure.dpi": 200,
                     "pdf.fonttype": 42, "font.family": "serif",
                     "font.serif": ["Times New Roman", "Times",
                                    "Nimbus Roman", "STIXGeneral"],
                     "mathtext.fontset": "stix"})

MACROS = []


def macro(name, value):
    MACROS.append(f"\\newcommand{{\\{name}}}{{{value}}}")


def table(name, spec, header, rows):
    (GEN / name).write_text("\n".join([
        f"\\begin{{tabular}}{{@{{}}{spec}@{{}}}}", r"\toprule", header,
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]) + "\n")


# ---------------------------------------------------------------- contexts
def context_distributions():
    """Percentiles of the deployment context marginals and per-QPU
    deadline-admissibility fractions, pooled over the confirmatory seeds."""
    rows = []
    for seed in range(100, 112):
        cm = ContextModel(seed=500 + seed)
        rng = np.random.default_rng(60_000 + seed)
        for w in range(9):
            for _ in range(100):
                c = cm.sample(rng, w)
                rows.append((c.deadline_s, c.rtt_s, c.queue_s["qpu_A"],
                             c.queue_s["qpu_B"], c.fail_prob))
    arr = np.asarray(rows)
    names = [(r"$T_{\max}$ (s)", 0), (r"$\tau$ (s)", 1),
             (r"$q_{\mathrm{qpu}_A}$ (s)", 2), (r"$q_{\mathrm{qpu}_B}$ (s)", 3),
             (r"$\phi$ (per round trip)", 4)]
    out = []
    for label, i in names:
        p10, p50, p90 = np.percentile(arr[:, i], [10, 50, 90])
        fmt = "{:.4f}" if i == 4 else "{:.2f}"
        out.append(f"{label} & {fmt.format(p10)} & {fmt.format(p50)} & "
                   f"{fmt.format(p90)} & {fmt.format(arr[:, i].mean())} {ROW}")
    table("tab_context_dist.tex", "lrrrr",
          r"Quantity & P10 & Median & P90 & Mean \\", out)
    deadline_p10, _, deadline_p90 = np.percentile(arr[:, 0], [10, 50, 90])
    macro("CtxDeadlinePten", f"{deadline_p10:.2f}")
    macro("CtxDeadlinePninety", f"{deadline_p90:.2f}")
    for b, col, nm in (("qpu_A", 2, "QpuA"), ("qpu_B", 3, "QpuB")):
        feas = float((arr[:, col] + arr[:, 1] <= arr[:, 0]).mean())
        macro(f"CtxFeas{nm}", f"{100*feas:.0f}\\%")
    macro("CtxDraws", f"{len(arr):,}".replace(",", "{,}"))


# ------------------------------------------------------------- method matrix
def method_family_matrix():
    df = pd.concat([pd.read_csv(p) for p in sorted(RES.glob("raw_main_*.csv"))],
                   ignore_index=True)
    df = df[(df.experiment == "exp1") & df.workload.isin(FAMILIES)]
    df = df.drop_duplicates(["workload", "seed", "method"], keep="last")
    label = {"joint_mf": r"\sys{}-MF", "joint_bo": r"\sys{}-SF",
             "random": r"\sys{}-Random", "feasible_random": r"\sys{}-Screened",
             "joint_rf": r"\sys{}-ET", "sequential": "Sequential",
             "accuracy_only": "Accuracy-only BO", "hand_designed": "Hand-designed",
             "always_simulator": "Edge sim.\\ only",
             "classical_only": "Classical only",
             "oracle_placement": "Hindsight placement$^{\\dagger}$"}
    order = ["joint_mf", "joint_bo", "random", "feasible_random", "joint_rf",
             "sequential", "accuracy_only", "hand_designed", "always_simulator",
             "classical_only", "oracle_placement"]
    pm = df.pivot_table(index="method", columns="workload", values="utility",
                        aggfunc="mean")
    fails = df.pivot_table(index="method", columns="workload",
                           values="deadline_miss", aggfunc="mean") + \
        df.pivot_table(index="method", columns="workload",
                       values="infeasible", aggfunc="mean")
    rows = []
    for m in order:
        if m not in pm.index:
            continue
        cells = [label[m]]
        for f in FAMILIES:
            cells.append(f"${pm.loc[m, f]:+.2f}$")
            cells.append(f"{100*fails.loc[m, f]:.0f}")
        pre = r"\midrule" + "\n" if m == "oracle_placement" else ""
        rows.append(pre + " & ".join(cells) + f" {ROW}")
    header = (r"Method & \multicolumn{2}{c}{Path} & \multicolumn{2}{c}{Channel}"
              r" & \multicolumn{2}{c}{Placement} & \multicolumn{2}{c}{VQC-synth.}"
              r" & \multicolumn{2}{c}{VQC-real} \\" "\n"
              r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}"
              r"\cmidrule(lr){8-9}\cmidrule(lr){10-11}" "\n"
              r" & Util. & Ev. & Util. & Ev. & Util. & Ev. & Util. & Ev. & "
              r"Util. & Ev. \\")
    table("tab_method_family.tex", "l" + "rr" * 5, header, rows)

    # hand-designed catastrophic decomposition (path selection family)
    hd = df[(df.method == "hand_designed") & (df.workload == "qaoa")]
    q = hd.quality_loss.mean(); dm = hd.deadline_miss.mean()
    inf = hd.infeasible.mean(); mny = 0.5 * hd.money.mean() / 0.01
    en = 0.3 * hd.energy_j.mean() / 50.0
    stale = -hd.utility.mean() - (q + 2 * dm + 5 * inf + mny + en)
    macro("HdU", f"{hd.utility.mean():+.2f}")
    macro("HdQ", f"{q:.2f}"); macro("HdDl", f"{2*dm:.2f}")
    macro("HdInf", f"{5*inf:.2f}"); macro("HdStale", f"{stale:.2f}")
    macro("HdMoney", f"{mny:.2f}"); macro("HdEnergy", f"{en:.2f}")
    return df


# ------------------------------------------------------------ reconciliation
def reconciliation(block_a):
    """Paired joint-vs-sequential differences in both seed blocks."""
    pa = block_a.pivot_table(index=["workload", "seed"], columns="method",
                             values="utility")
    da_bo = (pa["joint_bo"] - pa["sequential"]).dropna()
    da_mf = (pa["joint_mf"] - pa["sequential"]).dropna()
    cf = pd.read_csv(RES / "clearaccept_confirmatory.csv")
    pb = cf.pivot_table(index=["workload", "seed"], columns="method",
                        values="utility")
    db = (pb["joint_mf"] - pb["sequential"]).dropna()

    for d, nm in ((da_bo, "BlockABo"), (da_mf, "BlockAMf"), (db, "BlockB")):
        v = d.to_numpy()
        macro(f"{nm}Mean", f"{v.mean():+.2f}")
        macro(f"{nm}Median", f"{np.median(v):+.3f}")
        macro(f"{nm}TailCount", int((v > 2.0).sum()))
        if v.sum() > 0:
            macro(f"{nm}TailShare", f"{100*v[v > 2.0].sum()/v.sum():.0f}\\%")
        macro(f"{nm}N", len(v))
        macro(f"{nm}SmallShare", f"{100*float((np.abs(v) <= 0.25).mean()):.0f}\\%")
        top = np.sort(v)[-max(1, int(np.ceil(0.1 * len(v)))):]
        macro(f"{nm}TopShare",
              f"{100*top.sum()/v.sum():.0f}\\%" if v.sum() > 0 else "--")
    # sequential catastrophic rate per block (utility below -2)
    sa = block_a[block_a.method == "sequential"]
    sb = cf[cf.method == "sequential"]
    macro("SeqCatA", int((sa.utility < -2).sum()))
    macro("SeqCatB", int((sb.utility < -2).sum()))
    macro("SeqCatAN", len(sa)); macro("SeqCatBN", len(sb))

    # combined column figure: (a) engine bars per family, (b) block ECDFs
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(3.5, 2.0))
    engines = [("joint_mf", "CAPO-MF", "#4477aa", ""),
               ("random", "Random", "#ee9944", "//"),
               ("feasible_random", "Feas.-scr.", "#44aa55", "xx"),
               ("joint_rf", "Extra-trees", "#cc4433", "..")]
    fam_label = {"qaoa": "Path", "qaoa_channel": "Chan.", "qaoa_place": "Plc.",
                 "vqc": "V-syn", "vqc_cancer": "V-real"}
    fams = list(fam_label)
    fam_means = block_a.pivot_table(index="workload", columns="method",
                                    values="utility", aggfunc="mean")
    width = 0.2
    xs = np.arange(len(fams))
    for j, (m, lab, c, h) in enumerate(engines):
        axa.bar(xs + (j - 1.5) * width, [fam_means.loc[f, m] for f in fams],
                width, color=c, hatch=h, label=lab, edgecolor="black",
                linewidth=0.3)
    axa.set_xticks(xs)
    axa.set_xticklabels([fam_label[f] for f in fams], rotation=45,
                        ha="right", fontsize=6)
    axa.set_ylabel("deployment utility", fontsize=7)
    axa.tick_params(axis="y", labelsize=6)
    axa.legend(frameon=False, fontsize=5, loc="lower center",
               bbox_to_anchor=(0.42, 0.0), ncol=1,
               handlelength=1.2, handletextpad=0.4, labelspacing=0.3)
    axa.set_title("(a) proposal engines", fontsize=7)

    for d, lab, sty in ((da_bo, "seeds 0 to 11", "solid"),
                        (db, "seeds 100 to 111", "dashed")):
        v = np.sort(d.to_numpy())
        axb.step(v, np.arange(1, len(v) + 1) / len(v), where="post",
                 label=lab, lw=1.1, linestyle=sty)
    axb.set_xscale("symlog", linthresh=0.5)
    axb.set_xticks([-1, 0, 1, 10])
    axb.set_xticklabels(["$-1$", "$0$", "$1$", "$10$"], fontsize=6)
    axb.xaxis.set_minor_locator(plt.NullLocator())
    axb.tick_params(axis="y", labelsize=6)
    axb.axvline(0.0, color="gray", lw=0.6)
    axb.set_xlabel("joint $-$ sequential", fontsize=7)
    axb.set_ylabel("empirical CDF", fontsize=7)
    axb.legend(frameon=False, loc="lower right", fontsize=5.5,
               borderaxespad=0.4)
    axb.set_title("(b) seed blocks", fontsize=7)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(FIG / "fig_engines_reconcile.pdf", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ new arms
def new_arms():
    cf = pd.read_csv(RES / "clearaccept_confirmatory.csv")
    na = pd.read_csv(RES / "clearaccept_newarms.csv")
    sota = pd.read_csv(RES / "clearaccept_sota.csv")
    both = pd.concat([cf, na, sota], ignore_index=True)
    label = {"device_aware_sequential": "Device-aware+schedule",
             "device_aware_fallback": "~~+ classical escape hatch",
             "exhaustive_f0": "Exhaustive-$f_0$ + top-$k$",
             "capo_joint_only_espend": "\\sys{}-Joint, equal spend",
             "capo_joint_only": "\\sys{}-Joint (30 units)",
             "random": "\\sys{}-Random", "joint_mf": "\\sys{}-MF",
             "tpe": "TPE",
             "successive_halving": "Successive halving"}
    order = ["joint_mf", "random", "tpe", "successive_halving",
             "capo_joint_only", "capo_joint_only_espend",
             "exhaustive_f0", "device_aware_sequential", "device_aware_fallback"]
    g = both.groupby("method")
    rows = []
    for m in order:
        s = g.get_group(m)
        cls = float((s.cfg_placement == "classical").mean())
        rows.append(f"{label[m]} & {s.utility.mean():+.3f} & "
                    f"{s.utility_cvar10.mean():+.2f} & "
                    f"{100*s.service_violation.mean():.1f} & "
                    f"{s.search_cost_units.mean():.1f} & "
                    f"{100*cls:.0f} {ROW}")
    table("tab_newarms.tex", "lrrrrr",
          r"Method & Utility & CVaR$^{\mathrm{deploy}}_{0.1}$ & Viol.\ \% & "
          r"Spend & Class.\ \% \\", rows)

    piv = both.pivot_table(index=["workload", "seed"], columns="method",
                           values="utility")
    pv = both.pivot_table(index=["workload", "seed"], columns="method",
                          values="service_violation")
    pairs = [("tpe", "random", "TpeRand"),
             ("successive_halving", "random", "ShaRand"),
             ("successive_halving", "joint_mf", "ShaMf"),
             ("joint_mf", "device_aware_fallback", "MfDaf"),
             ("device_aware_fallback", "device_aware_sequential", "DafDas"),
             ("random", "exhaustive_f0", "RandExh"),
             ("random", "capo_joint_only_espend", "RandEspend"),
             ("capo_joint_only_espend", "capo_joint_only", "EspendJo")]
    for a, b, nm in pairs:
        m, lo, hi = stratified(piv, a, b)
        macro(f"Arm{nm}", f"{m:+.3f}")
        macro(f"Arm{nm}Lo", f"{lo:+.3f}"); macro(f"Arm{nm}Hi", f"{hi:+.3f}")
    m, lo, hi = stratified(pv, "device_aware_sequential", "device_aware_fallback")
    macro("ArmDafRiskPp", f"{100*m:.1f}")
    macro("ArmDafRiskPpLo", f"{100*lo:.1f}"); macro("ArmDafRiskPpHi", f"{100*hi:.1f}")
    macro("ArmExhSpend", f"{na[na.method=='exhaustive_f0'].search_cost_units.mean():.0f}")
    macro("ArmExhClassShare",
          f"{100*float((na[na.method=='exhaustive_f0'].cfg_placement=='classical').mean()):.0f}\\%")

    # pp-format harmonized confirmatory risk macros (B8/A8)
    pr = cf.pivot_table(index=["workload", "seed"], columns="method",
                        values="service_violation")
    for a, b, nm in (("sequential", "joint_mf", "ClearRiskPp"),
                     ("sequential", "capo_chance", "ChanceRiskPp")):
        m, lo, hi = stratified(pr, a, b)
        macro(nm, f"{100*m:.1f}")
        macro(f"{nm}Lo", f"{100*lo:.1f}"); macro(f"{nm}Hi", f"{100*hi:.1f}")


def main() -> int:
    GEN.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    context_distributions()
    block_a = method_family_matrix()
    reconciliation(block_a)
    new_arms()
    (GEN / "fable_revision_numbers.tex").write_text("\n".join(MACROS) + "\n")
    print(f"emitted {len(MACROS)} macros")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
