#!/usr/bin/env python3
"""Emit every number, table, and figure the manuscript uses.

Reads results/raw_main_*.csv; writes submission/paper/generated/*.tex and
submission/paper/figures/*.pdf. The manuscript must not contain any measured
value that this script does not emit.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
GEN = ROOT.parent / "submission" / "paper" / "generated"
FIG = ROOT.parent / "submission" / "paper" / "figures"
GEN.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "figure.dpi": 200,
                     "pdf.fonttype": 42,
                     # match the IEEEtran body font (Times) incl. math
                     "font.family": "serif",
                     "font.serif": ["Times New Roman", "Times",
                                    "Nimbus Roman", "STIXGeneral"],
                     "mathtext.fontset": "stix"})

METHOD_LABEL = {
    "joint_mf": r"\sys{}-MF", "joint_bo": r"\sys{}-SF",
    "random": "Random (joint)", "sequential": "Sequential",
    "accuracy_only": "Accuracy-only BO", "hand_designed": "Hand-designed",
    "always_simulator": "Edge sim. only", "classical_only": "Classical only",
    "oracle_placement": "Hindsight placement",
    "feasible_random": "Feas.-screened random",
    "joint_rf": "Extra-trees joint",
}
ORDER = ["joint_mf", "joint_bo", "random", "sequential", "accuracy_only",
         "hand_designed", "always_simulator", "classical_only",
         "feasible_random", "joint_rf", "oracle_placement"]
MPL_LABEL = {k: v.replace(r"\sys{}", "CAPO") for k, v in METHOD_LABEL.items()}


def load():
    return pd.concat([pd.read_csv(p) for p in sorted(RES.glob("raw_main_*.csv"))],
                     ignore_index=True)


def boot_ci(x, n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return float(np.mean(x)), float(np.percentile(means, 2.5)), \
        float(np.percentile(means, 97.5))


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"={p:.3f}"


MAIN_FAMILIES = ["qaoa", "qaoa_channel", "qaoa_place", "vqc", "vqc_cancer"]


def main():
    df = load()
    e1_all = df[df.experiment == "exp1"]
    # scale arms (n=16/20) use different budgets and method subsets; they are
    # analyzed separately and excluded from the pooled headline statistics
    e1 = e1_all[e1_all.workload.isin(MAIN_FAMILIES)]
    macros = []

    def macro(name, value):
        macros.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    n_seeds = e1.groupby(["workload", "method"]).size().max()
    macro("NumSeeds", int(n_seeds))

    # ---- Table 1: exp1 deployment utility per workload -------------------
    piv_m = e1.pivot_table(index="method", columns="workload", values="utility",
                           aggfunc="mean")
    piv_s = e1.pivot_table(index="method", columns="workload", values="utility",
                           aggfunc="std")
    dm = e1.pivot_table(index="method", columns="workload",
                        values="deadline_miss", aggfunc="mean")
    infz = e1.pivot_table(index="method", columns="workload",
                          values="infeasible", aggfunc="mean")
    best = {w: piv_m[w].drop("oracle_placement").max() for w in piv_m.columns}
    lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"Method & \multicolumn{2}{c}{Combinatorial (QAOA)} & "
             r"\multicolumn{2}{c}{Learning (VQC)} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r" & Utility & Fails & Utility & Fails \\",
             r"\midrule"]
    for m in ORDER:
        if m not in piv_m.index:
            continue
        cells = [METHOD_LABEL[m]]
        for w in ("qaoa", "vqc"):
            u, s = piv_m.loc[m, w], piv_s.loc[m, w]
            val = f"${u:.2f}\\,{{\\scriptstyle\\pm {s:.2f}}}$"
            if m != "oracle_placement" and abs(u - best[w]) < 1e-9:
                val = r"\best{" + val + "}"
            fails = 100 * (dm.loc[m, w] + infz.loc[m, w])
            cells += [val, f"{fails:.0f}"]
        pre = r"\midrule" + "\n" + r"\rowcolor{gray!15}" + "\n" \
            if m == "oracle_placement" else ""
        lines.append(pre + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_exp1.tex").write_text("\n".join(lines) + "\n")

    # ---- paired comparisons macros --------------------------------------
    pivu = e1.pivot_table(index=["workload", "seed"], columns="method",
                          values="utility")
    comps = {}
    for rival in ["sequential", "random", "accuracy_only", "hand_designed",
                  "always_simulator", "classical_only"]:
        d = (pivu["joint_bo"] - pivu[rival]).dropna().to_numpy()
        m, lo, hi = boot_ci(d)
        w = stats.wilcoxon(d, zero_method="wilcox")
        comps[rival] = (m, lo, hi, float(w.pvalue), float((d > 0).mean()), len(d))
    # Holm over the six comparisons
    ps = sorted(((k, v[3]) for k, v in comps.items()), key=lambda t: t[1])
    holm, run = {}, 0.0
    for i, (k, p) in enumerate(ps):
        run = max(run, min(1.0, (len(ps) - i) * p))
        holm[k] = run
    names = {"sequential": "Seq", "random": "Rand", "accuracy_only": "AccOnly",
             "hand_designed": "Hand", "always_simulator": "Sim",
             "classical_only": "Classical"}
    for k, (m, lo, hi, p, wr, n) in comps.items():
        nm = names[k]
        macro(f"Diff{nm}", f"{m:+.2f}")
        macro(f"DiffLo{nm}", f"{lo:+.2f}")
        macro(f"DiffHi{nm}", f"{hi:+.2f}")
        macro(f"PHolm{nm}", fmt_p(holm[k]))
        macro(f"PRaw{nm}", fmt_p(p))
        macro(f"Win{nm}", f"{100*wr:.0f}\\%")
    macro("NPairs", comps["sequential"][5])
    # per-workload joint vs sequential
    for w in ("qaoa", "vqc"):
        sub = e1[e1.workload == w].pivot_table(index="seed", columns="method",
                                               values="utility")
        d = (sub["joint_bo"] - sub["sequential"]).dropna().to_numpy()
        m, lo, hi = boot_ci(d)
        macro(f"DiffSeq{w.upper()}", f"{m:+.2f}")
        macro(f"DiffSeqLo{w.upper()}", f"{lo:+.2f}")
        macro(f"DiffSeqHi{w.upper()}", f"{hi:+.2f}")

    # ---- Figure: utility by method (symlog) ------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.6), sharey=True)
    for ax, w, title in zip(axes, ("qaoa", "vqc"),
                            ("Combinatorial\n(QAOA)",
                             "Learning\n(VQC)")):
        sub = e1[e1.workload == w]
        data = [sub[sub.method == m]["utility"].to_numpy() for m in ORDER]
        bp = ax.boxplot(data, vert=False, showfliers=True, widths=0.6,
                        flierprops=dict(marker=".", markersize=2))
        ax.set_yticks(range(1, len(ORDER) + 1),
                      [MPL_LABEL[m] for m in ORDER], fontsize=6)
        ax.set_xscale("symlog", linthresh=0.5)
        ax.set_xlim(right=0.2)
        ax.set_xticks([-10, -1, 0])
        ax.set_xticklabels(["$-10$", "$-1$", "$0$"], fontsize=6)
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_title(title, fontsize=7)
        ax.axvline(0, lw=0.5, color="gray")
        ax.grid(axis="x", lw=0.3, alpha=0.5)
    axes[0].invert_yaxis()
    fig.tight_layout()
    y0 = min(ax.get_position().y0 for ax in axes)
    fig.text(0.55, y0 - 0.09, "deployment utility (held-out, fid3)",
             ha="center", fontsize=7)
    fig.savefig(FIG / "fig_exp1_utility.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---- Figure: placement mix ------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.3), sharey=True)
    placements = ["classical", "local_sim", "remote_sim", "qpu_A", "qpu_B"]
    colors = ["#888888", "#4477aa", "#66ccee", "#ee6677", "#aa3377"]
    sel_methods = ["joint_mf", "joint_bo", "random", "sequential",
                   "accuracy_only", "hand_designed"]
    for ax, w in zip(axes, ("qaoa", "vqc")):
        sub = e1[e1.workload == w]
        bottoms = np.zeros(len(sel_methods))
        for pl, c in zip(placements, colors):
            shares = [np.mean(sub[sub.method == m]["cfg_placement"] == pl)
                      for m in sel_methods]
            ax.bar(range(len(sel_methods)), shares, bottom=bottoms, color=c,
                   label=pl.replace("_", "-"), width=0.65)
            bottoms += np.array(shares)
        ax.set_title("QAOA" if w == "qaoa" else "VQC", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_xticks(range(len(sel_methods)))
        ax.set_xticklabels([MPL_LABEL[m] for m in sel_methods],
                           rotation=45, ha="right", fontsize=6)
    axes[0].set_ylabel("placement share")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.03), columnspacing=0.8,
               handletextpad=0.3, fontsize=6)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "fig_placement_mix.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---- Exp 2 ablation table -------------------------------------------
    e2 = df[df.experiment == "exp2"]
    piv = e2.pivot_table(index=["workload", "seed"], columns="method",
                         values="utility")
    rows = []
    label = {"ablate_no_latency": "latency (staleness+deadline)",
             "ablate_no_failure": "failure/infeasibility",
             "ablate_no_energy": "energy", "ablate_no_money": "monetary",
             "ablate_no_staleness": "staleness only"}
    for c in ["ablate_no_latency", "ablate_no_failure", "ablate_no_energy",
              "ablate_no_money", "ablate_no_staleness"]:
        d = (piv["ablate_full"] - piv[c]).dropna().to_numpy()
        m, lo, hi = boot_ci(d)
        star = r"$^{\ast}$" if lo > 0 else ""
        rows.append(f"{label[c]} & ${m:+.3f}$ & $[{lo:+.3f},{hi:+.3f}]${star} \\\\")
        key = c.replace("ablate_no_", "").capitalize()
        macro(f"Abl{key}", f"{m:+.3f}")
        macro(f"AblLo{key}", f"{lo:+.3f}")
        macro(f"AblHi{key}", f"{hi:+.3f}")
    (GEN / "tab_ablation.tex").write_text(
        "\n".join([r"\begin{tabular}{@{}lrr@{}}", r"\toprule",
                   r"Omitted objective term & Utility cost & 95\% CI \\",
                   r"\midrule"] + rows +
                  [r"\bottomrule", r"\end{tabular}"]) + "\n")

    # ---- Exp 3 drift macros + figure ------------------------------------
    e3 = df[df.experiment == "exp3"]
    for tag in ("single_snapshot", "multi_env", "cvar"):
        sub = e3[e3.method == tag]
        macro("Regret" + {"single_snapshot": "Single", "multi_env": "Multi",
                          "cvar": "Cvar"}[tag], f"{sub['regret'].mean():.3f}")
    late = e3[e3.window >= 4]
    pivr = late.pivot_table(index=["workload", "seed"], columns="method",
                            values="utility_p10")
    for tag, nm in (("multi_env", "Multi"), ("cvar", "Cvar")):
        d = (pivr[tag] - pivr["single_snapshot"]).dropna().to_numpy()
        m, lo, hi = boot_ci(d)
        macro(f"HThree{nm}", f"{m:+.3f}")
        macro(f"HThreeLo{nm}", f"{lo:+.3f}")
        macro(f"HThreeHi{nm}", f"{hi:+.3f}")
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9), sharex=True)
    for ax, w in zip(axes, ("qaoa", "vqc")):
        sub = e3[e3.workload == w]
        for tag, c, lab in (("single_snapshot", "#4477aa", "single-snapshot"),
                            ("multi_env", "#ee6677", "multi-window"),
                            ("cvar", "#228833", "CVaR$_{0.3}$")):
            g = sub[sub.method == tag].groupby("window")["regret"].mean()
            ax.plot(g.index, g.values, marker="o", ms=2, lw=1, color=c,
                    label=lab)
        ax.set_title("QAOA" if w == "qaoa" else "VQC", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_xticks([0, 2, 4, 6, 8])
        ax.grid(lw=0.3, alpha=0.5)
    axes[0].set_ylabel("regret vs.\nhindsight ref.", fontsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.22, 1, 1))
    fig.text(0.55, 0.15, "deployment window", ha="center", fontsize=7)
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.55, 0.0), columnspacing=0.8,
               handletextpad=0.3, fontsize=6)
    fig.savefig(FIG / "fig_drift.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---- Exp 4 transfer macros ------------------------------------------
    e4 = df[df.experiment == "exp4"]
    piv4 = e4.pivot_table(index=["workload", "seed"], columns="method",
                          values="utility")
    for a, b, nm in (("warm_start_half_budget", "zero_shot", "WarmZero"),
                     ("warm_start_half_budget", "fresh_full_budget", "WarmFresh"),
                     ("fresh_full_budget", "zero_shot", "FreshZero")):
        d = (piv4[a] - piv4[b]).dropna().to_numpy()
        m, lo, hi = boot_ci(d)
        macro(f"Tr{nm}", f"{m:+.3f}")
        macro(f"TrLo{nm}", f"{lo:+.3f}")
        macro(f"TrHi{nm}", f"{hi:+.3f}")

    # ---- Exp 5 amortization: macros + figure ----------------------------
    h4 = []
    for (w, seed), grp in e1.groupby(["workload", "seed"]):
        g = grp.set_index("method")
        if "joint_bo" not in g.index or "hand_designed" not in g.index:
            continue
        c = float(g.loc["joint_bo", "search_cost_modeled_s"])
        du = float(g.loc["joint_bo", "utility"] - g.loc["hand_designed", "utility"])
        h4.append({"workload": w, "n_be": c / du if du > 0 else np.inf,
                   "c": c, "du": du})
    h4 = pd.DataFrame(h4)
    for w, nm in (("qaoa", "Qaoa"), ("vqc", "Vqc")):
        sub = h4[(h4.workload == w) & np.isfinite(h4.n_be)]
        macro(f"NbeMedian{nm}", f"{sub.n_be.median():.0f}")
        macro(f"NbeMin{nm}", f"{sub.n_be.min():.0f}")
        macro(f"NbeMax{nm}", f"{sub.n_be.max():.0f}")
    macro("NbeExistPct", f"{100*np.isfinite(h4.n_be).mean():.0f}\\%")
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    Ns = np.logspace(0, 3.2, 80)
    for w, c, lab in (("qaoa", "#4477aa", "QAOA"), ("vqc", "#ee6677", "VQC")):
        sub = h4[h4.workload == w]
        gains = np.array([[n * r.du - r.c for n in Ns] for r in sub.itertuples()])
        med = np.median(gains, axis=0)
        lo_, hi_ = np.percentile(gains, [25, 75], axis=0)
        ax.plot(Ns, med, color=c, lw=1.2, label=lab)
        ax.fill_between(Ns, lo_, hi_, color=c, alpha=0.2, lw=0)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlabel("number of reuses $N$")
    ax.set_ylabel("cumulative net gain\n$N\\Delta - C_{search}$")
    ax.legend(frameon=False)
    ax.grid(lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig_amortization.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---- Exp 6 fidelity consistency -------------------------------------
    e6 = df[df.experiment == "exp6"]
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.6))
    for ax, w in zip(axes, ("qaoa", "vqc")):
        sub = e6[e6.workload == w]
        # f1 drawn last (highest zorder) with a distinct marker so it stays
        # visible where f1 and f2 utilities coincide
        styles = {0: dict(color="#cccccc", marker="o", s=5, zorder=1),
                  1: dict(facecolors="none", edgecolors="#ee6677",
                          marker="^", s=16, linewidths=0.7, zorder=3),
                  2: dict(color="#4477aa", marker="s", s=6, zorder=2)}
        handles = {}
        for fid in (0, 2, 1):
            rho = stats.spearmanr(sub[f"u_fid{fid}"], sub["u_fid3"]).statistic
            handles[fid] = ax.scatter(sub["u_fid3"], sub[f"u_fid{fid}"],
                                      label=f"$f_{fid}$ ($\\rho$={rho:.2f})",
                                      **styles[fid])
            word = {0: "Zero", 1: "One", 2: "Two"}[fid]
            macro(f"Rho{'Qaoa' if w=='qaoa' else 'Vqc'}Fid{word}", f"{rho:.2f}")
        lims = [min(sub["u_fid3"].min(), -3), 0.02]
        ax.plot(lims, lims, lw=0.5, color="gray")
        ax.set_xlim(right=0.05)
        ax.set_ylim(top=0.05)
        ax.set_xscale("symlog", linthresh=0.5)
        ax.set_yscale("symlog", linthresh=0.5)
        ax.set_ylabel("search-fidelity utility")
        ax.set_title("QAOA" if w == "qaoa" else "VQC", fontsize=8)
        ax.legend([handles[0], handles[1], handles[2]],
                  [handles[0].get_label(), handles[1].get_label(),
                   handles[2].get_label()],
                  frameon=False, loc="upper left", handletextpad=0.1)
    axes[1].set_xlabel("deployment utility ($f_3$)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_fidelity.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---- weight sensitivity macros --------------------------------------
    for tag, scale in (("Half", 0.5), ("OneHalf", 1.5)):
        u = -(e1["quality_loss"] + 2.0 * e1["deadline_miss"]
              + 5.0 * e1["infeasible"] + scale * 0.5 * e1["money"] / 0.01
              + scale * 0.3 * e1["energy_j"] / 50.0)
        tmp = e1[["workload", "seed", "method"]].copy()
        tmp["u"] = u
        piv = tmp.pivot_table(index=["workload", "seed"], columns="method",
                              values="u")
        d = (piv["joint_bo"] - piv["sequential"]).dropna().to_numpy()
        m, lo, hi = boot_ci(d)
        macro(f"Sens{tag}", f"{m:+.2f}")
        macro(f"SensLo{tag}", f"{lo:+.2f}")
        macro(f"SensHi{tag}", f"{hi:+.2f}")


    # ---- Table: generalization across workload families (exp1, new kinds) --
    fam_label = {"qaoa": "Path selection (QAOA, $n{=}12$)",
                 "qaoa_channel": "Channel assignment (QAOA, $n{=}12$)",
                 "qaoa_place": "Service placement (QAOA, $n{=}12$)",
                 "vqc": "Synthetic classification (VQC)",
                 "vqc_cancer": "Breast cancer (VQC, real data)"}
    fam_methods = ["joint_mf", "random", "sequential", "hand_designed"]
    lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"Workload family & \sys{}-MF & Random & Sequential & Hand-des. \\",
             r"\midrule"]
    for fam in ("qaoa", "qaoa_channel", "qaoa_place", "vqc", "vqc_cancer"):
        sub = e1[e1.workload == fam]
        if not len(sub):
            continue
        cells = [fam_label[fam]]
        vals = {m: sub[sub.method == m]["utility"] for m in fam_methods}
        top = max(v.mean() for v in vals.values())
        for m in fam_methods:
            u, sd = vals[m].mean(), vals[m].std()
            cell = f"${u:.2f}\\,{{\\scriptstyle\\pm {sd:.2f}}}$"
            if abs(u - top) < 1e-9:
                cell = r"\best{" + cell + "}"
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_families.tex").write_text("\n".join(lines) + "\n")
    # per-family joint_mf vs sequential and vs hand macros
    for fam, nm in (("qaoa_channel", "Chan"), ("qaoa_place", "Place"),
                    ("vqc_cancer", "Cancer")):
        sub = e1[e1.workload == fam]
        if not len(sub):
            continue
        piv = sub.pivot_table(index="seed", columns="method", values="utility")
        for rival, rnm in (("sequential", "Seq"), ("hand_designed", "Hand")):
            d = (piv["joint_mf"] - piv[rival]).dropna().to_numpy()
            m, lo, hi = boot_ci(d)
            macro(f"Fam{nm}{rnm}", f"{m:+.2f}")
            macro(f"Fam{nm}{rnm}Lo", f"{lo:+.2f}")
            macro(f"Fam{nm}{rnm}Hi", f"{hi:+.2f}")

    # ---- Figure + macros: budget sweep (exp7 + exp1 at budget 30) --------
    e7 = df[df.experiment == "exp7"]
    if len(e7):
        b30 = e1[e1.method.isin(["random", "joint_bo", "joint_mf"])
                 & e1.workload.isin(["qaoa", "vqc"])].copy()
        b30["budget"] = 30.0
        sweep = pd.concat([e7, b30[e7.columns.intersection(b30.columns)]],
                          ignore_index=True)
        fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.1), sharex=True)
        for ax, w in zip(axes, ("qaoa", "vqc")):
            sub = sweep[sweep.workload == w]
            for m, c in (("random", "#888888"), ("joint_bo", "#66ccee"),
                         ("joint_mf", "#4477aa")):
                g = sub[sub.method == m].groupby("budget")["utility"]
                mean, sem = g.mean(), g.sem()
                ax.errorbar(mean.index, mean.values, yerr=1.96 * sem.values,
                            marker="o", ms=3, lw=1, capsize=2, color=c,
                            label=MPL_LABEL[m])
            ax.set_xscale("log")
            ax.set_xticks([10, 30, 60], ["10", "30", "60"])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.set_xlim(8.5, 70)
            ax.set_title("QAOA" if w == "qaoa" else "VQC", fontsize=8)
            ax.set_ylabel("deployment utility")
            ax.grid(lw=0.3, alpha=0.5)
        axes[1].set_xlabel("search budget ($f_2$-equiv. units)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, -0.03), columnspacing=1.0,
                   handletextpad=0.4)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(FIG / "fig_budget.pdf", bbox_inches="tight")
        plt.close(fig)

        # ---- combined column figure: (a) budget sweep, (b) fidelity ----
        fig, axg = plt.subplots(2, 2, figsize=(3.5, 3.0))
        for ax, w in zip(axg[0], ("qaoa", "vqc")):
            sub = sweep[sweep.workload == w]
            for m, c in (("random", "#888888"), ("joint_bo", "#66ccee"),
                         ("joint_mf", "#4477aa")):
                g = sub[sub.method == m].groupby("budget")["utility"]
                ax.errorbar(g.mean().index, g.mean().values,
                            yerr=1.96 * g.sem().values, marker="o", ms=2,
                            lw=0.9, capsize=1.5, color=c, label=MPL_LABEL[m])
            ax.set_xscale("log")
            ax.set_xticks([10, 30, 60], ["10", "30", "60"])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.set_xlim(8.5, 70)
            ax.tick_params(labelsize=6)
            ax.set_title(("(a) budget, " if w == "qaoa" else "(a) budget, ")
                         + ("QAOA" if w == "qaoa" else "VQC"), fontsize=7)
            ax.set_xlabel("budget (units)", fontsize=6)
            ax.grid(lw=0.3, alpha=0.5)
        axg[0][0].set_ylabel("deployment utility", fontsize=7)
        axg[0][0].legend(frameon=False, fontsize=5, handlelength=1.2,
                         labelspacing=0.25, loc="lower right")
        for ax, w in zip(axg[1], ("qaoa", "vqc")):
            sub = e6[e6.workload == w]
            styles = {0: dict(color="#cccccc", marker="o", s=4, zorder=1),
                      1: dict(facecolors="none", edgecolors="#ee6677",
                              marker="^", s=12, linewidths=0.6, zorder=3),
                      2: dict(color="#4477aa", marker="s", s=5, zorder=2)}
            hnd = {}
            for fid in (0, 2, 1):
                rho = stats.spearmanr(sub[f"u_fid{fid}"], sub["u_fid3"]).statistic
                hnd[fid] = ax.scatter(sub["u_fid3"], sub[f"u_fid{fid}"],
                                      label=f"$f_{fid}$ ({rho:.2f})",
                                      **styles[fid])
            lims = [min(sub["u_fid3"].min(), -3), 0.02]
            ax.plot(lims, lims, lw=0.5, color="gray")
            ax.set_xlim(right=0.05); ax.set_ylim(top=0.05)
            ax.set_xscale("symlog", linthresh=0.5)
            ax.set_yscale("symlog", linthresh=0.5)
            ax.set_xticks([-10, -1, 0])
            ax.set_xticklabels(["$-10$", "$-1$", "$0$"])
            ax.set_yticks([-10, -1, 0])
            ax.set_yticklabels(["$-10$", "$-1$", "$0$"])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.tick_params(labelsize=6)
            ax.set_title("(b) fidelity, " + ("QAOA" if w == "qaoa" else "VQC"),
                         fontsize=7)
            ax.set_xlabel("deployment utility ($f_3$)", fontsize=6)
            ax.legend([hnd[0], hnd[1], hnd[2]],
                      [hnd[0].get_label(), hnd[1].get_label(),
                       hnd[2].get_label()], frameon=False, fontsize=5,
                      loc="upper left", handletextpad=0.1, labelspacing=0.25)
        axg[1][0].set_ylabel("search utility", fontsize=7)
        fig.tight_layout(h_pad=0.8, w_pad=0.8)
        fig.savefig(FIG / "fig_budget_fidelity.pdf", bbox_inches="tight")
        plt.close(fig)
        # macro: joint_mf vs random at low/high budget (pooled workloads)
        for b, nm in ((10.0, "Ten"), (60.0, "Sixty")):
            piv = e7[e7.budget == b].pivot_table(index=["workload", "seed"],
                                                 columns="method",
                                                 values="utility")
            d = (piv["joint_mf"] - piv["random"]).dropna().to_numpy()
            m, lo, hi = boot_ci(d)
            macro(f"BudgetMfRand{nm}", f"{m:+.2f}")
            macro(f"BudgetMfRandLo{nm}", f"{lo:+.2f}")
            macro(f"BudgetMfRandHi{nm}", f"{hi:+.2f}")

    # ---- macros: scaling (exp8) -----------------------------------------
    e8 = df[df.experiment == "exp8"]
    if len(e8):
        g = e8.groupby("size")["eval_wall_s"].mean()
        for size, nm in ((6, "Six"), (8, "Eight"), (9, "Nine"), (12, "Twelve")):
            if size in g.index:
                macro(f"EvalMs{nm}", f"{1e3*g.loc[size]:.0f}")
        # empirical exponent: fit log(time) ~ n
        import numpy.polynomial.polynomial as npp
        ns = e8["size"].to_numpy()
        ts = np.log2(e8["eval_wall_s"].to_numpy())
        slope = np.polyfit(ns, ts, 1)[0]
        macro("ScaleSlope", f"{slope:.2f}")

    # ---- macros: fidelity rho restricted to QPU placements --------------
    for w, wn in (("qaoa", "Qaoa"), ("vqc", "Vqc")):
        sub = e6[(e6.workload == w)
                 & e6.cfg_placement.isin(["qpu_A", "qpu_B"])]
        if len(sub) >= 8:
            for fid, word in ((1, "One"), (2, "Two")):
                rho = stats.spearmanr(sub[f"u_fid{fid}"], sub["u_fid3"]).statistic
                macro(f"RhoQpu{wn}Fid{word}", f"{rho:.2f}")
            macro(f"NQpu{wn}", len(sub))


    # ---- scale study (qaoa_large n=16, qaoa_xl n=20) ---------------------
    lg = e1_all[e1_all.workload == "qaoa_large"]
    xl = e1_all[e1_all.workload == "qaoa_xl"]
    if len(lg):
        piv = lg.pivot_table(index="seed", columns="method", values="utility")
        for a, b, nm in (("joint_mf", "random", "MfRand"),
                         ("joint_mf", "sequential", "MfSeq"),
                         ("joint_mf", "classical_only", "MfCls")):
            d = (piv[a] - piv[b]).dropna().to_numpy()
            m, lo, hi = boot_ci(d)
            macro(f"Scale{nm}", f"{m:+.2f}")
            macro(f"Scale{nm}Lo", f"{lo:+.2f}")
            macro(f"Scale{nm}Hi", f"{hi:+.2f}")
        for meth, nm in (("joint_mf", "Mf"), ("joint_bo", "Bo"),
                         ("random", "Rand"), ("sequential", "Seq"),
                         ("classical_only", "Cls"), ("hand_designed", "Hand")):
            sub = lg[lg.method == meth]["utility"]
            if len(sub):
                macro(f"LgU{nm}", f"{sub.mean():.2f}")
                macro(f"LgUsd{nm}", f"{sub.std():.2f}")
        macro("LgQlossCls", f"{lg[lg.method=='classical_only']['quality_loss'].mean():.2f}")
    if len(xl):
        for meth, nm in (("joint_mf", "Mf"), ("random", "Rand"),
                         ("classical_only", "Cls")):
            sub = xl[xl.method == meth]["utility"]
            if len(sub):
                macro(f"XlU{nm}", f"{sub.mean():.2f}")

    # ---- exp10: noise-model mismatch robustness --------------------------
    e10 = df[df.experiment == "exp10"]
    if len(e10):
        t = e10.pivot_table(index="method", columns="fidelity", values="utility")
        rho = stats.spearmanr(t[3], t[4]).statistic
        macro("MisSpearman", f"{rho:.2f}")
        shift = (t[4] - t[3]).abs()
        macro("MisMaxShift", f"{shift.max():.2f}")
        macro("MisJointShift", f"{shift.get('joint_mf', float('nan')):.3f}")
        piv = e10[e10.fidelity == 4].pivot_table(index=["workload", "seed"],
                                                 columns="method",
                                                 values="utility")
        d = (piv["joint_bo"] - piv["sequential"]).dropna().to_numpy()
        m, lo, hi = boot_ci(d)
        macro("MisSeqDiff", f"{m:+.2f}")
        macro("MisSeqDiffLo", f"{lo:+.2f}")
        macro("MisSeqDiffHi", f"{hi:+.2f}")

    (GEN / "numbers.tex").write_text("\n".join(macros) + "\n")
    print(f"wrote {len(macros)} macros, 5 figures, 2 tables")


if __name__ == "__main__":
    main()
