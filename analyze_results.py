#!/usr/bin/env python3
"""Statistical analysis and hypothesis verdicts for the co-optimization study.

Implements the protocol from plan section 10: paired comparisons on identical
evaluation streams, bootstrap confidence intervals, effect sizes, Holm
correction across optimizer comparisons, and explicit H1-H4 verdicts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"


MAIN_FAMILIES = ["qaoa", "qaoa_channel", "qaoa_place", "vqc", "vqc_cancer"]


def load() -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(RES.glob("raw_main_*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    # scale arms (qaoa_large/qaoa_xl) and exp10 are analyzed separately
    return df[(df.experiment != "exp1")
              | df.workload.isin(MAIN_FAMILIES)].reset_index(drop=True)


def boot_ci(x: np.ndarray, n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.mean(x)), float(np.percentile(means, 2.5)), \
        float(np.percentile(means, 97.5))


def paired(df: pd.DataFrame, method_a: str, method_b: str, col="utility"):
    """Per (workload, seed) paired difference a-b."""
    piv = df.pivot_table(index=["workload", "seed"], columns="method",
                         values=col)
    if method_a not in piv or method_b not in piv:
        return None
    d = (piv[method_a] - piv[method_b]).dropna().to_numpy()
    if len(d) < 2:
        return None
    mean, lo, hi = boot_ci(d)
    w = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided") \
        if np.any(d != 0) else None
    return {"n": len(d), "mean_diff": mean, "ci_lo": lo, "ci_hi": hi,
            "win_rate": float((d > 0).mean()),
            "p": float(w.pvalue) if w else 1.0,
            "effect_size": float(mean / (d.std(ddof=1) + 1e-12))}


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def main() -> int:
    df = load()
    report: dict = {}
    lines: list[str] = []

    def say(s=""):
        lines.append(s)
        print(s)

    # ---------------- Experiment 1: search efficiency (RQ1, H1) ------------
    e1 = df[df.experiment == "exp1"]
    say("== Experiment 1: deployment utility by method (fid3, held-out) ==")
    tab = e1.groupby(["workload", "method"])["utility"].agg(["mean", "std", "count"])
    say(tab.round(3).to_string())
    say()
    comps = {}
    for rival in ["sequential", "random", "accuracy_only", "hand_designed",
                  "always_simulator", "classical_only", "joint_mf"]:
        r = paired(e1, "joint_bo", rival)
        if r:
            comps[f"joint_bo_vs_{rival}"] = r
    adj = holm({k: v["p"] for k, v in comps.items()})
    say("== Paired comparisons: joint_bo minus rival (positive favors joint) ==")
    for k, v in comps.items():
        say(f"{k:35s} diff={v['mean_diff']:+.3f} CI=[{v['ci_lo']:+.3f},{v['ci_hi']:+.3f}] "
            f"win={v['win_rate']:.2f} p={v['p']:.4f} p_holm={adj[k]:.4f} d={v['effect_size']:+.2f}")
    report["exp1"] = {"comparisons": comps, "holm": adj}

    margin = 0.05  # preregistered practical margin on utility scale
    h1 = comps.get("joint_bo_vs_sequential")
    h1_pass = bool(h1 and h1["ci_lo"] > 0 and h1["mean_diff"] > margin)
    say()
    say(f"H1 (joint > sequential by margin {margin}): "
        f"{'SUPPORTED' if h1_pass else 'NOT SUPPORTED'}")
    report["H1"] = {"pass": h1_pass, "detail": h1}

    # placement mix chosen by each optimizer
    say()
    say("== Placement choices (share per method) ==")
    mix = e1.groupby(["workload", "method", "cfg_placement"]).size() \
        .groupby(level=[0, 1]).apply(lambda s: (s / s.sum()).round(2))
    say(mix.to_string())

    # ---------------- Experiment 2: objective ablation (RQ2) ---------------
    e2 = df[df.experiment == "exp2"]
    if len(e2):
        say()
        say("== Experiment 2: utility on FULL objective after searching with a term removed ==")
        piv = e2.pivot_table(index=["workload", "seed"], columns="method",
                             values="utility")
        base = piv.get("ablate_full")
        drops = {}
        for c in piv.columns:
            if c == "ablate_full":
                continue
            d = (base - piv[c]).dropna().to_numpy()
            if len(d) >= 2:
                m, lo, hi = boot_ci(d)
                drops[c] = {"utility_cost_of_omission": m, "ci": [lo, hi]}
                say(f"{c:20s} omission costs {m:+.3f} utility CI=[{lo:+.3f},{hi:+.3f}]")
        report["exp2"] = drops

    # ---------------- Experiment 3: drift robustness (RQ3, H2, H3) ---------
    e3 = df[df.experiment == "exp3"]
    if len(e3):
        say()
        say("== Experiment 3: regret vs hindsight oracle across windows ==")
        t = e3.groupby(["workload", "method"])["regret"].agg(["mean", "std"])
        say(t.round(3).to_string())
        # H2: single-snapshot degrades on later windows vs window 0
        ss = e3[e3.method == "single_snapshot"]
        w0 = ss[ss.window == 0].groupby(["workload", "seed"])["utility"].mean()
        wl = ss[ss.window >= 4].groupby(["workload", "seed"])["utility"].mean()
        d = (w0 - wl).dropna().to_numpy()
        m, lo, hi = boot_ci(d)
        h2_pass = bool(lo > 0)
        say(f"H2 single-snapshot degradation (w0 minus w>=4 utility): "
            f"{m:+.3f} CI=[{lo:+.3f},{hi:+.3f}] -> "
            f"{'SUPPORTED' if h2_pass else 'NOT SUPPORTED'}")
        report["H2"] = {"pass": h2_pass, "mean": m, "ci": [lo, hi]}
        # H3: robust variants cut worst-case (p10) utility loss on later windows
        late = e3[e3.window >= 4]
        pivr = late.pivot_table(index=["workload", "seed"], columns="method",
                                values="utility_p10")
        verdicts = {}
        for robust in ("multi_env", "cvar"):
            d = (pivr[robust] - pivr["single_snapshot"]).dropna().to_numpy()
            m, lo, hi = boot_ci(d)
            verdicts[robust] = {"mean": m, "ci": [lo, hi], "pass": bool(lo > 0)}
            say(f"H3 {robust:12s} tail-utility gain vs single-snapshot: "
                f"{m:+.3f} CI=[{lo:+.3f},{hi:+.3f}]")
        h3_pass = any(v["pass"] for v in verdicts.values())
        say(f"H3: {'SUPPORTED' if h3_pass else 'NOT SUPPORTED'}")
        report["H3"] = {"pass": h3_pass, "variants": verdicts}

    # ---------------- Experiment 4: transfer (RQ5) -------------------------
    e4 = df[df.experiment == "exp4"]
    if len(e4):
        say()
        say("== Experiment 4: backend/context transfer ==")
        t = e4.groupby(["workload", "method"])["utility"].agg(["mean", "std"])
        say(t.round(3).to_string())
        piv = e4.pivot_table(index=["workload", "seed"], columns="method",
                             values="utility")
        for a, b in [("warm_start_half_budget", "zero_shot"),
                     ("fresh_full_budget", "zero_shot"),
                     ("warm_start_half_budget", "fresh_full_budget")]:
            d = (piv[a] - piv[b]).dropna().to_numpy()
            m, lo, hi = boot_ci(d)
            say(f"{a} minus {b}: {m:+.3f} CI=[{lo:+.3f},{hi:+.3f}]")
        report["exp4"] = t["mean"].to_dict() if hasattr(t["mean"], "to_dict") else {}

    # ---------------- Experiment 5: amortization (RQ4, H4) -----------------
    say()
    say("== Experiment 5: amortization (from exp1 ledger) ==")
    h4_rows = []
    for (wl_, seed), grp in e1.groupby(["workload", "seed"]):
        g = grp.set_index("method")
        if "joint_bo" not in g.index or "hand_designed" not in g.index:
            continue
        c_search = float(g.loc["joint_bo", "search_cost_modeled_s"])
        # per-use advantage in scalarized cost-inclusive utility
        du = float(g.loc["joint_bo", "utility"] - g.loc["hand_designed", "utility"])
        n_be = c_search / du if du > 0 else np.inf
        h4_rows.append({"workload": wl_, "seed": seed, "c_search_s": c_search,
                        "du_per_use": du, "n_break_even": n_be})
    h4df = pd.DataFrame(h4_rows)
    say(h4df.round(3).to_string(index=False))
    finite = h4df[np.isfinite(h4df.n_break_even)]
    say(f"break-even exists in {len(finite)}/{len(h4df)} cases; "
        f"median N_be={finite.n_break_even.median():.1f}" if len(finite)
        else "no break-even anywhere")
    # H4: for low-reuse workloads search does not amortize -> check whether
    # median break-even exceeds a one-off reuse count of ~1-10 uses.
    h4_pass = bool(len(finite) == 0 or finite.n_break_even.median() > 10)
    say(f"H4 (low-reuse workloads do not amortize): "
        f"{'SUPPORTED' if h4_pass else 'NOT SUPPORTED'}")
    report["H4"] = {"pass": h4_pass,
                    "median_n_be": float(finite.n_break_even.median())
                    if len(finite) else None}

    # ---------------- Experiment 6: fidelity consistency -------------------
    e6 = df[df.experiment == "exp6"]
    if len(e6):
        say()
        say("== Experiment 6: rank correlation of cheap fidelities with deployment (fid3) ==")
        for wl_, grp in e6.groupby("workload"):
            for fid in (0, 1, 2):
                rho = stats.spearmanr(grp[f"u_fid{fid}"], grp["u_fid3"]).statistic
                say(f"{wl_} fid{fid} vs fid3: spearman rho={rho:+.3f}")
        report["exp6"] = "see text"

    # ---------------- weight sensitivity (plan section 6) ------------------
    say()
    say("== Scalarization sensitivity: joint_bo vs sequential under +/-50% weights ==")
    # re-scalarize from stored components
    for scale_tag, (wm, we) in {"money_x0.5_energy_x0.5": (0.5, 0.5),
                                "money_x1.5_energy_x1.5": (1.5, 1.5)}.items():
        u = -(e1["quality_loss"] + 2.0 * e1["deadline_miss"]
              + 5.0 * e1["infeasible"]
              + wm * 0.5 * e1["money"] / 0.01 + we * 0.3 * e1["energy_j"] / 50.0)
        tmp = e1[["workload", "seed", "method"]].copy()
        tmp["u"] = u
        piv = tmp.pivot_table(index=["workload", "seed"], columns="method", values="u")
        d = (piv["joint_bo"] - piv["sequential"]).dropna().to_numpy()
        m, lo, hi = boot_ci(d)
        say(f"{scale_tag}: diff={m:+.3f} CI=[{lo:+.3f},{hi:+.3f}]")

    (RES / "analysis_report.txt").write_text("\n".join(lines))
    (RES / "verdicts.json").write_text(json.dumps(
        {k: v for k, v in report.items() if k.startswith("H")}, indent=2,
        default=lambda o: None))
    say()
    say(f"report written to {RES / 'analysis_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
