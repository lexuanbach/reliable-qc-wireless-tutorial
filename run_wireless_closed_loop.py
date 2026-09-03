#!/usr/bin/env python3
"""Dynamic dense-channel assignment with next-window network outcomes.

The experiment is a controlled IEEE 802.11-like interference simulation.  It
compares completed selection policies on the same four execution paths.  The
shallow-QAOA paths use exact statevectors with sampled outcomes.  Their queue
and latency values are modeled.  The experiment is not a radio testbed or a
physical-QPU measurement.
"""

from __future__ import annotations

import argparse
import itertools
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

from coopt import qaoa_probs
from run_reliability_selection import cp_bounds


ARTIFACT = Path(__file__).resolve().parent
RESULTS = ARTIFACT / "results"
FIGURES = ARTIFACT.parent / "submission" / "paper" / "figures"
GENERATED = ARTIFACT.parent / "submission" / "paper" / "generated"
N_AP = 8
ASSIGNMENTS = np.array(list(itertools.product((0, 1), repeat=N_AP)), dtype=np.int8)
PATHS = ("local_greedy", "local_exact", "qaoa_local", "qaoa_remote")
POLICIES = (
    "capo_cert",
    "expected_loss",
    "empirical_chance",
    "rolling",
    "local_greedy",
    "hindsight_oracle",
)


def topology(seed: int) -> np.ndarray:
    rng = np.random.default_rng(810_000 + seed)
    xy = rng.uniform(0, 1, size=(N_AP, 2))
    distance = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    gain = np.exp(-4.0 * distance)
    gain[distance > 0.62] = 0
    np.fill_diagonal(gain, 0)
    return gain


def assignment_costs(gain: np.ndarray, demand: np.ndarray,
                     previous: np.ndarray) -> np.ndarray:
    edge_weight = gain * np.sqrt(demand[:, None] * demand[None, :])
    same = ASSIGNMENTS[:, :, None] == ASSIGNMENTS[:, None, :]
    interference = 0.5 * np.sum(same * edge_weight[None, :, :], axis=(1, 2))
    switches = np.mean(ASSIGNMENTS != previous[None, :], axis=1)
    return interference + 0.12 * switches


def greedy_assignment(gain: np.ndarray, demand: np.ndarray,
                      previous: np.ndarray) -> np.ndarray:
    # One budgeted local move is a realistic fast fallback.  It can react to
    # the largest conflict without paying for a full combinatorial search.
    costs = assignment_costs(gain, demand, previous)
    candidates = [previous.copy()]
    for node in range(N_AP):
        trial = previous.copy()
        trial[node] = 1 - trial[node]
        candidates.append(trial)
    indices = [
        int(np.flatnonzero(np.all(ASSIGNMENTS == candidate, axis=1))[0])
        for candidate in candidates
    ]
    return ASSIGNMENTS[indices[int(np.argmin(costs[indices]))]].copy()


def qaoa_assignment(costs: np.ndarray, depth: int, seed: int) -> np.ndarray:
    normalized = (costs - costs.min()) / (costs.max() - costs.min() + 1e-12)
    if depth == 1:
        params = np.array([0.85, 0.62])
    else:
        params = np.array([0.85, 0.44, 0.62, 0.31])
    probabilities = qaoa_probs(normalized, N_AP, params)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(len(probabilities), size=256, p=probabilities)
    return ASSIGNMENTS[sampled[np.argmin(costs[sampled])]].copy()


def path_outcome(path: str, gain: np.ndarray, demand: np.ndarray,
                 previous: np.ndarray, seed: int, window: int,
                 deadline_s: float = 0.15) -> dict:
    costs = assignment_costs(gain, demand, previous)
    if path == "local_greedy":
        proposed = greedy_assignment(gain, demand, previous)
        delay_s = 0.008
    elif path == "local_exact":
        proposed = ASSIGNMENTS[int(np.argmin(costs))].copy()
        delay_s = 0.085
    elif path == "qaoa_local":
        proposed = qaoa_assignment(costs, 1, seed * 1000 + window)
        delay_s = 0.055
    elif path == "qaoa_remote":
        proposed = qaoa_assignment(costs, 2, seed * 1000 + window + 17)
        rng = np.random.default_rng(820_000_000 + seed * 1000 + window)
        queue = 0.025 if rng.random() > 0.18 else 0.34
        delay_s = 0.060 + queue
    else:
        raise ValueError(path)
    deadline_miss = delay_s > deadline_s
    fallback = greedy_assignment(gain, demand, previous)
    applied = fallback if deadline_miss else proposed
    index = np.flatnonzero(np.all(ASSIGNMENTS == applied, axis=1))[0]
    normalized_interference = float(costs[index] / (costs.max() + 1e-12))
    action_loss = float(min(1.0, normalized_interference + 0.35 * deadline_miss))
    return {
        "path": path,
        "proposed": proposed,
        "applied": applied,
        "delay_s": delay_s,
        "deadline_miss": int(deadline_miss),
        "fallback_used": int(deadline_miss),
        "action_loss": action_loss,
        "interference": normalized_interference,
    }


def confirmation(seed: int, gain: np.ndarray, draws: int) -> pd.DataFrame:
    rng = np.random.default_rng(830_000 + seed)
    previous = np.arange(N_AP) % 2
    rows = []
    for draw in range(draws):
        demand = rng.lognormal(mean=-0.05, sigma=0.35, size=N_AP)
        for path in PATHS:
            out = path_outcome(path, gain, demand, previous, seed + 10_000, draw)
            rows.append({
                "path": path,
                "draw": draw,
                "action_loss": out["action_loss"],
                "violation": out["deadline_miss"],
                "delay_s": out["delay_s"],
            })
    return pd.DataFrame(rows)


def initial_choices(panel: pd.DataFrame, eta: float, alpha: float) -> dict:
    summary = panel.groupby("path", as_index=False).agg(
        action_loss=("action_loss", "mean"),
        failures=("violation", "sum"),
        draws=("violation", "size"),
    )
    summary["risk"] = summary.failures / summary.draws
    summary["ucb"] = [
        cp_bounds(int(row.failures), int(row.draws), alpha / len(summary))[1]
        for row in summary.itertuples()
    ]
    certified = summary[summary.ucb <= eta]
    capo = "local_greedy" if certified.empty else certified.sort_values(
        ["action_loss", "ucb"]
    ).iloc[0].path
    empirical = summary[summary.risk <= eta]
    empirical_path = "local_greedy" if empirical.empty else empirical.sort_values(
        ["action_loss", "risk"]
    ).iloc[0].path
    return {
        "capo_cert": capo,
        "expected_loss": summary.sort_values("action_loss").iloc[0].path,
        "empirical_chance": empirical_path,
        "local_greedy": "local_greedy",
        "certificate_issued": not certified.empty,
    }


def transition(gain: np.ndarray, demand: np.ndarray, backlog: np.ndarray,
               assignment: np.ndarray) -> tuple[np.ndarray, float, float]:
    same = assignment[:, None] == assignment[None, :]
    interference = np.sum(gain * same, axis=1)
    capacity = 1.35 / (1.0 + 1.8 * interference)
    offered = demand + backlog
    served = np.minimum(offered, capacity)
    next_backlog = np.maximum(0.0, offered - served)
    dropped = np.maximum(0.0, next_backlog - 2.0)
    next_backlog = np.minimum(next_backlog, 2.0)
    return next_backlog, float(next_backlog.mean()), float(dropped.sum())


def one_policy(seed: int, policy: str, gain: np.ndarray, loads: np.ndarray,
               initial: dict, panel: pd.DataFrame, windows: int) -> dict:
    previous = np.arange(N_AP) % 2
    backlog = np.zeros(N_AP)
    history = panel.copy()
    totals = {
        "interference": [],
        "backlog": [],
        "dropped": [],
        "switches": [],
        "deadline_miss": [],
        "fallback": [],
        "action_loss": [],
    }
    selected_paths = []
    for window in range(windows):
        demand = loads[window] + 0.15 * backlog
        outcomes = {
            path: path_outcome(path, gain, demand, previous, seed, window)
            for path in PATHS
        }
        if policy in ("capo_cert", "expected_loss", "empirical_chance", "local_greedy"):
            path = initial[policy]
        elif policy == "rolling":
            recent = history[history.draw >= history.draw.max() - 4]
            path = recent.groupby("path").action_loss.mean().idxmin()
        elif policy == "hindsight_oracle":
            path = min(PATHS, key=lambda name: outcomes[name]["action_loss"])
        else:
            raise ValueError(policy)
        selected_paths.append(path)
        outcome = outcomes[path]
        applied = outcome["applied"]
        next_backlog, backlog_mean, dropped = transition(
            gain, loads[window], backlog, applied
        )
        totals["interference"].append(outcome["interference"])
        totals["backlog"].append(backlog_mean)
        totals["dropped"].append(dropped)
        totals["switches"].append(float(np.mean(applied != previous)))
        totals["deadline_miss"].append(outcome["deadline_miss"])
        totals["fallback"].append(outcome["fallback_used"])
        totals["action_loss"].append(outcome["action_loss"])
        for candidate, candidate_outcome in outcomes.items():
            history.loc[len(history)] = {
                "path": candidate,
                "draw": panel.draw.max() + 1 + window,
                "action_loss": candidate_outcome["action_loss"],
                "violation": candidate_outcome["deadline_miss"],
                "delay_s": candidate_outcome["delay_s"],
            }
        previous = applied
        backlog = next_backlog
    return {
        "seed": seed,
        "policy": policy,
        "windows": windows,
        "certificate_issued": bool(initial["certificate_issued"]) if policy == "capo_cert" else False,
        "dominant_selected_path": max(set(selected_paths), key=selected_paths.count),
        "mean_interference": float(np.mean(totals["interference"])),
        "mean_backlog": float(np.mean(totals["backlog"])),
        "dropped_demand": float(np.sum(totals["dropped"])),
        "switch_rate": float(np.mean(totals["switches"])),
        "deadline_miss_rate": float(np.mean(totals["deadline_miss"])),
        "fallback_rate": float(np.mean(totals["fallback"])),
        "action_loss": float(np.mean(totals["action_loss"])),
    }


def bootstrap_contrast(frame: pd.DataFrame, value: str, comparator: str,
                       seed: int = 20260903) -> tuple[float, float, float]:
    pivot = frame.pivot(index="seed", columns="policy", values=value)
    delta = (pivot[comparator] - pivot["capo_cert"]).to_numpy()
    rng = np.random.default_rng(seed)
    draws = delta[rng.integers(0, len(delta), size=(20_000, len(delta)))].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(delta.mean()), float(lo), float(hi)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=48)
    parser.add_argument("--windows", type=int, default=40)
    parser.add_argument("--confirmation-draws", type=int, default=40)
    parser.add_argument("--eta", type=float, default=0.20)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--replot", action="store_true",
                        help="reuse stored wireless_closed_loop.csv; redraw only")
    args = parser.parse_args()
    if args.replot and (RESULTS / "wireless_closed_loop.csv").is_file():
        data = pd.read_csv(RESULTS / "wireless_closed_loop.csv")
        rows = None
    else:
        rows = []
        for seed in range(args.seeds):
            gain = topology(seed)
            rng = np.random.default_rng(840_000 + seed)
            loads = np.empty((args.windows, N_AP))
            loads[0] = rng.lognormal(mean=-0.10, sigma=0.25, size=N_AP)
            for window in range(1, args.windows):
                innovation = rng.lognormal(mean=-0.10, sigma=0.30, size=N_AP)
                loads[window] = 0.72 * loads[window - 1] + 0.28 * innovation
            panel = confirmation(seed, gain, args.confirmation_draws)
            initial = initial_choices(panel, args.eta, args.alpha)
            for policy in POLICIES:
                rows.append(one_policy(seed, policy, gain, loads, initial, panel, args.windows))
            print(f"wireless seed={seed} selected={initial['capo_cert']}", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    GENERATED.mkdir(parents=True, exist_ok=True)
    if rows is not None:
        data = pd.DataFrame(rows)
        data.to_csv(RESULTS / "wireless_closed_loop.csv", index=False)
    summary = data.groupby("policy", as_index=False).agg(
        action_loss=("action_loss", "mean"),
        interference=("mean_interference", "mean"),
        backlog=("mean_backlog", "mean"),
        dropped=("dropped_demand", "mean"),
        switches=("switch_rate", "mean"),
        violations=("deadline_miss_rate", "mean"),
        fallback=("fallback_rate", "mean"),
    )
    summary.to_csv(RESULTS / "wireless_closed_loop_summary.csv", index=False)
    contrasts = {
        comparator: {
            metric: bootstrap_contrast(data, metric, comparator)
            for metric in ("action_loss", "mean_backlog", "dropped_demand")
        }
        for comparator in ("expected_loss", "empirical_chance", "rolling", "local_greedy")
    }

    labels = {
        "capo_cert": "CAPO-Cert",
        "expected_loss": "Expected-loss",
        "empirical_chance": "Empirical-chance",
        "rolling": "Rolling",
        "local_greedy": "Local greedy",
        "hindsight_oracle": "Hindsight oracle",
    }
    order = list(POLICIES)
    plot = summary.set_index("policy").loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.25))
    colors = ["#4477AA", "#CC6677", "#AA3377", "#228833", "#BBBBBB", "#EECC66"]
    axes[0].bar(range(len(order)), plot.backlog, color=colors)
    axes[0].set_ylabel("mean next-window backlog")
    axes[1].bar(range(len(order)), plot.dropped, color=colors)
    axes[1].set_ylabel("dropped demand per trajectory")
    for axis in axes:
        axis.set_xticks(range(len(order)), [labels[name] for name in order], rotation=34, ha="right")
        axis.grid(axis="y", alpha=0.25, linewidth=0.4)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_wireless_closed_loop.pdf", bbox_inches="tight")
    plt.close(fig)

    table_rows = []
    for policy in order:
        row = plot.loc[policy]
        table_rows.append(
            f"{labels[policy]} & {row.interference:.3f} & {row.backlog:.3f} & "
            f"{row.dropped:.2f} & {100*row.switches:.1f}\\% & "
            f"{100*row.violations:.1f}\\% \\\\"
        )
    table = [
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"Policy & Interference & Backlog & Dropped & Switches & Late \\",
        r"\midrule",
        *table_rows,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (GENERATED / "tab_wireless_closed_loop.tex").write_text("\n".join(table) + "\n")

    rolling = contrasts["rolling"]
    greedy = contrasts["local_greedy"]
    macros = [
        f"\\newcommand{{\\WirelessClusters}}{{{args.seeds}}}",
        f"\\newcommand{{\\WirelessWindows}}{{{args.windows}}}",
        f"\\newcommand{{\\WirelessDeploymentWindows}}{{{args.seeds * args.windows:,}}}",
        f"\\newcommand{{\\WirelessConfirmationDraws}}{{{args.confirmation_draws}}}",
        f"\\newcommand{{\\WirelessCertRate}}{{{100*data[data.policy == 'capo_cert'].certificate_issued.mean():.1f}\\%}}",
        f"\\newcommand{{\\WirelessBacklogVsRolling}}{{{rolling['mean_backlog'][0]:+.3f}}}",
        f"\\newcommand{{\\WirelessBacklogVsRollingLo}}{{{rolling['mean_backlog'][1]:+.3f}}}",
        f"\\newcommand{{\\WirelessBacklogVsRollingHi}}{{{rolling['mean_backlog'][2]:+.3f}}}",
        f"\\newcommand{{\\WirelessDroppedVsGreedy}}{{{greedy['dropped_demand'][0]:+.2f}}}",
        f"\\newcommand{{\\WirelessDroppedVsGreedyLo}}{{{greedy['dropped_demand'][1]:+.2f}}}",
        f"\\newcommand{{\\WirelessDroppedVsGreedyHi}}{{{greedy['dropped_demand'][2]:+.2f}}}",
    ]
    (GENERATED / "wireless_closed_loop_numbers.tex").write_text("\n".join(macros) + "\n")
    manifest = {
        "evidence": "controlled dynamic channel-assignment simulation with exact-statevector QAOA candidates",
        "not_evidence": "not a radio testbed and not physical-QPU execution",
        "access_points": N_AP,
        "channels": 2,
        "clusters": args.seeds,
        "deployment_windows_per_cluster": args.windows,
        "confirmation_draws": args.confirmation_draws,
        "eta": args.eta,
        "family_alpha": args.alpha,
        "candidate_paths": PATHS,
        "policies": POLICIES,
        "contrasts": contrasts,
    }
    (RESULTS / "wireless_closed_loop_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"summary": summary.to_dict("records"), "contrasts": contrasts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
