#!/usr/bin/env python3
"""Completed-policy reliability experiment for CAPO-Cert.

Every deployable policy sees the same six frozen finalists, confirmation
contexts, operational loss, abstention option, and local classical fallback.
CAPO-Cert deploys a nonfallback finalist only when a simultaneous one-sided
Clopper-Pearson upper bound meets the declared service-risk target.  The main
block uses independent draws from the confirmation distribution.  A separate
drift block changes the queue-burst distribution and compares a static
certificate with expiration plus fallback and rolling recertification.

All quantum outcomes are exact-statevector or controlled noisy simulations.
No row in this experiment is a physical-QPU measurement.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

from coopt import ContextModel, Workload, evaluate, sample_config, space_for


ARTIFACT = Path(__file__).resolve().parent
OUT = ARTIFACT / "results"
REGIMES = ("qpu_opportunity", "queue_stress")
POLICIES = (
    "capo_cert",
    "expected_loss",
    "empirical_chance",
    "deadline_threshold",
    "resource_plan",
    "balanced_scheduler",
    "classical_only",
)
PLACEMENTS = ("classical", "qpu_A", "qpu_B")
FALLBACK_CANDIDATE = -1
FALLBACK_QUALITY_LOSS = 0.30


@dataclass(frozen=True)
class Decision:
    selected_candidate: int | None
    applied_candidate: int
    certificate_issued: bool
    fallback_used: bool
    fallback_reason: str


def service_context(seed: int, draw: int, regime: str, phase: int,
                    shifted: bool = False, drift_strength: float | None = None):
    """Return a reproducible context from a declared queue mixture.

    The stable block uses the same mixture for confirmation and deployment.
    The drift block raises the burst probabilities only after confirmation.
    Independent phase identifiers prevent context reuse across data splits.
    """
    rng = np.random.default_rng(
        73_000_000 + phase * 1_000_000 + seed * 10_000 + draw
    )
    cm = ContextModel(seed=9_000 + seed)
    ctx = cm.sample(rng, draw % cm.n_windows)
    ctx.rtt_s = 0.020
    ctx.fail_prob = 0.001
    ctx.deadline_s = 0.35
    ctx.energy_budget_j = 500.0
    ctx.money_budget = 1.0
    ctx.volatility = 0.03
    strength = float(shifted) if drift_strength is None else drift_strength
    if strength < 0:
        raise ValueError("drift_strength must be nonnegative")
    if regime == "qpu_opportunity":
        stable_a, stable_b = 0.15, 0.02
        delta_a, delta_b = 0.10, 0.03
        ctx.eps["qpu_A"], ctx.eps["qpu_B"] = 0.006, 0.003
    elif regime == "queue_stress":
        stable_a, stable_b = 0.40, 0.25
        delta_a, delta_b = 0.15, 0.15
        ctx.eps["qpu_A"], ctx.eps["qpu_B"] = 0.009, 0.006
    else:
        raise ValueError(regime)
    burst_a = float(np.clip(stable_a + strength * delta_a, 0.0, 1.0))
    burst_b = float(np.clip(stable_b + strength * delta_b, 0.0, 1.0))
    ctx.queue_s["qpu_A"] = 1.20 if rng.random() < burst_a else 0.025
    ctx.queue_s["qpu_B"] = 0.90 if rng.random() < burst_b else 0.015
    return ctx


def cfg_key(cfg: dict) -> tuple:
    return tuple(sorted(cfg.items()))


def violation(comp: dict) -> int:
    return int(max(comp["deadline_miss"], comp["infeasible"],
                   comp["realized_fail"]))


def action_loss(comp: dict) -> float:
    """Operational loss in [0, 1], with unit loss on service violation."""
    return 1.0 if violation(comp) else float(np.clip(comp["quality_loss"], 0, 1))


def cp_bounds(failures: int, draws: int, alpha: float) -> tuple[float, float]:
    """Exact one-sided Clopper-Pearson bounds at tail probability alpha."""
    lo = 0.0 if failures == 0 else float(
        beta.ppf(alpha, failures, draws - failures + 1)
    )
    hi = 1.0 if failures == draws else float(
        beta.ppf(1 - alpha, failures + 1, draws - failures)
    )
    return lo, hi


def make_finalists(wl: Workload, seed: int, regime: str,
                   candidates: int = 72, per_placement: int = 2) -> list[dict]:
    """Analytically screen a joint pool and retain equal placement coverage."""
    rng = np.random.default_rng(71_000_000 + seed)
    space = space_for("qaoa_large")
    pool: dict[tuple, dict] = {}
    while len(pool) < candidates:
        cfg = sample_config(space, rng)
        cfg["warm_start"] = 1
        cfg["opt_iters"] = 20
        cfg["placement"] = PLACEMENTS[rng.integers(len(PLACEMENTS))]
        pool[cfg_key(cfg)] = cfg
    scored = []
    for index, cfg in enumerate(pool.values()):
        losses = []
        for draw in range(6):
            ctx = service_context(seed, draw, regime, phase=0)
            comp = evaluate(
                wl,
                cfg,
                ctx,
                0,
                np.random.default_rng(72_000_000 + seed * 1000 + index * 10 + draw),
            )
            losses.append(action_loss(comp))
        scored.append((cfg, float(np.mean(losses))))
    finalists = []
    for placement in PLACEMENTS:
        group = sorted(
            (row for row in scored if row[0]["placement"] == placement),
            key=lambda row: row[1],
        )
        finalists.extend(cfg for cfg, _ in group[:per_placement])
    assert len(finalists) == len(PLACEMENTS) * per_placement
    return finalists


def evaluate_panel(wl: Workload, finalists: list[dict], seed: int, regime: str,
                   draws: int, phase: int, shifted: bool = False
                   ) -> tuple[pd.DataFrame, list[dict]]:
    """Evaluate every finalist on common contexts at deployment fidelity."""
    rows = []
    raw = []
    for candidate, cfg in enumerate(finalists):
        comps = []
        for draw in range(draws):
            ctx = service_context(seed, draw, regime, phase, shifted=shifted)
            rng = np.random.default_rng(
                74_000_000 + phase * 1_000_000 + seed * 10_000
                + candidate * draws + draw
            )
            comp = evaluate(wl, cfg, ctx, 3, rng)
            comps.append(comp)
            raw.append({
                "seed": seed,
                "regime": regime,
                "phase": phase,
                "shifted": shifted,
                "candidate": candidate,
                "draw": draw,
                "placement": cfg["placement"],
                "action_loss": action_loss(comp),
                "service_violation": violation(comp),
                "quality_loss": comp["quality_loss"],
                "delay_s": comp["delay_s"],
                "deadline_s": ctx.deadline_s,
            })
        failures = sum(violation(c) for c in comps)
        rows.append({
            "candidate": candidate,
            "placement": cfg["placement"],
            "mean_action_loss": float(np.mean([action_loss(c) for c in comps])),
            "mean_quality_loss": float(np.mean([c["quality_loss"] for c in comps])),
            "mean_delay_s": float(np.mean([c["delay_s"] for c in comps])),
            "p90_delay_s": float(np.quantile([c["delay_s"] for c in comps], 0.90)),
            "risk": failures / draws,
            "failures": failures,
            "draws": draws,
        })
    # The fallback is an already validated action, not another searched
    # circuit.  It is immediately feasible and timely, with a declared
    # quality penalty for reusing a conservative route.
    rows.append({
        "candidate": FALLBACK_CANDIDATE,
        "placement": "validated_local_fallback",
        "mean_action_loss": FALLBACK_QUALITY_LOSS,
        "mean_quality_loss": FALLBACK_QUALITY_LOSS,
        "mean_delay_s": 0.002,
        "p90_delay_s": 0.002,
        "risk": 0.0,
        "failures": 0,
        "draws": draws,
    })
    for draw in range(draws):
        raw.append({
            "seed": seed,
            "regime": regime,
            "phase": phase,
            "shifted": shifted,
            "candidate": FALLBACK_CANDIDATE,
            "draw": draw,
            "placement": "validated_local_fallback",
            "action_loss": FALLBACK_QUALITY_LOSS,
            "service_violation": 0,
            "quality_loss": FALLBACK_QUALITY_LOSS,
            "delay_s": 0.002,
            "deadline_s": 0.35,
        })
    return pd.DataFrame(rows), raw


def enrich_panel(panel: pd.DataFrame, family_alpha: float) -> pd.DataFrame:
    p = panel.copy()
    k = len(p)
    bounds = [
        cp_bounds(int(r.failures), int(r.draws), family_alpha / k)
        for r in p.itertuples()
    ]
    p["risk_lcb"] = [x[0] for x in bounds]
    p["risk_ucb"] = [x[1] for x in bounds]
    return p


def decision(candidate: int | None, fallback: int, issued: bool,
             reason: str = "none") -> Decision:
    if candidate is None:
        return Decision(None, fallback, False, True, reason)
    return Decision(candidate, candidate, issued, candidate == fallback,
                    "selected_classical_fallback" if candidate == fallback else reason)


def select_policies(panel: pd.DataFrame, sla: float, family_alpha: float,
                    fallback_loss: float = FALLBACK_QUALITY_LOSS
                    ) -> tuple[dict[str, Decision], pd.DataFrame]:
    """Apply matched selectors with one common classical fallback."""
    p = enrich_panel(panel[panel.candidate >= 0], family_alpha)
    fallback = FALLBACK_CANDIDATE

    certified = p[p.risk_ucb <= sla]
    capo_row = None if certified.empty else certified.sort_values(
        ["mean_action_loss", "risk_ucb"]
    ).iloc[0]
    capo = None if capo_row is None or (
        capo_row.mean_action_loss >= fallback_loss
    ) else int(capo_row.candidate)

    expected_row = p.sort_values(["mean_action_loss", "risk"]).iloc[0]
    expected = None if expected_row.mean_action_loss >= fallback_loss else int(
        expected_row.candidate
    )

    empirical_pool = p[p.risk <= sla]
    empirical_row = None if empirical_pool.empty else empirical_pool.sort_values(
        ["mean_action_loss", "risk"]
    ).iloc[0]
    empirical = None if empirical_row is None or (
        empirical_row.mean_action_loss >= fallback_loss
    ) else int(empirical_row.candidate)

    timely = p[p.p90_delay_s <= 0.35]
    deadline_row = None if timely.empty else timely.sort_values(
        ["mean_quality_loss", "risk"]
    ).iloc[0]
    deadline = None if deadline_row is None or (
        deadline_row.mean_action_loss >= fallback_loss
    ) else int(deadline_row.candidate)

    # A Qonductor-inspired resource plan.  This is a transparent policy
    # abstraction, not a reimplementation of the published system.
    resource_score = (
        p.mean_quality_loss
        + 0.35 * np.minimum(p.p90_delay_s / 0.35, 4.0)
        + 1.5 * p.risk
    )
    resource_row = p.assign(score=resource_score).sort_values("score").iloc[0]
    resource = None if resource_row.mean_action_loss >= fallback_loss else int(
        resource_row.candidate
    )

    # A QOS-inspired balanced scheduler.  It preserves candidates within a
    # small quality tolerance, then minimizes risk and upper-tail latency.
    quality_band = p[p.mean_quality_loss <= p.mean_quality_loss.min() + 0.02]
    balanced_score = quality_band.risk + 0.20 * np.minimum(
        quality_band.p90_delay_s / 0.35, 4.0
    )
    balanced_row = quality_band.assign(score=balanced_score).sort_values("score").iloc[0]
    balanced = None if balanced_row.mean_action_loss >= fallback_loss else int(
        balanced_row.candidate
    )

    choices = {
        "capo_cert": decision(capo, fallback, capo is not None,
                              "no_candidate_met_simultaneous_bound"),
        "expected_loss": decision(expected, fallback, False),
        "empirical_chance": decision(
            empirical,
            fallback,
            False,
            "no_candidate_met_empirical_risk_target",
        ),
        "deadline_threshold": decision(
            deadline,
            fallback,
            False,
            "no_candidate_met_p90_deadline",
        ),
        "resource_plan": decision(
            resource, fallback, False, "candidate_loss_exceeded_fallback"
        ),
        "balanced_scheduler": decision(
            balanced, fallback, False, "candidate_loss_exceeded_fallback"
        ),
        "classical_only": decision(None, fallback, False, "local_policy_only"),
    }
    return choices, p


def append_policy_rows(rows: list[dict], deployment: pd.DataFrame,
                       confirmation: pd.DataFrame, decisions: dict[str, Decision],
                       seed: int, regime: str, experiment: str,
                       certificate_expiry: str, sla: float) -> None:
    oracle_row = deployment.sort_values("mean_action_loss").iloc[0]
    for policy, chosen in decisions.items():
        applied = deployment.loc[
            deployment.candidate == chosen.applied_candidate
        ].iloc[0]
        selected = None
        if chosen.selected_candidate is not None:
            selected = confirmation.loc[
                confirmation.candidate == chosen.selected_candidate
            ].iloc[0]
        rows.append({
            "workload": "qaoa_large",
            "seed": seed,
            "regime": regime,
            "experiment": experiment,
            "policy": policy,
            "selected_candidate": chosen.selected_candidate,
            "applied_candidate": chosen.applied_candidate,
            "applied_action": applied.placement,
            "placement": applied.placement,
            "certificate_issued": chosen.certificate_issued,
            "fallback_used": chosen.fallback_used,
            "fallback_reason": chosen.fallback_reason,
            "certificate_expiry": certificate_expiry,
            "selection_risk": np.nan if selected is None else selected.risk,
            "selection_risk_lcb": np.nan if selected is None else selected.risk_lcb,
            "selection_risk_ucb": np.nan if selected is None else selected.risk_ucb,
            "false_qualification": bool(
                chosen.certificate_issued and applied.risk > sla
            ),
            "action_loss": applied.mean_action_loss,
            "quality_loss": applied.mean_quality_loss,
            "service_violation": applied.risk,
            "delay_s": applied.mean_delay_s,
            "oracle_candidate": int(oracle_row.candidate),
            "oracle_placement": oracle_row.placement,
            "oracle_action_loss": oracle_row.mean_action_loss,
            "regret": applied.mean_action_loss - oracle_row.mean_action_loss,
            "qpu_selected": applied.placement in ("qpu_A", "qpu_B"),
            "qpu_beneficial": oracle_row.placement in ("qpu_A", "qpu_B"),
        })


def drift_decisions(static: dict[str, Decision], rolling_panel: pd.DataFrame,
                    sla: float, alpha: float) -> dict[str, Decision]:
    """Construct deployable drift policies from the common fallback action."""
    fallback = static["classical_only"].applied_candidate
    rolling, _ = select_policies(rolling_panel, sla, alpha)
    capo_static = static["capo_cert"]
    return {
        "capo_static": Decision(
            capo_static.selected_candidate,
            capo_static.applied_candidate,
            capo_static.certificate_issued,
            capo_static.fallback_used,
            capo_static.fallback_reason,
        ),
        "capo_expire_fallback": Decision(
            None, fallback, False, True, "detected_queue_regime_shift"
        ),
        "capo_rolling_recert": rolling["capo_cert"],
        "empirical_rolling": rolling["empirical_chance"],
        "classical_only": rolling["classical_only"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=24)
    parser.add_argument("--seed-offset", type=int, default=120)
    parser.add_argument("--selection-draws", type=int, default=120)
    parser.add_argument("--deployment-draws", type=int, default=100)
    parser.add_argument("--rolling-draws", type=int, default=120)
    parser.add_argument("--sla", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--regimes", default=",".join(REGIMES))
    parser.add_argument("--out", default="reliability_selection")
    args = parser.parse_args()

    started = time.time()
    selection_rows: list[dict] = []
    deployment_rows: list[dict] = []
    policy_rows: list[dict] = []
    drift_rows: list[dict] = []
    for regime in args.regimes.split(","):
        for seed in range(args.seed_offset, args.seed_offset + args.seeds):
            wl = Workload.make("qaoa_large", seed)
            finalists = make_finalists(wl, seed, regime)
            confirmation, raw_confirmation = evaluate_panel(
                wl, finalists, seed, regime, args.selection_draws, phase=1
            )
            choices, bounded_confirmation = select_policies(
                confirmation, args.sla, args.alpha
            )
            stable, raw_stable = evaluate_panel(
                wl, finalists, seed, regime, args.deployment_draws, phase=2
            )
            selection_rows.extend(raw_confirmation)
            deployment_rows.extend(raw_stable)
            append_policy_rows(
                policy_rows,
                stable,
                bounded_confirmation,
                choices,
                seed,
                regime,
                "stable_independent_deployment",
                "valid_for_declared_stable_block",
                args.sla,
            )

            shifted_confirmation, _ = evaluate_panel(
                wl,
                finalists,
                seed,
                regime,
                args.rolling_draws,
                phase=3,
                shifted=True,
            )
            shifted_deployment, _ = evaluate_panel(
                wl,
                finalists,
                seed,
                regime,
                args.deployment_draws,
                phase=4,
                shifted=True,
            )
            shifted_choices, shifted_bounded = select_policies(
                shifted_confirmation, args.sla, args.alpha
            )
            drift = drift_decisions(choices, shifted_confirmation, args.sla, args.alpha)
            append_policy_rows(
                drift_rows,
                shifted_deployment,
                shifted_bounded,
                drift,
                seed,
                regime,
                "shifted_queue_deployment",
                "expired_at_detected_queue_regime_change",
                args.sla,
            )
            print(
                f"{regime} seed={seed} stable="
                f"{choices['capo_cert'].applied_candidate} drift="
                f"{shifted_choices['capo_cert'].applied_candidate}",
                flush=True,
            )

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(policy_rows).to_csv(OUT / f"{args.out}.csv", index=False)
    pd.DataFrame(drift_rows).to_csv(OUT / f"{args.out}_drift.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(
        OUT / f"{args.out}_selection_draws.csv", index=False
    )
    pd.DataFrame(deployment_rows).to_csv(
        OUT / f"{args.out}_deployment_draws.csv", index=False
    )
    manifest = {
        "created_unix": time.time(),
        "elapsed_s": time.time() - started,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "seeds_per_regime": args.seeds,
        "seed_offset": args.seed_offset,
        "selection_draws": args.selection_draws,
        "deployment_draws": args.deployment_draws,
        "rolling_draws": args.rolling_draws,
        "sla": args.sla,
        "family_alpha": args.alpha,
        "finalists": 6,
        "simultaneous_method": "Bonferroni one-sided Clopper-Pearson",
        "candidate_generation": "72 analytic candidates, top two per placement",
        "placements": PLACEMENTS,
        "fallback": {
            "candidate_id": FALLBACK_CANDIDATE,
            "label": "validated_local_fallback",
            "quality_loss": FALLBACK_QUALITY_LOSS,
            "delay_s": 0.002,
            "service_violation": 0.0,
        },
        "stable_burst_probability": {
            "qpu_opportunity": {"qpu_A": 0.15, "qpu_B": 0.02},
            "queue_stress": {"qpu_A": 0.40, "qpu_B": 0.25},
        },
        "shifted_burst_probability": {
            "qpu_opportunity": {"qpu_A": 0.25, "qpu_B": 0.05},
            "queue_stress": {"qpu_A": 0.55, "qpu_B": 0.40},
        },
        "primary_loss": "1 on service violation, otherwise normalized quality loss",
        "matched_rights": "same finalists, records, actions, abstention, and classical fallback",
        "evidence": "exact statevector plus controlled service model, no QPU measurement",
    }
    (OUT / f"{args.out}_manifest.json").write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
