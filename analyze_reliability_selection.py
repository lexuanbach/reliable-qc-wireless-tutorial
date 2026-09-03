#!/usr/bin/env python3
"""Clustered analysis and LaTeX assets for completed CAPO-Cert policies."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from run_reliability_selection import select_policies


ARTIFACT = Path(__file__).resolve().parent
RES = ARTIFACT / "results"
GEN = ARTIFACT.parent / "submission" / "paper" / "generated"
ROW_END = r"\\"
POLICY_LABEL = {
    "capo_cert": "CAPO-Cert",
    "expected_loss": "Expected-loss",
    "empirical_chance": "Empirical-chance",
    "deadline_threshold": "Deadline-threshold",
    "resource_plan": "Fidelity-runtime plan",
    "balanced_scheduler": "Balanced scheduler",
    "classical_only": "Classical-only",
    "capo_static": "Static certificate",
    "capo_expire_fallback": "Expire + fallback",
    "capo_rolling_recert": "Rolling recertification",
    "empirical_rolling": "Rolling empirical-chance",
}
REGIME_LABEL = {"qpu_opportunity": "QPU opportunity", "queue_stress": "Queue stress"}


def cluster_bootstrap(frame: pd.DataFrame, value: str, a: str, b: str,
                      seed: int = 20260902, draws: int = 20_000
                      ) -> tuple[float, float, float]:
    """Resample seed clusters within regime, then weight regimes equally."""
    pivot = frame.pivot_table(index=["regime", "seed"], columns="policy", values=value)
    groups = [(g[a] - g[b]).dropna().to_numpy()
              for _, g in pivot.groupby(level="regime")]
    estimate = float(np.mean([g.mean() for g in groups]))
    rng = np.random.default_rng(seed)
    boot = np.empty(draws)
    for index in range(draws):
        boot[index] = np.mean([
            group[rng.integers(0, len(group), len(group))].mean()
            for group in groups
        ])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return estimate, float(lo), float(hi)


def sign_flip_pvalue(frame: pd.DataFrame, value: str, a: str, b: str,
                     seed: int = 20260905, draws: int = 100_000) -> float:
    pivot = frame.pivot_table(index=["regime", "seed"], columns="policy", values=value)
    delta = (pivot[a] - pivot[b]).dropna().to_numpy()
    observed = abs(float(delta.mean()))
    rng = np.random.default_rng(seed)
    extreme = 0
    batch = 2000
    for start in range(0, draws, batch):
        n = min(batch, draws - start)
        signs = rng.choice((-1.0, 1.0), size=(n, len(delta)))
        extreme += int((np.abs((signs * delta).mean(axis=1)) >= observed).sum())
    return (extreme + 1) / (draws + 1)


def leave_one_out_range(frame: pd.DataFrame, value: str, a: str, b: str
                        ) -> tuple[float, float]:
    pivot = frame.pivot_table(index=["regime", "seed"], columns="policy", values=value)
    delta = (pivot[a] - pivot[b]).dropna()
    estimates = [float(delta.drop(index).mean()) for index in delta.index]
    return min(estimates), max(estimates)


def aggregate_panel(raw: pd.DataFrame) -> pd.DataFrame:
    return raw.groupby(["candidate", "placement"], as_index=False).agg(
        mean_action_loss=("action_loss", "mean"),
        mean_quality_loss=("quality_loss", "mean"),
        mean_delay_s=("delay_s", "mean"),
        p90_delay_s=("delay_s", lambda x: float(np.quantile(x, 0.90))),
        risk=("service_violation", "mean"),
        failures=("service_violation", "sum"),
        draws=("service_violation", "size"),
    )


def sensitivity(selection_draws: pd.DataFrame, deployment_draws: pd.DataFrame
                ) -> pd.DataFrame:
    """Reapply CAPO-Cert across confirmation budgets, targets, and confidence."""
    rows = []
    for regime in sorted(selection_draws.regime.unique()):
        seeds = sorted(selection_draws[selection_draws.regime == regime].seed.unique())
        for seed in seeds:
            s_all = selection_draws[
                (selection_draws.regime == regime) & (selection_draws.seed == seed)
            ]
            d_all = deployment_draws[
                (deployment_draws.regime == regime) & (deployment_draws.seed == seed)
            ]
            deployment = aggregate_panel(d_all)
            for m in (40, 80, 120):
                confirmation = aggregate_panel(s_all[s_all.draw < m])
                for eta in (0.05, 0.10, 0.15):
                    for alpha in (0.01, 0.05, 0.10):
                        for fallback_loss in (0.20, 0.30, 0.40):
                            choices, _ = select_policies(
                                confirmation, eta, alpha, fallback_loss
                            )
                            choice = choices["capo_cert"]
                            applied = deployment[
                                deployment.candidate == choice.applied_candidate
                            ].iloc[0]
                            realized_loss = (
                                fallback_loss if choice.fallback_used
                                else applied.mean_action_loss
                            )
                            rows.append({
                                "regime": regime,
                                "seed": seed,
                                "confirmation_draws": m,
                                "eta": eta,
                                "alpha": alpha,
                                "fallback_loss": fallback_loss,
                                "certificate_issued": choice.certificate_issued,
                                "fallback_used": choice.fallback_used,
                                "service_violation": applied.risk,
                                "action_loss": realized_loss,
                            })
    return pd.DataFrame(rows)


def main() -> int:
    GEN.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(RES / "reliability_selection.csv")
    drift = pd.read_csv(RES / "reliability_selection_drift.csv")
    selection_draws = pd.read_csv(RES / "reliability_selection_selection_draws.csv")
    deployment_draws = pd.read_csv(RES / "reliability_selection_deployment_draws.csv")

    summary = data.groupby(["regime", "policy"], as_index=False).agg(
        action_loss=("action_loss", "mean"),
        violation=("service_violation", "mean"),
        regret=("regret", "mean"),
        qpu_rate=("qpu_selected", "mean"),
        certificate_rate=("certificate_issued", "mean"),
        fallback_rate=("fallback_used", "mean"),
    )
    summary.to_csv(RES / "reliability_selection_summary.csv", index=False)

    contrasts = {}
    for comparator in (
        "expected_loss",
        "empirical_chance",
        "deadline_threshold",
        "resource_plan",
        "balanced_scheduler",
        "classical_only",
    ):
        contrasts[comparator] = {
            "action_loss_reduction": cluster_bootstrap(
                data, "action_loss", comparator, "capo_cert"
            ),
            "violation_reduction": cluster_bootstrap(
                data, "service_violation", comparator, "capo_cert", seed=20260903
            ),
            "regret_reduction": cluster_bootstrap(
                data, "regret", comparator, "capo_cert", seed=20260904
            ),
        }

    cert = data[data.policy == "capo_cert"].copy()
    issued = cert[cert.certificate_issued]
    fallback = cert[~cert.certificate_issued]
    certificate_diagnostics = {
        "certificate_rate": float(cert.certificate_issued.mean()),
        "false_qualification_rate": float(issued.false_qualification.mean()) if len(issued) else 0.0,
        "conditional_certified_risk": float(issued.service_violation.mean()) if len(issued) else None,
        "conditional_fallback_risk": float(fallback.service_violation.mean()) if len(fallback) else None,
        "unconditional_completed_policy_risk": float(cert.service_violation.mean()),
        "issued_clusters": int(len(issued)),
        "fallback_clusters": int(len(fallback)),
    }

    drift_summary = drift.groupby(["regime", "policy"], as_index=False).agg(
        action_loss=("action_loss", "mean"),
        violation=("service_violation", "mean"),
        certificate_rate=("certificate_issued", "mean"),
        fallback_rate=("fallback_used", "mean"),
    )
    drift_summary.to_csv(RES / "reliability_selection_drift_summary.csv", index=False)

    sens = sensitivity(selection_draws, deployment_draws)
    sens.to_csv(RES / "reliability_selection_sensitivity.csv", index=False)
    sens_summary = sens.groupby(
        ["confirmation_draws", "eta", "alpha", "fallback_loss"], as_index=False
    ).agg(
        certificate_rate=("certificate_issued", "mean"),
        violation=("service_violation", "mean"),
        action_loss=("action_loss", "mean"),
    )
    sens_summary.to_csv(RES / "reliability_selection_sensitivity_summary.csv", index=False)

    example = issued.sort_values(["regime", "seed"]).iloc[0]
    example_context = selection_draws[
        (selection_draws.regime == example.regime)
        & (selection_draws.seed == example.seed)
        & (selection_draws.candidate >= 0)
    ]
    example_panel = aggregate_panel(example_context)
    _, example_bounds = select_policies(
        example_panel, sla=0.10, family_alpha=0.05
    )
    example_selection = example_context[
        example_context.candidate == example.selected_candidate
    ]
    example_failures = int(example_selection.service_violation.sum())
    example_draws = int(len(example_selection))
    example_ucb = float(example.selection_risk_ucb)
    zero_failure_minimum = int(math.ceil(math.log(0.05 / 6) / math.log(1 - 0.10)))

    rows = []
    order = [
        "capo_cert",
        "expected_loss",
        "empirical_chance",
        "deadline_threshold",
        "resource_plan",
        "balanced_scheduler",
        "classical_only",
    ]
    for regime in ("qpu_opportunity", "queue_stress"):
        block = summary[summary.regime == regime].set_index("policy")
        for policy in order:
            r = block.loc[policy]
            rows.append(
                f"{REGIME_LABEL[regime]} & {POLICY_LABEL[policy]} & "
                f"{r.action_loss:.3f} & {100*r.violation:.1f}\\% & "
                f"{100*r.certificate_rate:.0f}\\% & {100*r.fallback_rate:.0f}\\% "
                + ROW_END
            )
    table = [
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"Regime & Policy & Loss & Violation & Certificate & Fallback \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (GEN / "tab_reliability_selection.tex").write_text("\n".join(table) + "\n")

    example_rows = []
    for row in example_bounds.sort_values("candidate").itertuples():
        qualifies = "yes" if row.risk_ucb <= 0.10 else "no"
        placement = str(row.placement).replace("_", r"\_")
        example_rows.append(
            f"F{int(row.candidate)} & {placement} & "
            f"{int(row.failures)}/{int(row.draws)} & {row.risk:.3f} & "
            f"{row.risk_ucb:.3f} & {qualifies} " + ROW_END
        )
    example_table = [
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"Finalist & Venue & Failures & $\widehat p$ & Upper & Qualifies " + ROW_END,
        r"\midrule",
        *example_rows,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (GEN / "tab_certificate_example.tex").write_text(
        "\n".join(example_table) + "\n"
    )

    drift_rows = []
    drift_order = [
        "capo_static",
        "capo_expire_fallback",
        "capo_rolling_recert",
        "empirical_rolling",
        "classical_only",
    ]
    for regime in ("qpu_opportunity", "queue_stress"):
        block = drift_summary[drift_summary.regime == regime].set_index("policy")
        for policy in drift_order:
            r = block.loc[policy]
            drift_rows.append(
                f"{REGIME_LABEL[regime]} & {POLICY_LABEL[policy]} & "
                f"{r.action_loss:.3f} & {100*r.violation:.1f}\\% & "
                f"{100*r.certificate_rate:.0f}\\% & {100*r.fallback_rate:.0f}\\% "
                + ROW_END
            )
    drift_table = [
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"Regime & Drift policy & Loss & Violation & Certificate & Fallback \\",
        r"\midrule",
        *drift_rows,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (GEN / "tab_certificate_drift.tex").write_text("\n".join(drift_table) + "\n")

    sensitivity_specs = [
        (40, 0.10, 0.05, 0.30),
        (80, 0.10, 0.05, 0.30),
        (120, 0.10, 0.05, 0.30),
        (120, 0.05, 0.05, 0.30),
        (120, 0.15, 0.05, 0.30),
        (120, 0.10, 0.01, 0.30),
        (120, 0.10, 0.10, 0.30),
        (120, 0.10, 0.05, 0.20),
        (120, 0.10, 0.05, 0.40),
    ]
    sensitivity_rows = []
    for m, eta, alpha, fallback_loss in sensitivity_specs:
        row = sens_summary[
            (sens_summary.confirmation_draws == m)
            & (sens_summary.eta == eta)
            & (sens_summary.alpha == alpha)
            & (sens_summary.fallback_loss == fallback_loss)
        ].iloc[0]
        sensitivity_rows.append(
            f"{m} & {eta:.2f} & {alpha:.2f} & {fallback_loss:.2f} & "
            f"{100*row.certificate_rate:.1f}\\% & {100*row.violation:.2f}\\% & "
            f"{row.action_loss:.3f} " + ROW_END
        )
    sensitivity_table = [
        r"\begin{tabular}{@{}rrrrrrr@{}}",
        r"\toprule",
        r"$m_c$ & $\eta$ & $\alpha$ & $L_{\rm fb}$ & Certificate & Violation & Loss \\",
        r"\midrule",
        *sensitivity_rows,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (GEN / "tab_certificate_sensitivity.tex").write_text(
        "\n".join(sensitivity_table) + "\n"
    )

    def pct(x: float) -> str:
        return f"{100 * x:.2f}"

    eu = contrasts["expected_loss"]
    ec = contrasts["empirical_chance"]
    static = drift[drift.policy == "capo_static"].service_violation.mean()
    expire = drift[drift.policy == "capo_expire_fallback"].service_violation.mean()
    rolling = drift[drift.policy == "capo_rolling_recert"].service_violation.mean()
    primary_sens = sens_summary[
        (sens_summary.confirmation_draws == 120)
        & (sens_summary.eta == 0.10)
        & (sens_summary.alpha == 0.05)
        & (sens_summary.fallback_loss == 0.30)
    ].iloc[0]
    low_m_sens = sens_summary[
        (sens_summary.confirmation_draws == 40)
        & (sens_summary.eta == 0.10)
        & (sens_summary.alpha == 0.05)
        & (sens_summary.fallback_loss == 0.30)
    ].iloc[0]
    p_value = sign_flip_pvalue(
        data, "service_violation", "expected_loss", "capo_cert"
    )
    influence_lo, influence_hi = leave_one_out_range(
        data, "service_violation", "expected_loss", "capo_cert"
    )

    macros = [
        f"\\newcommand{{\\CertLossVsExpected}}{{{eu['action_loss_reduction'][0]:+.3f}}}",
        f"\\newcommand{{\\CertLossVsExpectedLo}}{{{eu['action_loss_reduction'][1]:+.3f}}}",
        f"\\newcommand{{\\CertLossVsExpectedHi}}{{{eu['action_loss_reduction'][2]:+.3f}}}",
        f"\\newcommand{{\\CertRiskVsExpectedPP}}{{{pct(eu['violation_reduction'][0])}}}",
        f"\\newcommand{{\\CertRiskVsExpectedLoPP}}{{{pct(eu['violation_reduction'][1])}}}",
        f"\\newcommand{{\\CertRiskVsExpectedHiPP}}{{{pct(eu['violation_reduction'][2])}}}",
        f"\\newcommand{{\\CertRiskVsEmpiricalPP}}{{{pct(ec['violation_reduction'][0])}}}",
        f"\\newcommand{{\\CertRiskVsEmpiricalLoPP}}{{{pct(ec['violation_reduction'][1])}}}",
        f"\\newcommand{{\\CertRiskVsEmpiricalHiPP}}{{{pct(ec['violation_reduction'][2])}}}",
        f"\\newcommand{{\\CertIssuedRate}}{{{pct(certificate_diagnostics['certificate_rate'])}\\%}}",
        f"\\newcommand{{\\CertFalseQualification}}{{{pct(certificate_diagnostics['false_qualification_rate'])}\\%}}",
        f"\\newcommand{{\\CertConditionalRisk}}{{{pct(certificate_diagnostics['conditional_certified_risk'])}\\%}}",
        f"\\newcommand{{\\CertFallbackRisk}}{{{pct(certificate_diagnostics['conditional_fallback_risk'])}\\%}}",
        f"\\newcommand{{\\CertCompletedRisk}}{{{pct(certificate_diagnostics['unconditional_completed_policy_risk'])}\\%}}",
        f"\\newcommand{{\\CertClusters}}{{{data[['regime', 'seed']].drop_duplicates().shape[0]}}}",
        f"\\newcommand{{\\CertSeedsPerRegime}}{{{data.seed.nunique()}}}",
        f"\\newcommand{{\\CertFinalists}}{{{selection_draws[selection_draws.candidate >= 0].candidate.nunique()}}}",
        f"\\newcommand{{\\CertConfirmationDraws}}{{{selection_draws.draw.nunique()}}}",
        f"\\newcommand{{\\CertDeploymentDraws}}{{{deployment_draws.draw.nunique()}}}",
        f"\\newcommand{{\\CertPermutationP}}{{{p_value:.4f}}}",
        f"\\newcommand{{\\CertInfluenceLoPP}}{{{100*influence_lo:.2f}}}",
        f"\\newcommand{{\\CertInfluenceHiPP}}{{{100*influence_hi:.2f}}}",
        f"\\newcommand{{\\CertExampleSeed}}{{{int(example.seed)}}}",
        f"\\newcommand{{\\CertExampleFailures}}{{{example_failures}}}",
        f"\\newcommand{{\\CertExampleDraws}}{{{example_draws}}}",
        f"\\newcommand{{\\CertExampleUcb}}{{{example_ucb:.3f}}}",
        f"\\newcommand{{\\CertZeroFailureMinimum}}{{{zero_failure_minimum}}}",
        f"\\newcommand{{\\CertFortyDrawIssued}}{{{pct(low_m_sens.certificate_rate)}\\%}}",
        f"\\newcommand{{\\CertPrimaryIssued}}{{{pct(primary_sens.certificate_rate)}\\%}}",
        f"\\newcommand{{\\CertDriftStaticRisk}}{{{pct(static)}\\%}}",
        f"\\newcommand{{\\CertDriftExpireRisk}}{{{pct(expire)}\\%}}",
        f"\\newcommand{{\\CertDriftRollingRisk}}{{{pct(rolling)}\\%}}",
    ]
    (GEN / "reliability_selection_numbers.tex").write_text("\n".join(macros) + "\n")

    report = {
        "summary": summary.to_dict("records"),
        "certificate_diagnostics": certificate_diagnostics,
        "clustered_contrasts": contrasts,
        "randomization_p_expected_loss_violation": p_value,
        "leave_one_cluster_out_expected_loss_violation_range": [influence_lo, influence_hi],
        "worked_example": {
            "regime": example.regime,
            "seed": int(example.seed),
            "failures": example_failures,
            "draws": example_draws,
            "upper_bound": example_ucb,
            "eta": 0.10,
            "family_confidence": 0.95,
            "finalists": 6,
            "per_finalist_alpha": 0.05 / 6,
            "zero_failure_minimum_draws": zero_failure_minimum,
        },
        "drift_summary": drift_summary.to_dict("records"),
        "sensitivity_summary": sens_summary.to_dict("records"),
    }
    (RES / "reliability_selection_analysis.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
