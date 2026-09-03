#!/usr/bin/env python3
"""Paired queue-drift sensitivity for the completed CAPO-Cert policy."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from coopt import Workload, evaluate
from run_reliability_selection import (
    FALLBACK_CANDIDATE,
    FALLBACK_QUALITY_LOSS,
    action_loss,
    make_finalists,
    service_context,
    violation,
)


ARTIFACT = Path(__file__).resolve().parent
RESULTS = ARTIFACT / "results"
GENERATED = ARTIFACT.parent / "submission" / "paper" / "generated"
STRENGTHS = (0.0, 0.5, 1.0, 1.5)
DRAWS = 100


def main() -> int:
    decisions = pd.read_csv(RESULTS / "reliability_selection.csv")
    decisions = decisions[decisions.policy == "capo_cert"]
    rows = []
    for record in decisions.itertuples():
        candidate = int(record.applied_candidate)
        finalists = None
        if candidate != FALLBACK_CANDIDATE:
            workload = Workload.make("qaoa_large", int(record.seed))
            finalists = make_finalists(workload, int(record.seed), record.regime)
            config = finalists[candidate]
        for strength in STRENGTHS:
            if candidate == FALLBACK_CANDIDATE:
                mean_loss = FALLBACK_QUALITY_LOSS
                risk = 0.0
            else:
                losses = []
                failures = []
                for draw in range(DRAWS):
                    context = service_context(
                        int(record.seed), draw, record.regime, phase=30,
                        drift_strength=strength,
                    )
                    rng = np.random.default_rng(
                        93_000_000 + int(record.seed) * 10_000
                        + candidate * DRAWS + draw
                    )
                    outcome = evaluate(workload, config, context, 3, rng)
                    losses.append(action_loss(outcome))
                    failures.append(violation(outcome))
                mean_loss = float(np.mean(losses))
                risk = float(np.mean(failures))
            rows.append({
                "regime": record.regime,
                "seed": int(record.seed),
                "drift_strength": strength,
                "policy": "static_certificate",
                "action_loss": mean_loss,
                "service_violation": risk,
            })
            rows.append({
                "regime": record.regime,
                "seed": int(record.seed),
                "drift_strength": strength,
                "policy": "expire_fallback",
                "action_loss": FALLBACK_QUALITY_LOSS,
                "service_violation": 0.0,
            })

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "drift_magnitude_sensitivity.csv", index=False)
    summary = frame.groupby(["drift_strength", "policy"], as_index=False).agg(
        action_loss=("action_loss", "mean"),
        service_violation=("service_violation", "mean"),
    )
    summary.to_csv(RESULTS / "drift_magnitude_sensitivity_summary.csv", index=False)

    table_rows = []
    for strength in STRENGTHS:
        block = summary[summary.drift_strength == strength].set_index("policy")
        static = block.loc["static_certificate"]
        expired = block.loc["expire_fallback"]
        table_rows.append(
            f"{strength:.1f} & {static.action_loss:.3f} & "
            f"{100*static.service_violation:.2f}\\% & "
            f"{expired.action_loss:.3f} & "
            f"{100*expired.service_violation:.2f}\\% \\\\"
        )
    table = [
        r"\begin{tabular}{@{}rrrrr@{}}",
        r"\toprule",
        r"Shift & \multicolumn{2}{c}{Static certificate} & \multicolumn{2}{c}{Expire + fallback} \\",
        r" & Loss & Violation & Loss & Violation \\",
        r"\midrule",
        *table_rows,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (GENERATED / "tab_drift_magnitude.tex").write_text("\n".join(table) + "\n")

    static = summary[summary.policy == "static_certificate"].set_index(
        "drift_strength"
    )
    macros = [
        f"\\newcommand{{\\DriftStaticZeroRisk}}{{{100*static.loc[0.0, 'service_violation']:.2f}\\%}}",
        f"\\newcommand{{\\DriftStaticHighRisk}}{{{100*static.loc[1.5, 'service_violation']:.2f}\\%}}",
    ]
    (GENERATED / "drift_magnitude_numbers.tex").write_text("\n".join(macros) + "\n")
    report = {
        "evidence": "paired controlled service-model sensitivity",
        "drift_strengths": list(STRENGTHS),
        "deployment_draws_per_cluster": DRAWS,
        "summary": summary.to_dict("records"),
    }
    (RESULTS / "drift_magnitude_sensitivity.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
