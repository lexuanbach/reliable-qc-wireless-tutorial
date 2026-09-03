#!/usr/bin/env python3
"""Reconstruct the tutorial's main decision checks from stored outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def path_example() -> dict:
    rows = read_csv("clearaccept_confirmatory.csv")
    selected = {
        row["method"]: row
        for row in rows
        if row["workload"] == "qaoa"
        and int(row["seed"]) == 101
        and row["method"] in {"sequential", "joint_mf"}
    }
    if set(selected) != {"sequential", "joint_mf"}:
        raise RuntimeError("stored path-example rows are missing")

    def terms(row: dict[str, str]) -> dict:
        quality = float(row["quality_loss"])
        total_loss = -float(row["utility"])
        return {
            "quality_loss": quality,
            "systems_loss": total_loss - quality,
            "total_loss": total_loss,
            "placement": row["cfg_placement"],
        }

    sequential = terms(selected["sequential"])
    joint = terms(selected["joint_mf"])
    return {
        "workload": "qaoa_path",
        "seed": 101,
        "evidence": "calibration-conditioned statevector simulation",
        "sequential": sequential,
        "joint": joint,
        "quality_sacrifice": joint["quality_loss"] - sequential["quality_loss"],
        "systems_loss_reduction": (
            sequential["systems_loss"] - joint["systems_loss"]
        ),
        "joint_utility_gain": (
            sequential["total_loss"] - joint["total_loss"]
        ),
    }


def reliability_check() -> dict:
    with (RESULTS / "reliability_selection_analysis.json").open(
        encoding="utf-8"
    ) as handle:
        analysis = json.load(handle)
    estimate, low, high = analysis["clustered_contrasts"]["expected_loss"][
        "violation_reduction"
    ]
    diagnostics = analysis["certificate_diagnostics"]
    return {
        "comparison": "completed exact-binomial policy vs expected-loss policy",
        "clusters": sum(
            1 for _ in {
                (row["regime"], row["seed"])
                for row in read_csv("reliability_selection.csv")
            }
        ),
        "violation_reduction": estimate,
        "clustered_95_interval": [low, high],
        "certificate_rate": diagnostics["certificate_rate"],
        "false_qualification_rate": diagnostics["false_qualification_rate"],
        "conditional_certified_risk": diagnostics["conditional_certified_risk"],
        "fallback_risk": diagnostics["conditional_fallback_risk"],
        "completed_policy_risk": diagnostics["unconditional_completed_policy_risk"],
        "same_finalists_actions_abstention_and_fallback": True,
    }


def wireless_check() -> dict:
    with (RESULTS / "wireless_closed_loop_manifest.json").open(
        encoding="utf-8"
    ) as handle:
        manifest = json.load(handle)
    rows = read_csv("wireless_closed_loop_summary.csv")
    return {
        "evidence": manifest["evidence"],
        "claim_boundary": manifest["not_evidence"],
        "clusters": manifest["clusters"],
        "deployment_windows_per_cluster": manifest[
            "deployment_windows_per_cluster"
        ],
        "policy_summary": rows,
    }


def trace_check() -> dict:
    with (RESULTS / "fable_analysis.json").open(encoding="utf-8") as handle:
        analysis = json.load(handle)
    utility = analysis["trace_joint_mf_vs_sequential"]
    risk = analysis["trace_risk_joint_mf_vs_sequential"]
    return {
        "comparison": "joint multi-fidelity vs sequential",
        "utility_gain": utility[0],
        "utility_95_interval": utility[1:],
        "violation_change": risk[0],
        "violation_change_95_interval": risk[1:],
        "evidence": "historical service trace replay with simulated circuits",
    }


def drift_magnitude_check() -> dict:
    with (RESULTS / "drift_magnitude_sensitivity.json").open(
        encoding="utf-8"
    ) as handle:
        report = json.load(handle)
    return {
        "evidence": report["evidence"],
        "drift_strengths": report["drift_strengths"],
        "summary": report["summary"],
    }


def amortization_check() -> dict:
    rows = {
        row["rival"]: row
        for row in read_csv("tutorial_amortization_summary.csv")
    }
    return {
        rival: {
            "selection_policy": rows[rival]["selection_policy"],
            "cases": int(rows[rival]["cases"]),
            "finite_fraction": float(rows[rival]["finite_fraction"]),
            "median_break_even_uses_among_finite": float(
                rows[rival]["median_n_be"]
            ),
        }
        for rival in ("hand_designed", "classical_only")
    }


def main() -> None:
    report = {
        "tutorial_quickstart": "stored-output reconstruction",
        "path_example": path_example(),
        "reliability": reliability_check(),
        "trace_replay": trace_check(),
        "wireless_closed_loop": wireless_check(),
        "queue_drift_magnitude": drift_magnitude_check(),
        "amortization": amortization_check(),
        "claim_boundary": (
            "No record in this report is a physical-QPU execution. "
            "Circuit outcomes are simulated."
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
