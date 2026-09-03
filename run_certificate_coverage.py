#!/usr/bin/env python3
"""Monte Carlo audit of simultaneous exact-binomial reliability bounds."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import beta


ARTIFACT = Path(__file__).resolve().parent
RESULTS = ARTIFACT / "results"
GENERATED = ARTIFACT.parent / "submission" / "paper" / "generated"


def upper_bounds(counts: np.ndarray, draws: int, alpha: float) -> np.ndarray:
    upper = beta.ppf(1 - alpha, counts + 1, draws - counts)
    upper[counts == draws] = 1.0
    return upper


def run_case(probabilities: np.ndarray, draws: int, family_alpha: float,
             repetitions: int, seed: int, eta: float) -> dict:
    rng = np.random.default_rng(seed)
    counts = rng.binomial(draws, probabilities, size=(repetitions, len(probabilities)))
    upper = upper_bounds(counts, draws, family_alpha / len(probabilities))
    simultaneous_coverage = np.all(upper >= probabilities[None, :], axis=1)
    unsafe = probabilities > eta
    false_certification = np.any(upper[:, unsafe] <= eta, axis=1) if unsafe.any() else np.zeros(repetitions, dtype=bool)
    qualification = upper <= eta
    return {
        "probabilities": probabilities.tolist(),
        "draws": draws,
        "family_alpha": family_alpha,
        "repetitions": repetitions,
        "simultaneous_coverage": float(simultaneous_coverage.mean()),
        "familywise_miscoverage": float(1 - simultaneous_coverage.mean()),
        "false_certification_probability": float(false_certification.mean()),
        "mean_qualified_candidates": float(qualification.sum(axis=1).mean()),
        "any_qualification_probability": float(qualification.any(axis=1).mean()),
    }


def main() -> int:
    eta = 0.10
    repetitions = 100_000
    scenarios = {
        "all_just_above_target": np.full(6, 0.105),
        "mixed_safe_and_unsafe": np.array([0.03, 0.05, 0.08, 0.12, 0.15, 0.20]),
        "all_safe": np.array([0.01, 0.03, 0.05, 0.06, 0.08, 0.10]),
    }
    report = {
        name: run_case(values, 120, 0.05, repetitions, 860_000 + index, eta)
        for index, (name, values) in enumerate(scenarios.items())
    }
    report["sample_size_sensitivity"] = [
        run_case(scenarios["mixed_safe_and_unsafe"], draws, alpha,
                 repetitions, 870_000 + draws * 10 + int(alpha * 100), eta)
        for draws in (40, 80, 120, 240)
        for alpha in (0.01, 0.05, 0.10)
    ]
    RESULTS.mkdir(parents=True, exist_ok=True)
    GENERATED.mkdir(parents=True, exist_ok=True)
    (RESULTS / "certificate_coverage.json").write_text(json.dumps(report, indent=2))
    primary = report["all_just_above_target"]
    mixed = report["mixed_safe_and_unsafe"]
    macros = [
        f"\\newcommand{{\\CoverageRepetitions}}{{{repetitions:,}}}",
        f"\\newcommand{{\\CoverageFamilywise}}{{{100*primary['simultaneous_coverage']:.2f}\\%}}",
        f"\\newcommand{{\\CoverageFalseCert}}{{{100*primary['false_certification_probability']:.2f}\\%}}",
        f"\\newcommand{{\\CoverageMixedFalseCert}}{{{100*mixed['false_certification_probability']:.2f}\\%}}",
    ]
    (GENERATED / "certificate_coverage_numbers.tex").write_text("\n".join(macros) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
