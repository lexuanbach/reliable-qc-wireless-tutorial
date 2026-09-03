#!/usr/bin/env python3
"""Leak gate + presence + standalone-import check."""
import sys
from pathlib import Path
root = Path(__file__).resolve().parent
bad = [p for p in root.rglob("*") if p.suffix in (".tex", ".pdf")]
assert not bad, f"manuscript leak: {bad[:3]}"
required = ["coopt.py", "search.py", "networkqbench.py", "run_experiments.py",
            "verify_assumption.py", "run_reliability_selection.py",
            "run_drift_magnitude_sensitivity.py",
            "analyze_reliability_selection.py", "run_wireless_closed_loop.py",
            "requirements.txt", "environment.yml", "VERSION",
            "results/raw_main_qaoa.csv",
            "results/revision_analysis.json", "results/clearaccept_sota.csv",
            "results/reliability_selection.csv",
            "results/reliability_selection_manifest.json", "DATA_DICTIONARY.md",
            "results/reliability_selection_drift.csv",
            "results/drift_magnitude_sensitivity.json",
            "results/certificate_coverage.json",
            "results/wireless_closed_loop.csv",
            "results/wireless_closed_loop_manifest.json",
            "results/trace_runtime_uncertainty.json",
            "results/tutorial_amortization_summary.csv",
            "tutorial_quickstart.py"]
missing = [r for r in required if not (root / r).exists()]
assert not missing, f"missing: {missing}"
sys.path.insert(0, str(root))
import coopt  # noqa: F401  resolves networkqbench from the local copy
import pandas as pd
assert len(pd.read_csv(root / "results/raw_main_qaoa.csv")) > 100
reliability = pd.read_csv(root / "results/reliability_selection.csv")
fields = {
    "certificate_issued", "selected_candidate", "applied_action",
    "fallback_reason", "certificate_expiry"
}
assert fields.issubset(reliability.columns), fields - set(reliability.columns)
print("artifact check: OK")
