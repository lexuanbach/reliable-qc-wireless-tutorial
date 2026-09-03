# Reproduction guide

Snapshot: `paper06-tutorial-artifact-2026.09-r4`

All quantum outcomes are simulations. QuantumQueue and DAQEC provide historical service inputs only. No command requires QPU credentials.

## Environment

Tested on macOS arm64 with Python 3.14.6. The exact package versions are in `requirements.txt` and `environment.yml`.

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Allow about 3 GB of free memory for the largest exact-statevector runs. Stored-output checks use less than 500 MB.

## Quick stored-output audit

Expected wall time is below one minute.

```bash
python3 tutorial_quickstart.py
python3 check_artifact.py
python3 run_certificate_coverage.py
python3 analyze_reliability_selection.py
python3 run_drift_magnitude_sensitivity.py
```

## Medium end-to-end rerun

This tier reruns the two headline experiments with reduced cluster counts. Expected wall time is 5 to 15 minutes on a recent laptop. Peak memory is about 1 GB.

```bash
python3 run_reliability_selection.py --seeds 6 --out reliability_selection_medium
python3 run_wireless_closed_loop.py --seeds 12 --windows 40 --confirmation-draws 40
```

The manuscript results use 24 seeds per service regime for reliability plus 48 wireless clusters. Run the default commands to regenerate those exact files.

```bash
python3 run_reliability_selection.py
python3 analyze_reliability_selection.py
python3 run_drift_magnitude_sensitivity.py
python3 run_wireless_closed_loop.py
python3 analyze_trace_runtime_uncertainty.py
```

## Full historical experiment set

The complete rerun can take several hours because it searches exact-statevector workloads across multiple seed blocks. Run from this artifact directory.

```bash
python3 test_coopt.py
python3 run_experiments.py --seeds 6 --budget 30 --workloads qaoa --experiments 1,2,3,4,6 --tag main_qaoa
python3 run_experiments.py --seeds 6 --budget 30 --workloads vqc --experiments 1,2,3,4,6 --tag main_vqc
python3 run_experiments.py --seeds 6 --seed-offset 6 --budget 30 --workloads qaoa --experiments 1 --tag main_qaoa_s2
python3 run_experiments.py --seeds 6 --seed-offset 6 --budget 30 --workloads vqc --experiments 1 --tag main_vqc_s2
python3 run_experiments.py --seeds 12 --budget 30 --workloads qaoa_channel,qaoa_place --experiments 1 --tag main_fam
python3 run_experiments.py --seeds 12 --budget 30 --workloads vqc_cancer --experiments 1 --tag main_cancer
python3 run_experiments.py --seeds 12 --workloads qaoa --experiments 7,8 --tag main_sweepq
python3 run_experiments.py --seeds 12 --workloads vqc --experiments 7 --tag main_sweepv
python3 run_experiments.py --seeds 12 --experiments 10 --tag main_mismatch
python3 run_clear_accept_experiments.py
python3 run_clear_accept_experiments.py --methods exhaustive_f0,device_aware_fallback,capo_joint_only_espend --out clearaccept_newarms
python3 run_clear_accept_experiments.py --methods tpe,successive_halving --out clearaccept_sota
python3 run_reliability_selection.py
python3 run_certificate_coverage.py
python3 run_drift_magnitude_sensitivity.py
python3 run_wireless_closed_loop.py
python3 run_fable_trace_replay.py
python3 run_fable_robustness.py
python3 analyze_trace_runtime_uncertainty.py
python3 run_revision_queue_stress.py
python3 verify_assumption.py
python3 analyze_results.py
python3 analyze_revision.py
python3 analyze_clear_accept.py
python3 analyze_fable.py
python3 analyze_reliability_selection.py
python3 analyze_fable_revision.py
python3 amortization_ledger.py --search-cost-s 1 --baseline-search-cost-s 0 --utility-gain 0.1
```

Manifests in `results/` record seeds, evidence labels, and environment details. Analysis scripts regenerate the LaTeX tables, figures, and macros outside the artifact folder.
