# Paper 06 standalone tutorial artifact

This versioned artifact supports the tutorial on reliability-aware
multi-fidelity selection for quantum-enabled wireless control. CAPO is its
runnable selection example. All quantum execution is exact statevector
simulation under labeled noise models. Deployment scores use held-out
perturbed simulation. Every latency, energy, or monetary figure is a labeled
model or a labeled trace-calibrated quantity. The package is self-contained,
including its workload generators in `networkqbench.py`. No .tex/.pdf
ships in this folder; `check_artifact.py` enforces that (analysis scripts
that would emit LaTeX macro files skip that step when run standalone or
write outside this tree).

Snapshot: `paper06-tutorial-artifact-2026.09-r4`

## Experiment map
| Stage | Run | Analyze |
|---|---|---|
| Exp1-8 core (5 families, budgets, scale, mismatch) | run_experiments.py | analyze_results.py, make_paper_assets.py |
| Engine ablations + amortization + baselines | run_experiments.py (methods flags) | analyze_revision.py, amortization_ledger.py |
| Confirmatory ledger / components / cost | run_clear_accept_experiments.py | analyze_clear_accept.py |
| Same-finalist certified reliability selection | run_reliability_selection.py | analyze_reliability_selection.py |
| Exact-binomial coverage audit | run_certificate_coverage.py | stored JSON and generated macros |
| Dynamic wireless channel assignment | run_wireless_closed_loop.py | integrated clustered analysis |
| Trace replay, mitigation/winsor/digit sensitivity | run_fable_trace_replay.py, run_fable_robustness.py | analyze_fable.py |
| Runtime-residual propagation | stored trace selections | analyze_trace_runtime_uncertainty.py |
| Queue-envelope stress | run_revision_queue_stress.py | analyze_revision.py |
| Assumption verification (108/108 dominance) | verify_assumption.py | — |

## Headline numbers (regenerable)
- Joint search vs sequential design-then-deploy: family-stratified +1.51 utility (95% CI [+0.59,+2.63]); Holm-adjusted rank test not significant; gain concentrated in tail-failure avoidance.
- Proposal engines (GP, extra-trees, screened/plain random) statistically tied on the same joint space across budgets 10-60 and up to n=20 / |Z|=12,000.
- Comparator-aware amortization for the tutorial's `joint_mf` policy uses incremental search cost. Finite break-even occurs in 100% of cases versus the hand default, with a median of 12 uses. It occurs in 65% of cases versus Classical-only, with a median of 5,066 uses among finite cases.
- Noise-model mismatch (heavier tails, G2^0.9, readout flips): method ranking Spearman 1.00.
- Completed CAPO-Cert vs Expected-loss: 7.06 percentage points fewer service violations with clustered 95% CI [4.35, 9.88]. The action-loss reduction is -0.032 with interval [-0.052, -0.014], which quantifies the reliability premium.
- CAPO-Cert issues in 20.8% of 48 stable seed-regime clusters. Conditional certified risk is 1.90%, fallback risk is 0%, completed-policy risk is 0.40%, and no false qualification is observed.
- In 100,000 all-unsafe Bernoulli panels at true risk 0.105, simultaneous exact-binomial coverage is 97.93% and false certification is 2.07%.
- In 48 dynamic channel-assignment trajectories, CAPO-Cert matches Expected-loss, Empirical-chance, rolling selection, and the hindsight oracle on the reliable local path. It prevents 3.24 dropped-demand units per trajectory relative to budgeted local greedy, with clustered interval [2.27, 4.27].

Use `requirements.txt` or `environment.yml` for the pinned environment.

## Tutorial quick start

The tutorial manuscript uses this artifact as a reference implementation.
The quickest audit reads the immutable stored outputs and reconstructs the
path-selection, completed reliability, trace-replay, dynamic wireless, and
comparator-aware amortization examples:

    python3 tutorial_quickstart.py
    python3 check_artifact.py

The quick start prints stable JSON. Its claim-boundary field records that all
circuit outcomes are simulated. Historical queue and calibration data are used
only in the trace-replay block.

`analyze_fable_revision.py` emits the revision assets: the three-arm table (`tab_newarms`), context-distribution table (`tab_context_dist`), the seed-block reconciliation figure and macros, and the full method-by-family matrix (`tab_method_family`), all from stored CSVs plus `results/clearaccept_newarms.csv`.
