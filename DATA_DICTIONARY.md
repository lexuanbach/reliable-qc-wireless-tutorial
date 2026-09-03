# CAPO result data dictionary

Raw result files are keyed by `experiment`, `workload`, `seed`, and `method`.
Configuration columns have a `cfg_` prefix. All quantum and systems behavior
is simulated or modeled unless explicitly stated otherwise.

| Field | Meaning |
|---|---|
| `workload` | QAOA path/channel/placement family or VQC synthetic/real family. |
| `method` | CAPO engine, sequential/fixed policy, or hindsight placement bound. |
| `utility`, `utility_p10` | Mean and lower-tail held-out deployment utility. |
| `quality_loss` | Feasibility-aware task loss or deployed classification error. |
| `delay_s`, `deadline_miss` | Modeled complete latency and mean miss indicator. |
| `infeasible`, `realized_fail` | Constraint/budget infeasibility and modeled execution failure. |
| `energy_j`, `money`, `cost_scalar` | Modeled resource components retained for rescoring. |
| `search_cost_modeled_s` | Charged search ledger used in break-even calculations. |
| `cfg_*` | Selected circuit/model, training, execution, and placement variables. |

Revision summaries under `results/` include family-stratified comparisons,
leave-one-family-out intervals, comparator-aware amortization,
deadline/failure sensitivity, a literature-calibrated queue envelope, and the
ten-seed 20-variable extension. The queue envelope is not raw provider
telemetry or a current service-level estimate.

`clearaccept_newarms.csv` shares the confirmatory schema and holds the three
revision arms (`exhaustive_f0`, `device_aware_fallback`,
`capo_joint_only_espend`) on the same seeds, contexts, and 40 paired
deployment draws; `search_cost_units` records their full charged spend
(the exhaustive arm intentionally exceeds the proposal-loop threshold).

`clearaccept_sota.csv` uses the same confirmatory schema for the matched-budget
TPE and successive-halving baselines. Its rows share the seed, context, action
space, utility, and deployment-draw definitions of the confirmatory block.

`reliability_selection.csv` contains one row per policy, seed, and service
regime for the completed same-finalist CAPO-Cert experiment. `action_loss` is
one on any service violation and otherwise equals normalized task-quality
loss. `selection_risk_ucb` is the simultaneous exact one-sided binomial upper
bound. `certificate_issued` records whether that bound meets the 0.10 target.
`selected_candidate` is empty on abstention. `applied_candidate` and
`applied_action` identify what entered deployment. `fallback_reason` explains
every abstention. `certificate_expiry` identifies the validity window.
`false_qualification` marks an issued certificate whose independent stable
deployment risk exceeds the target. The companion draw files retain each
finalist plus fallback evaluation. `reliability_selection_manifest.json`
records the 120/100 draw split, 24 seeds per regime, stable and shifted queue
mixtures, finalist construction, and fallback contract.

`reliability_selection_drift.csv` contains static, expired, rolling
recertification, rolling empirical, and local-only policies after a declared
queue-regime change. `reliability_selection_sensitivity.csv` reapplies the
selector at confirmation sizes 40, 80, and 120 across three risk targets plus
three confidence levels plus three fallback-loss values.

`drift_magnitude_sensitivity.csv` reuses the stable completed decision at
queue-shift strengths 0, 0.5, 1.0, and 1.5. It pairs static certificate reuse
with immediate expiry plus the validated fallback on 100 deployment contexts
per seed-regime cluster.

`certificate_coverage.json` records 100,000-panel Monte Carlo audits of the
simultaneous exact-binomial bounds. It reports familywise coverage, false
certification, and qualification frequency for safe, unsafe, and mixed-risk
panels.

`wireless_closed_loop.csv` contains one row per policy and topology-load
cluster. `mean_interference`, `mean_backlog`, `dropped_demand`, `switch_rate`,
`deadline_miss_rate`, and `fallback_rate` follow the applied channel action
through 40 deployment windows. The hindsight oracle is nondeployable.

`trace_runtime_uncertainty.json` propagates held-out Extra-Trees log-runtime
residuals through the fixed selected configurations in the historical replay.

`tutorial_amortization_summary.csv` is the authoritative break-even summary
for the tutorial's `joint_mf` policy. It records 60 seed-family comparisons
against the hand-designed default plus 60 against Classical-only. The paper,
appendix figure, README, and quick start read this same computation.
