# Master Dataset Schema

This document defines the schema for `data/processed/master.parquet`, the consolidated analysis dataset.

## Overview

The master dataset consolidates all experimental sessions into a single analysis-ready table. Each row represents one QEC experimental trial (one code distance, one method, one shot batch).

## Schema Definition

### Identifiers

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `session_id` | string | Unique session identifier | `brisbane_20250601_001` |
| `trial_id` | string | Unique trial identifier | `brisbane_20250601_001_d3_adaptive_001` |
| `backend` | string | IBM Quantum backend name | `ibm_brisbane` |
| `date` | date | Session date (YYYY-MM-DD) | `2025-06-01` |
| `timestamp` | datetime | Exact timestamp (UTC) | `2025-06-01T14:32:15Z` |

### Experimental Conditions

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `code_distance` | int | Surface code distance | `3`, `5`, `7` |
| `syndrome_rounds` | int | Number of syndrome rounds | `3` |
| `method` | string | Method variant | `baseline`, `adaptive` |
| `decoder` | string | Decoder used | `mwpm`, `mwpm_adaptive` |
| `qec_shots` | int | Number of QEC shots in trial | `4096` |

### Qubit Selection

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `qubit_subset` | array[int] | Selected data qubits | `[12, 14, 16, 24, 26]` |
| `ancilla_subset` | array[int] | Selected ancilla qubits | `[13, 15, 23, 25]` |
| `selection_method` | string | How qubits were selected | `calibration`, `probe_composite` |
| `composite_score` | float | Composite error index (probe method) | `0.0234` |

### Probe Metrics (for adaptive method)

| Column | Type | Description | Units |
|--------|------|-------------|-------|
| `probe_t1_mean` | float | Mean T₁ of selected qubits | μs |
| `probe_t1_std` | float | Std dev of T₁ | μs |
| `probe_t2_mean` | float | Mean T₂ of selected qubits | μs |
| `probe_t2_std` | float | Std dev of T₂ | μs |
| `probe_readout_mean` | float | Mean readout error | probability |
| `probe_readout_std` | float | Std dev of readout error | probability |
| `probe_shots` | int | Shots used for probes | count |

### Calibration Metrics (from backend)

| Column | Type | Description | Units |
|--------|------|-------------|-------|
| `cal_t1_mean` | float | Backend-reported mean T₁ | μs |
| `cal_t2_mean` | float | Backend-reported mean T₂ | μs |
| `cal_readout_mean` | float | Backend-reported readout error | probability |
| `cal_age_hours` | float | Hours since last calibration | hours |

### Primary Outcomes

| Column | Type | Description | Units |
|--------|------|-------------|-------|
| `logical_errors` | int | Count of logical errors | count |
| `total_shots` | int | Total QEC shots | count |
| `logical_error_rate` | float | `logical_errors / total_shots` | probability |
| `logical_error_rate_ci_lower` | float | 95% CI lower bound | probability |
| `logical_error_rate_ci_upper` | float | 95% CI upper bound | probability |

### Syndrome Statistics

| Column | Type | Description | Units |
|--------|------|-------------|-------|
| `syndrome_weight_mean` | float | Mean syndrome weight per round | count |
| `syndrome_weight_std` | float | Std dev of syndrome weight | count |
| `burst_index` | float | Burst index (observed/expected) | ratio |
| `max_burst_length` | int | Longest consecutive error run | count |
| `temporal_correlation` | float | Adjacent-round correlation | correlation |

### Metadata

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `qiskit_version` | string | Qiskit version used | `1.0.0` |
| `runtime_version` | string | qiskit-ibm-runtime version | `0.20.0` |
| `random_seed` | int | Random seed for reproducibility | `42` |
| `git_commit` | string | Git commit hash | `a1b2c3d4` |

## Derived Columns (computed at analysis time)

| Column | Type | Description | Formula |
|--------|------|-------------|---------|
| `paired_diff` | float | Error rate difference | `baseline_error - adaptive_error` |
| `relative_improvement` | float | Relative improvement | `paired_diff / baseline_error` |
| `is_improvement` | bool | Adaptive better than baseline | `paired_diff > 0` |

## Data Types

- All floating-point values are stored as float64
- All integer values are stored as int64
- Strings are stored as UTF-8
- Arrays are stored as Parquet list types
- Timestamps are stored as Parquet timestamp[us, UTC]

## Missing Values

- Missing numeric values: `null` (Parquet null, reads as NaN in pandas)
- Missing strings: empty string `""`
- Missing arrays: empty array `[]`

## File Size Estimate

With ~100 sessions × 2 methods × 3 distances × ~10 trials ≈ 6,000 rows
- Uncompressed: ~5 MB
- Snappy compressed: ~1 MB

## Reading the Data

```python
import pandas as pd

# Read entire dataset
df = pd.read_parquet("data/processed/master.parquet")

# Read specific columns
df = pd.read_parquet(
    "data/processed/master.parquet",
    columns=["session_id", "method", "logical_error_rate"]
)

# Filter while reading (requires pyarrow)
import pyarrow.parquet as pq
table = pq.read_table(
    "data/processed/master.parquet",
    filters=[("backend", "=", "ibm_brisbane"), ("method", "=", "adaptive")]
)
df = table.to_pandas()
```

## Validation

Schema validation is performed by `src/utils.py:validate_master_schema()`:

```python
from src.utils import validate_master_schema

df = pd.read_parquet("data/processed/master.parquet")
validate_master_schema(df)  # Raises ValueError if invalid
```
