#!/usr/bin/env python3
"""Empirically verify Assumption 2 (first-order stochastic dominance of the
trained circuit distribution over uniform, in cost) for every circuit-instance
pair the experiments use: warm-start family parameters at p in {1,2,3} on all
three task families and the 12 experiment seeds. Emits a macro file."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from coopt import GENERATORS, qaoa_probs, warm_params
from networkqbench import _normalized_cost

checked = violated = 0
worst = 0.0
for task in ("routing", "channel", "placement"):
    for seed in range(12):
        inst = GENERATORS[task](12, seed)
        norm = _normalized_cost(inst.costs)
        order = np.argsort(inst.costs)
        for p in (1, 2, 3):
            theta = warm_params(task, 12, p)
            probs = qaoa_probs(norm, inst.n_vars, theta)
            # FOSD in cost: cumulative prob of sorted-by-cost states must
            # dominate the uniform cumulative at every threshold.
            c_pi = np.cumsum(probs[order])
            c_u = np.arange(1, len(order) + 1) / len(order)
            gap = float((c_u - c_pi).max())  # >0 would violate dominance
            checked += 1
            worst = max(worst, gap)
            if gap > 1e-12:
                violated += 1
                print(f"VIOLATION {task} seed={seed} p={p} gap={gap:.2e}")
print(f"checked={checked} violated={violated} worst_gap={worst:.3e}")
out = Path(__file__).resolve().parents[1] / "submission/paper/generated/assumption.tex"
out.write_text(
    f"\\newcommand{{\\AsmChecked}}{{{checked}}}\n"
    f"\\newcommand{{\\AsmViolated}}{{{violated}}}\n")
