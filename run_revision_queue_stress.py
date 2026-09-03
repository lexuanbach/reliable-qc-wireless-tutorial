#!/usr/bin/env python3
"""Replay selected configurations under a published quantum-cloud queue envelope.

Ravi et al. report more than 30% of IBM cloud jobs waiting over two hours and
about 10% waiting at least one day.  Without access to their raw trace, this
script uses a declared 20-point envelope satisfying those reported quantiles:
30% at 30 s, 35% at 30 min, 25% at 4 h, and 10% at 24 h.  It is a
literature-calibrated stress replay, not a provider trace or current SLA.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from coopt import ContextModel, Workload, evaluate, utility_from_components


ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
MAIN_FAMILIES = ["qaoa", "qaoa_channel", "qaoa_place", "vqc", "vqc_cancer"]
TEST_WINDOWS = list(range(9))


def final_eval(wl, cfg, cm, windows, seed, draws=20):
    rng = np.random.default_rng(777_000 + seed)
    utils, comps = [], []
    for i in range(draws):
        ctx = cm.sample(rng, windows[i % len(windows)])
        comp = evaluate(wl, cfg, ctx, 3, rng)
        utils.append(utility_from_components(comp))
        comps.append(comp)
    keys = ("quality_loss", "delay_s", "deadline_miss", "infeasible", "energy_j",
            "money", "cost_scalar", "realized_fail")
    result = {key: float(np.mean([comp[key] for comp in comps])) for key in keys}
    result["utility"] = float(np.mean(utils))
    result["utility_p10"] = float(np.percentile(utils, 10))
    return result


class PublishedQueueEnvelope(ContextModel):
    values = np.asarray([30.0] * 6 + [1800.0] * 7 + [14_400.0] * 5
                        + [86_400.0] * 2)

    def sample(self, rng: np.random.Generator, window: int):
        ctx = super().sample(rng, window)
        queue = float(self.values[rng.integers(len(self.values))])
        # Backend B is the lower-queue alternative but remains in the same
        # stress regime; this preserves heterogeneous backend choice.
        ctx.queue_s["qpu_A"] = queue
        ctx.queue_s["qpu_B"] = 0.5 * queue
        return ctx


def load_exp1() -> pd.DataFrame:
    data = pd.concat([pd.read_csv(path) for path in sorted(RES.glob("raw_main_*.csv"))],
                     ignore_index=True)
    data = data[(data.experiment == "exp1") & data.workload.isin(MAIN_FAMILIES)]
    return data.drop_duplicates(["workload", "seed", "method"], keep="last")


def main() -> int:
    source = load_exp1()
    cfg_cols = [c for c in source if c.startswith("cfg_")]
    rows = []
    for row in source.itertuples(index=False):
        record = row._asdict()
        cfg = {col[4:]: record[col] for col in cfg_cols
               if col in record and pd.notna(record[col])}
        for key, value in list(cfg.items()):
            if key != "placement" and isinstance(value, (float, np.floating)) \
                    and float(value).is_integer():
                cfg[key] = int(value)
        wl = Workload.make(record["workload"], int(record["seed"]))
        cm = PublishedQueueEnvelope(seed=500 + int(record["seed"]))
        agg = final_eval(wl, cfg, cm, TEST_WINDOWS, int(record["seed"]), draws=20)
        rows.append({"experiment": "revision_queue_envelope",
                     "workload": record["workload"], "seed": int(record["seed"]),
                     "method": record["method"],
                     "cfg_placement": cfg["placement"], **agg})
        print(f"queue-envelope {record['workload']} seed={record['seed']} "
              f"{record['method']}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "raw_revision_queue_envelope.csv", index=False)
    print(f"wrote {len(out)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
