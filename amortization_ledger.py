#!/usr/bin/env python3
"""Standalone CAPO break-even ledger with stable JSON output."""

from __future__ import annotations

import argparse
import json
import math


def break_even(search_cost_s: float, baseline_search_cost_s: float,
               utility_gain: float, value_per_utility_s: float) -> dict:
    incremental_search_s = max(0.0, search_cost_s - baseline_search_cost_s)
    value_per_use_s = utility_gain * value_per_utility_s
    uses = math.inf if value_per_use_s <= 0 else math.ceil(incremental_search_s / value_per_use_s)
    return {
        "search_cost_s": search_cost_s,
        "baseline_search_cost_s": baseline_search_cost_s,
        "incremental_search_s": incremental_search_s,
        "utility_gain_per_use": utility_gain,
        "value_per_utility_s": value_per_utility_s,
        "value_gain_per_use_s": value_per_use_s,
        "break_even_uses": None if math.isinf(uses) else uses,
        "amortizes": not math.isinf(uses),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-cost-s", type=float, required=True)
    parser.add_argument("--baseline-search-cost-s", type=float, required=True)
    parser.add_argument("--utility-gain", type=float, required=True)
    parser.add_argument("--value-per-utility-s", type=float, default=1.0)
    args = parser.parse_args()
    print(json.dumps(break_even(args.search_cost_s, args.baseline_search_cost_s,
                                args.utility_gain, args.value_per_utility_s), indent=2))


if __name__ == "__main__":
    main()
