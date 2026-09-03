"""Sanity tests for the co-optimization models."""
import numpy as np
import coopt
from coopt import ContextModel, Workload, evaluate, sample_config, space_for
from search import run_method
from amortization_ledger import break_even


def test_context_drift_is_deterministic():
    a = ContextModel(seed=1)
    b = ContextModel(seed=1)
    ra = a.sample(np.random.default_rng(0), 3)
    rb = b.sample(np.random.default_rng(0), 3)
    assert ra.eps == rb.eps and ra.queue_s == rb.queue_s


def test_qaoa_eval_components():
    wl = Workload.make("qaoa", 0)
    cm = ContextModel(seed=2)
    ctx = cm.sample(np.random.default_rng(1), 0)
    for placement in coopt.PLACEMENTS:
        cfg = coopt.hand_designed("qaoa")
        cfg["placement"] = placement
        comp = evaluate(wl, cfg, ctx, 2, np.random.default_rng(3))
        assert comp["quality_loss"] >= 0 and comp["delay_s"] > 0
        assert np.isfinite(comp["utility"])


def test_noise_hurts_quality_on_average():
    wl = Workload.make("qaoa", 1)
    cm = ContextModel(seed=3)
    cfg = coopt.hand_designed("qaoa")
    cfg["placement"] = "qpu_B"
    g_ideal, g_noisy = [], []
    for i in range(10):
        ctx = cm.sample(np.random.default_rng(i), 0)
        g_ideal.append(evaluate(wl, cfg, ctx, 1, np.random.default_rng(50 + i))["quality_loss"])
        g_noisy.append(evaluate(wl, cfg, ctx, 2, np.random.default_rng(50 + i))["quality_loss"])
    assert np.mean(g_noisy) >= np.mean(g_ideal) - 1e-9


def test_vqc_learns_better_than_chance():
    wl = Workload.make("vqc", 0)
    cm = ContextModel(seed=4)
    ctx = cm.sample(np.random.default_rng(5), 0)
    cfg = coopt.hand_designed("vqc")
    cfg["placement"] = "remote_sim"
    comp = evaluate(wl, cfg, ctx, 1, np.random.default_rng(6))
    assert comp["quality_loss"] < 0.45, comp


def test_ledger_accumulates():
    wl = Workload.make("vqc", 1)
    cm = ContextModel(seed=5)
    led = coopt.Ledger()
    cfg = sample_config(space_for("vqc"), np.random.default_rng(7))
    ctx = cm.sample(np.random.default_rng(8), 0)
    evaluate(wl, cfg, ctx, 2, np.random.default_rng(9), led)
    evaluate(wl, cfg, ctx, 0, np.random.default_rng(9), led)
    assert abs(led.spent - 1.02) < 1e-9
    assert led.evaluations == 2 and len(led.unique_configs) == 1
    assert led.fidelity_counts == {0: 1, 2: 1}


def test_new_revision_baselines_return_configs():
    wl = Workload.make("qaoa", 0)
    cm = ContextModel(seed=2)
    for method in ("feasible_random", "joint_rf"):
        cfg, ledger = run_method(method, wl, cm, [0, 1], 10.0,
                                 np.random.default_rng(91))
        assert cfg["placement"] in coopt.PLACEMENTS
        assert ledger.spent > 0


def test_clearaccept_method_variants_return_configs():
    wl = Workload.make("qaoa", 4)
    cm = ContextModel(seed=8)
    methods = ("capo_joint_only", "capo_mf_no_rescore", "capo_chance",
               "device_aware_sequential", "fixed_design_scheduler")
    for method in methods:
        cfg, ledger = run_method(method, wl, cm, [0, 1, 2], 8.0,
                                 np.random.default_rng(110 + len(method)))
        assert cfg["placement"] in coopt.PLACEMENTS
        assert ledger.evaluations > 0 and ledger.spent > 0


def test_standalone_amortization_ledger():
    result = break_even(105.0, 5.0, 0.25, 10.0)
    assert result["incremental_search_s"] == 100.0
    assert result["break_even_uses"] == 40
    assert break_even(10.0, 0.0, 0.0, 1.0)["break_even_uses"] is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")
