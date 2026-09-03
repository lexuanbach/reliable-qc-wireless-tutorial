"""Search methods compared in the co-optimization study (plan sections 7-8).

Each optimizer receives a cost budget measured in fid2-equivalent evaluation
units and returns (selected_config, ledger). All randomness flows through the
supplied generator so runs are reproducible and paired across methods.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError:
    pass

import itertools

from coopt import (
    ContextModel,
    FIDELITY_COST,
    Ledger,
    Workload,
    encode_config,
    evaluate,
    hand_designed,
    mean_utility,
    sample_config,
    space_for,
    utility_from_components,
)

DRAWS = 3  # context draws per search-time evaluation
RESCORE_TOP = 3
RESCORE_DRAWS = 4
WINSOR = -6.0  # GP target clip: tail penalties otherwise wreck the surrogate


def _score(wl, cfg, cm, windows, rng, ledger, fidelity=2, draws=DRAWS,
           drop_terms=(), cvar_alpha=None):
    return mean_utility(wl, cfg, cm, windows, rng, fidelity, draws, ledger,
                        drop_terms, cvar_alpha)


def _reselect(wl, scored, cm, windows, rng, ledger, **kw):
    """Re-score the top observed configs with fresh draws before selection.

    Charged to the ledger for every method equally; controls winner's curse
    under noisy search-time evaluations.
    """
    scored = sorted(scored, key=lambda t: -t[1])[:RESCORE_TOP]
    best_cfg, best_u = scored[0][0], -np.inf
    for cfg, _ in scored:
        u = _score(wl, cfg, cm, windows, rng, ledger, draws=RESCORE_DRAWS, **kw)
        if u > best_u:
            best_u, best_cfg = u, cfg
    return best_cfg


def random_search(wl: Workload, cm: ContextModel, windows, budget: float,
                  rng: np.random.Generator, **kw):
    space = space_for(wl.kind)
    ledger = Ledger()
    scored = []
    while ledger.spent < budget:
        cfg = sample_config(space, rng)
        scored.append((cfg, _score(wl, cfg, cm, windows, rng, ledger, **kw)))
    return _reselect(wl, scored, cm, windows, rng, ledger, **kw), ledger


def random_search_no_rescore(wl: Workload, cm: ContextModel, windows,
                             budget: float, rng: np.random.Generator, **kw):
    """Joint-space-only ablation: fid2 random search without fresh re-scoring."""
    space = space_for(wl.kind)
    ledger = Ledger()
    scored = []
    while ledger.spent < budget:
        cfg = sample_config(space, rng)
        scored.append((cfg, _score(wl, cfg, cm, windows, rng, ledger, **kw)))
    return max(scored, key=lambda item: item[1])[0], ledger


def feasible_random_search(wl: Workload, cm: ContextModel, windows,
                           budget: float, rng: np.random.Generator, **kw):
    """Simple deadline/feasibility-screened random-search baseline.

    Batches are evaluated at analytic fidelity. The best candidate predicted
    to meet deadlines and budgets is promoted to fid2; if none is feasible,
    the highest-utility candidate is promoted. The final re-scoring stage is
    identical to every other search method and all screening is charged.
    """
    space = space_for(wl.kind)
    ledger = Ledger()
    scored = []
    while ledger.spent < budget:
        screened = []
        for _ in range(12):
            if ledger.spent >= budget:
                break
            cfg = sample_config(space, rng)
            ctx = cm.sample(rng, windows[rng.integers(len(windows))])
            comp = evaluate(wl, cfg, ctx, 0, rng, ledger)
            u = utility_from_components(comp, kw.get("drop_terms", ()))
            screened.append((cfg, u, comp["deadline_miss"], comp["infeasible"]))
        if not screened or ledger.spent >= budget:
            break
        feasible = [r for r in screened if r[2] == 0 and r[3] == 0]
        pick = max(feasible or screened, key=lambda r: r[1])[0]
        scored.append((pick, _score(wl, pick, cm, windows, rng, ledger, **kw)))
    if not scored:
        return random_search(wl, cm, windows, budget, rng, **kw)
    return _reselect(wl, scored, cm, windows, rng, ledger, **kw), ledger


def sequential_search(wl: Workload, cm: ContextModel, windows, budget: float,
                      rng: np.random.Generator, **kw):
    """Design-then-deploy: stage 1 optimizes quality only on a noiseless
    simulator; stage 2 fixes the design and tunes systems variables."""
    space = space_for(wl.kind)
    design_keys = [k for k in space if k not in
                   ("placement", "shots", "mitigation", "compile_level")]
    ledger = Ledger()
    # stage 1: quality-only, fid1, placement pinned to remote_sim
    best_design, best_q = None, np.inf
    while ledger.spent < 0.6 * budget:
        cfg = sample_config(space, rng)
        cfg["placement"] = "remote_sim"
        cfg["shots"] = 2048
        cfg["mitigation"] = 0
        ctx = cm.sample(rng, windows[rng.integers(len(windows))])
        comp = evaluate(wl, cfg, ctx, 1, rng, ledger)
        if comp["quality_loss"] < best_q:
            best_q = comp["quality_loss"]
            best_design = {k: cfg[k] for k in design_keys}
    # stage 2: tune systems variables with the design frozen
    scored = []
    while ledger.spent < budget:
        cfg = sample_config(space, rng)
        cfg.update(best_design)
        scored.append((cfg, _score(wl, cfg, cm, windows, rng, ledger, **kw)))
    return _reselect(wl, scored, cm, windows, rng, ledger, **kw), ledger


class _GP:
    """Thin wrapper around sklearn GP regression for the surrogate."""

    def __init__(self):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel
        kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(1e-2)
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                           alpha=1e-6, n_restarts_optimizer=1,
                                           random_state=0)

    def fit(self, X, y):
        self.gp.fit(np.array(X), np.array(y))

    def predict(self, X):
        mu, sd = self.gp.predict(np.array(X), return_std=True)
        return mu, np.maximum(sd, 1e-9)


class _RF:
    """Categorical-friendly extra-trees surrogate with ensemble uncertainty."""

    def __init__(self):
        from sklearn.ensemble import ExtraTreesRegressor
        self.rf = ExtraTreesRegressor(
            n_estimators=64, min_samples_leaf=2, max_features=0.8,
            bootstrap=True, random_state=0, n_jobs=1)

    def fit(self, X, y):
        self.rf.fit(np.asarray(X), np.asarray(y))

    def predict(self, X):
        X = np.asarray(X)
        preds = np.asarray([tree.predict(X) for tree in self.rf.estimators_])
        return preds.mean(0), np.maximum(preds.std(0), 1e-6)


def _expected_improvement(mu, sd, best):
    from scipy.stats import norm
    z = (mu - best) / sd
    return sd * (z * norm.cdf(z) + norm.pdf(z))


def joint_bo(wl: Workload, cm: ContextModel, windows, budget: float,
             rng: np.random.Generator, multi_fidelity: bool = False,
             init_data: tuple | None = None, rescore: bool = True,
             winsor_floor: float | None = WINSOR, **kw):
    """Joint Bayesian optimization over the full configuration space.

    multi_fidelity=True screens EI candidates through analytic (fid0) and
    noiseless (fid1) checks before spending fid2 evaluations (plan 7.1).
    """
    space = space_for(wl.kind)
    ledger = Ledger()
    X, y, cfgs = [], [], []
    if init_data is not None:
        X0, y0, c0 = init_data
        X.extend(X0); y.extend(y0); cfgs.extend(c0)
    n_init = 8
    while len(y) < n_init and ledger.spent < budget:
        cfg = sample_config(space, rng)
        u = _score(wl, cfg, cm, windows, rng, ledger, **kw)
        X.append(encode_config(space, cfg)); y.append(u); cfgs.append(cfg)
    while ledger.spent < budget:
        gp = _GP()
        raw_targets = np.array(y)
        yw = raw_targets if winsor_floor is None else np.maximum(raw_targets, winsor_floor)
        gp.fit(X, yw)
        cands = [sample_config(space, rng) for _ in range(256)]
        Xc = np.array([encode_config(space, c) for c in cands])
        mu, sd = gp.predict(Xc)
        ei = _expected_improvement(mu, sd, float(yw.max()))
        order = np.argsort(-ei)
        if multi_fidelity:
            # screen top-9 by EI at fid0, top-3 survivors at fid1, best at fid2
            short = [cands[i] for i in order[:9]]
            s0 = [(c, _score(wl, c, cm, windows, rng, ledger, fidelity=0,
                             draws=1, **kw)) for c in short]
            s0.sort(key=lambda t: -t[1])
            s1 = [(c, _score(wl, c, cm, windows, rng, ledger, fidelity=1,
                             draws=1, **kw)) for c, _ in s0[:3]]
            s1.sort(key=lambda t: -t[1])
            pick = s1[0][0]
        else:
            pick = cands[int(order[0])]
        u = _score(wl, pick, cm, windows, rng, ledger, **kw)
        X.append(encode_config(space, pick)); y.append(u); cfgs.append(pick)
    pick = _reselect(wl, list(zip(cfgs, y)), cm, windows, rng, ledger, **kw) \
        if rescore else cfgs[int(np.argmax(y))]
    return pick, ledger, (X, y, cfgs)


def chance_constrained_search(wl: Workload, cm: ContextModel, windows,
                              budget: float, rng: np.random.Generator,
                              risk_limit: float = 0.10, **kw):
    """CAPO variant with an empirical deadline/infeasibility chance constraint.

    Analytic draws screen the joint space; promoted candidates use fid2. The
    finalists are re-evaluated on fresh contexts and selected by utility among
    candidates whose empirical violation rate does not exceed ``risk_limit``.
    """
    space = space_for(wl.kind)
    ledger = Ledger()
    scored = []
    while ledger.spent < budget:
        cfg = sample_config(space, rng)
        analytic = []
        for _ in range(5):
            ctx = cm.sample(rng, windows[rng.integers(len(windows))])
            analytic.append(evaluate(wl, cfg, ctx, 0, rng, ledger))
        predicted_risk = np.mean([max(c["deadline_miss"], c["infeasible"])
                                  for c in analytic])
        if predicted_risk > risk_limit:
            continue
        utility = _score(wl, cfg, cm, windows, rng, ledger, fidelity=2,
                         draws=DRAWS, **kw)
        scored.append((cfg, utility))
    if not scored:
        return feasible_random_search(wl, cm, windows, budget, rng, **kw)
    finalists = sorted(scored, key=lambda item: -item[1])[:RESCORE_TOP]
    rescored = []
    for cfg, _ in finalists:
        comps = []
        for _ in range(RESCORE_DRAWS):
            ctx = cm.sample(rng, windows[rng.integers(len(windows))])
            comps.append(evaluate(wl, cfg, ctx, 2, rng, ledger))
        risk = float(np.mean([max(c["deadline_miss"], c["infeasible"])
                              for c in comps]))
        utility = float(np.mean([utility_from_components(c, kw.get("drop_terms", ()))
                                 for c in comps]))
        rescored.append((cfg, utility, risk))
    admissible = [row for row in rescored if row[2] <= risk_limit]
    return max(admissible or rescored, key=lambda row: row[1] - 100.0 * row[2])[0], ledger


def device_aware_sequential(wl: Workload, cm: ContextModel, windows,
                            budget: float, rng: np.random.Generator, **kw):
    """Modular baseline: noisy-device-aware design, then service scheduling."""
    space = space_for(wl.kind)
    design_keys = [k for k in space if k not in
                   ("placement", "shots", "mitigation", "compile_level")]
    ledger = Ledger(); best_design = None; best_quality = np.inf
    while ledger.spent < 0.6 * budget:
        cfg = sample_config(space, rng)
        cfg["placement"] = "qpu_B"; cfg["compile_level"] = 2
        ctx = cm.sample(rng, windows[rng.integers(len(windows))])
        comp = evaluate(wl, cfg, ctx, 2, rng, ledger)
        if comp["quality_loss"] < best_quality:
            best_quality = comp["quality_loss"]
            best_design = {k: cfg[k] for k in design_keys}
    scored = []
    while ledger.spent < budget:
        cfg = sample_config(space, rng); cfg.update(best_design)
        scored.append((cfg, _score(wl, cfg, cm, windows, rng, ledger, **kw)))
    return _reselect(wl, scored, cm, windows, rng, ledger, **kw), ledger


def fixed_design_scheduler(wl: Workload, cm: ContextModel, windows,
                           budget: float, rng: np.random.Generator, **kw):
    """Modular baseline: queue-aware execution search with a fixed design."""
    space = space_for(wl.kind)
    design_keys = [k for k in space if k not in
                   ("placement", "shots", "mitigation", "compile_level")]
    fixed = hand_designed(wl.kind)
    scored = []; ledger = Ledger()
    while ledger.spent < budget:
        cfg = sample_config(space, rng)
        cfg.update({k: fixed[k] for k in design_keys})
        scored.append((cfg, _score(wl, cfg, cm, windows, rng, ledger, **kw)))
    return _reselect(wl, scored, cm, windows, rng, ledger, **kw), ledger


def joint_rf(wl: Workload, cm: ContextModel, windows, budget: float,
             rng: np.random.Generator, **kw):
    """Joint search with a categorical-friendly extra-trees surrogate."""
    space = space_for(wl.kind)
    ledger = Ledger()
    X, y, cfgs = [], [], []
    while len(y) < 8 and ledger.spent < budget:
        cfg = sample_config(space, rng)
        X.append(encode_config(space, cfg))
        y.append(_score(wl, cfg, cm, windows, rng, ledger, **kw))
        cfgs.append(cfg)
    while ledger.spent < budget:
        model = _RF()
        model.fit(X, np.maximum(np.asarray(y), WINSOR))
        cands = [sample_config(space, rng) for _ in range(256)]
        mu, sd = model.predict([encode_config(space, c) for c in cands])
        pick = cands[int(np.argmax(mu + 0.75 * sd))]
        X.append(encode_config(space, pick))
        y.append(_score(wl, pick, cm, windows, rng, ledger, **kw))
        cfgs.append(pick)
    return _reselect(wl, list(zip(cfgs, y)), cm, windows, rng, ledger, **kw), ledger


def accuracy_only_bo(wl: Workload, cm: ContextModel, windows, budget: float,
                     rng: np.random.Generator, **kw):
    """BO that optimizes task quality alone; systems terms ignored during
    search (plan section 8 'accuracy-only optimization')."""
    space = space_for(wl.kind)
    ledger = Ledger()
    X, y, cfgs = [], [], []
    while ledger.spent < budget:
        if len(y) < 8:
            cfg = sample_config(space, rng)
        else:
            gp = _GP(); gp.fit(X, y)
            cands = [sample_config(space, rng) for _ in range(256)]
            mu, sd = gp.predict([encode_config(space, c) for c in cands])
            cfg = cands[int(np.argmax(_expected_improvement(mu, sd, max(y))))]
        ctx = cm.sample(rng, windows[rng.integers(len(windows))])
        comp = evaluate(wl, cfg, ctx, 2, rng, ledger)
        X.append(encode_config(space, cfg))
        y.append(-comp["quality_loss"])
        cfgs.append(cfg)
    order = np.argsort(y)[::-1][:RESCORE_TOP]
    best_cfg, best_q = cfgs[int(order[0])], -np.inf
    for i in order:
        ctx = cm.sample(rng, windows[rng.integers(len(windows))])
        comp = evaluate(wl, cfgs[int(i)], ctx, 2, rng, ledger)
        if -comp["quality_loss"] > best_q:
            best_q, best_cfg = -comp["quality_loss"], cfgs[int(i)]
    return best_cfg, ledger


def exhaustive_f0_topk(wl: Workload, cm: ContextModel, windows, budget: float,
                       rng: np.random.Generator, **kw):
    """Exhaustive analytic sweep with top-k promotion (revision arm E1).

    Every configuration in the joint space is scored once at analytic fidelity
    on a sampled context; the top ``RESCORE_TOP`` configurations are promoted
    to the same charged fid2 re-scoring stage every other method uses. The
    sweep itself is charged in full, so realized spend exceeds the proposal
    budget on the larger spaces (reported in the cost ledger, not hidden).
    """
    space = space_for(wl.kind)
    ledger = Ledger()
    keys = list(space)
    # common random numbers: every configuration is scored on the same
    # panel of one context per training window, so the sweep ranks by
    # analytic-model differences rather than per-config context luck
    panel = [cm.sample(rng, w) for w in windows]
    drop = kw.get("drop_terms", ())
    scored = []
    for values in itertools.product(*(space[k] for k in keys)):
        cfg = dict(zip(keys, values))
        u = np.mean([utility_from_components(
            evaluate(wl, cfg, ctx, 0, rng, ledger), drop) for ctx in panel])
        scored.append((cfg, float(u)))
    return _reselect(wl, scored, cm, windows, rng, ledger, **kw), ledger


def device_aware_fallback(wl: Workload, cm: ContextModel, windows,
                          budget: float, rng: np.random.Generator, **kw):
    """Device-aware sequential comparator with a post-hoc classical escape
    hatch (revision arm E2): after the comparator commits, the classical-
    placement variant of its winner is scored on fresh charged draws and kept
    when it wins. Isolates how much of the comparator's loss is placement
    versus its frozen stage-1 design."""
    cfg, ledger = device_aware_sequential(wl, cm, windows, budget, rng, **kw)
    alt = dict(cfg)
    alt["placement"] = "classical"
    u_sel = _score(wl, cfg, cm, windows, rng, ledger, draws=RESCORE_DRAWS, **kw)
    u_alt = _score(wl, alt, cm, windows, rng, ledger, draws=RESCORE_DRAWS, **kw)
    return (alt if u_alt > u_sel else cfg), ledger


def random_search_no_rescore_equal_spend(wl: Workload, cm: ContextModel,
                                         windows, budget: float,
                                         rng: np.random.Generator, **kw):
    """Equal-total-spend contrast for re-scoring (revision arm E3): the
    no-rescore arm receives the re-scoring stage's fid2 spend as additional
    proposal budget, so the two procedures are compared at the same total
    ledger charge rather than at the proposal-loop threshold alone."""
    extra = RESCORE_TOP * RESCORE_DRAWS * FIDELITY_COST[2]
    return random_search_no_rescore(wl, cm, windows, budget + extra, rng, **kw)


def tpe_search(wl: Workload, cm: ContextModel, windows, budget: float,
               rng: np.random.Generator, **kw):
    """Tree-structured Parzen Estimator engine inside the CAPO contract,
    using the original authors' implementation (Hyperopt, Bergstra et al.):
    ``hyperopt.tpe.suggest`` proposes configurations over an ``hp.choice``
    space; every proposal is scored by the shared charged evaluator and the
    standard re-scored selection returns the winner."""
    from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
    space = space_for(wl.kind)
    hp_space = {k: hp.choice(k, v) for k, v in space.items()}
    ledger = Ledger()
    cfgs, scores = [], []

    def objective(cfg):
        u = _score(wl, cfg, cm, windows, rng, ledger, **kw)
        cfgs.append(dict(cfg)); scores.append(u)
        return {"loss": -u, "status": STATUS_OK}

    def out_of_budget(trials, *args):
        return ledger.spent >= budget, args

    fmin(objective, hp_space, algo=tpe.suggest, max_evals=10_000,
         trials=Trials(), rstate=np.random.default_rng(rng.integers(2**31)),
         early_stop_fn=out_of_budget, show_progressbar=False, verbose=False)
    return _reselect(wl, list(zip(cfgs, scores)), cm, windows, rng, ledger,
                     **kw), ledger


def successive_halving_search(wl: Workload, cm: ContextModel, windows,
                              budget: float, rng: np.random.Generator, **kw):
    """Successive-halving / Hyperband-style engine (Jamieson & Talwalkar,
    2016; Li et al., 2018) inside the CAPO contract: brackets of random
    configurations climb the fidelity ladder as the resource, with a third
    surviving each rung. Distinct from CAPO-MF, which screens
    expected-improvement proposals rather than whole random brackets."""
    space = space_for(wl.kind)
    ledger = Ledger()
    scored = []
    while ledger.spent < budget:
        bracket = [sample_config(space, rng) for _ in range(18)]
        r0 = [(c, _score(wl, c, cm, windows, rng, ledger, fidelity=0,
                         draws=1, **kw)) for c in bracket]
        r0.sort(key=lambda t: -t[1])
        r1 = [(c, _score(wl, c, cm, windows, rng, ledger, fidelity=1,
                         draws=1, **kw)) for c, _ in r0[:6]]
        r1.sort(key=lambda t: -t[1])
        for c, _ in r1[:2]:
            scored.append((c, _score(wl, c, cm, windows, rng, ledger, **kw)))
    if not scored:
        return random_search(wl, cm, windows, budget, rng, **kw)
    return _reselect(wl, scored, cm, windows, rng, ledger, **kw), ledger


def fixed_policy(name: str, wl: Workload) -> dict:
    cfg = hand_designed(wl.kind)
    if name == "always_simulator":
        cfg["placement"] = "local_sim"
    elif name == "classical_only":
        cfg["placement"] = "classical"
    elif name == "fixed_qpu_A":
        cfg["placement"] = "qpu_A"
    return cfg


def oracle_placement(wl: Workload, base_cfg: dict, cm: ContextModel, windows,
                     rng: np.random.Generator) -> dict:
    """Upper bound: picks the placement with the best realized fid3 utility."""
    from coopt import PLACEMENTS
    best_cfg, best_u = None, -np.inf
    for pl in PLACEMENTS:
        cfg = dict(base_cfg); cfg["placement"] = pl
        u = mean_utility(wl, cfg, cm, windows, np.random.default_rng(rng.integers(2**31)),
                         3, 10)
        if u > best_u:
            best_u, best_cfg = u, cfg
    return best_cfg


METHODS_MAIN = ["random", "sequential", "joint_bo", "joint_mf", "accuracy_only",
                "hand_designed", "always_simulator", "classical_only",
                "feasible_random", "joint_rf", "capo_joint_only",
                "capo_mf_no_rescore", "capo_chance", "device_aware_sequential",
                "fixed_design_scheduler", "exhaustive_f0",
                "device_aware_fallback", "capo_joint_only_espend",
                "tpe", "successive_halving"]


def run_method(name: str, wl: Workload, cm: ContextModel, windows, budget: float,
               rng: np.random.Generator, **kw):
    """Dispatch. Returns (config, ledger)."""
    if name == "random":
        return random_search(wl, cm, windows, budget, rng, **kw)
    if name == "capo_joint_only":
        return random_search_no_rescore(wl, cm, windows, budget, rng, **kw)
    if name == "feasible_random":
        return feasible_random_search(wl, cm, windows, budget, rng, **kw)
    if name == "sequential":
        return sequential_search(wl, cm, windows, budget, rng, **kw)
    if name == "joint_bo":
        cfg, led, _ = joint_bo(wl, cm, windows, budget, rng, **kw)
        return cfg, led
    if name == "joint_mf":
        cfg, led, _ = joint_bo(wl, cm, windows, budget, rng, multi_fidelity=True, **kw)
        return cfg, led
    if name == "capo_mf_no_rescore":
        cfg, led, _ = joint_bo(wl, cm, windows, budget, rng,
                               multi_fidelity=True, rescore=False, **kw)
        return cfg, led
    if name == "capo_chance":
        return chance_constrained_search(wl, cm, windows, budget, rng, **kw)
    if name == "device_aware_sequential":
        return device_aware_sequential(wl, cm, windows, budget, rng, **kw)
    if name == "fixed_design_scheduler":
        return fixed_design_scheduler(wl, cm, windows, budget, rng, **kw)
    if name == "tpe":
        return tpe_search(wl, cm, windows, budget, rng, **kw)
    if name == "successive_halving":
        return successive_halving_search(wl, cm, windows, budget, rng, **kw)
    if name == "exhaustive_f0":
        return exhaustive_f0_topk(wl, cm, windows, budget, rng, **kw)
    if name == "device_aware_fallback":
        return device_aware_fallback(wl, cm, windows, budget, rng, **kw)
    if name == "capo_joint_only_espend":
        return random_search_no_rescore_equal_spend(wl, cm, windows, budget, rng, **kw)
    if name == "joint_rf":
        return joint_rf(wl, cm, windows, budget, rng, **kw)
    if name == "accuracy_only":
        return accuracy_only_bo(wl, cm, windows, budget, rng, **kw)
    if name in ("hand_designed", "always_simulator", "classical_only", "fixed_qpu_A"):
        return fixed_policy(name, wl), Ledger()
    raise ValueError(name)
