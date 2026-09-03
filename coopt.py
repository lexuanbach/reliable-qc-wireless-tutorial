"""Core models for QPU-aware circuit/hyperparameter/placement co-optimization.

Everything quantum in this module is an exact statevector simulation with an
explicit global-depolarizing + shot-noise degradation model conditioned on a
synthetic calibration state. No value here is a hardware measurement; every
latency/energy/money figure is a labeled model. Deployment ("fid3") is a
held-out noisy simulation with perturbed calibration, standing in for the
plan's hardware-validation stage.

The artifact bundles the workload generators and classical solvers that it
uses, so the released snapshot runs without a sibling research project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import sys
import time
from pathlib import Path

import numpy as np

_NQB = Path(__file__).resolve().parent
sys.path.insert(0, str(_NQB))

from networkqbench import (  # noqa: E402
    GENERATORS,
    Instance,
    _apply_mixer,
    _normalized_cost,
    annealing_solver,
    exact_solver,
)

# Larger routing instances for the scale study (additive; sizes <=12 untouched).
import networkqbench as _nqb  # noqa: E402

_ORIG_ROUTING_DIMS = _nqb._routing_dims
_nqb._routing_dims = lambda size: {16: (4, 4), 20: (4, 5)}.get(size) \
    or _ORIG_ROUTING_DIMS(size)

# --------------------------------------------------------------------------
# Configuration space (plan section 5)
# --------------------------------------------------------------------------

PLACEMENTS = ["local_sim", "remote_sim", "qpu_A", "qpu_B", "classical"]
SHOTS_CHOICES = [128, 512, 2048]
COMPILE_LEVELS = [0, 1, 2]
COMPILE_GATE_FACTOR = {0: 1.00, 1: 0.85, 2: 0.72}
COMPILE_TIME_S = {0: 0.01, 1: 0.05, 2: 0.20}

QAOA_SPACE = {
    "p": [1, 2, 3],
    "opt_iters": [20, 40, 80],
    "warm_start": [0, 1],
    "shots": SHOTS_CHOICES,
    "mitigation": [0, 1],
    "compile_level": COMPILE_LEVELS,
    "placement": PLACEMENTS,
}

# Widened space for the scale study (|Z| = 12000).
QAOA_SPACE_LARGE = {
    "p": [1, 2, 3, 4],
    "opt_iters": [10, 20, 40, 80, 160],
    "warm_start": [0, 1],
    "shots": [64, 128, 512, 2048, 8192],
    "mitigation": [0, 1],
    "compile_level": COMPILE_LEVELS,
    "placement": PLACEMENTS,
}

VQC_SPACE = {
    "n_qubits": [2, 4, 6],
    "layers": [1, 2, 3],
    "reupload": [0, 1],
    "train_iters": [30, 60, 120],
    "lr": [0.05, 0.15],
    "shots": SHOTS_CHOICES,
    "mitigation": [0, 1],
    "compile_level": COMPILE_LEVELS,
    "placement": PLACEMENTS,
}


def space_for(workload: str) -> dict:
    if workload == "qaoa_large":
        return QAOA_SPACE_LARGE
    return QAOA_SPACE if workload.startswith("qaoa") else VQC_SPACE


def sample_config(space: dict, rng: np.random.Generator) -> dict:
    return {k: v[rng.integers(len(v))] for k, v in space.items()}


def encode_config(space: dict, cfg: dict) -> np.ndarray:
    feats: list[float] = []
    for key, choices in space.items():
        if key == "placement":
            feats.extend(1.0 if cfg[key] == p else 0.0 for p in PLACEMENTS)
        else:
            vals = np.asarray(choices, dtype=float)
            v = float(cfg[key])
            feats.append((v - vals.min()) / (vals.max() - vals.min() + 1e-12))
    return np.array(feats)


def hand_designed(workload: str) -> dict:
    """Hardware-efficient default a practitioner might pick (plan section 8)."""
    if workload.startswith("qaoa"):
        return {"p": 1, "opt_iters": 40, "warm_start": 0, "shots": 512,
                "mitigation": 0, "compile_level": 1, "placement": "qpu_A"}
    return {"n_qubits": 4, "layers": 2, "reupload": 0, "train_iters": 60,
            "lr": 0.05, "shots": 512, "mitigation": 0, "compile_level": 1,
            "placement": "qpu_A"}


# --------------------------------------------------------------------------
# Context model: calibration drift, queues, network (plan sections 5 and 7.2)
# --------------------------------------------------------------------------

BACKEND_BASE = {
    # eps_2q, queue_s, per_circuit_s, per_shot_s, money_per_shot
    "qpu_A": (0.004, 2.0, 0.010, 2.0e-4, 3.0e-6),
    "qpu_B": (0.012, 0.3, 0.008, 1.0e-4, 1.0e-6),
}
EDGE_FLOPS = 5.0e5      # modeled edge-CPU amplitude-ops/s for local statevector
REMOTE_FLOPS = 2.0e9    # modeled datacenter simulator rate
POWER_W = {"local_sim": 8.0, "remote_sim": 250.0, "classical": 8.0}
QPU_ENERGY_PER_CIRCUIT_J = 5.0
QPU_ENERGY_PER_SHOT_J = 1.0e-3
REMOTE_SIM_MONEY_PER_S = 1.0e-5
MITIGATION_SHOT_MULTIPLIER = 3.0
MITIGATION_ERROR_MULTIPLIER = 0.5


@dataclass
class Context:
    window: int
    eps: dict            # backend -> two-qubit error rate (calibration state)
    queue_s: dict        # backend -> queue wait
    rtt_s: float
    fail_prob: float     # per-roundtrip network failure probability
    deadline_s: float
    energy_budget_j: float
    money_budget: float
    volatility: float
    reuse_count: int


class ContextModel:
    """Windowed drift process for calibration, queue and network state."""

    def __init__(self, seed: int, n_windows: int = 9):
        rng = np.random.default_rng(seed)
        self.n_windows = n_windows
        self.walk = {}
        for b in BACKEND_BASE:
            steps = rng.normal(0.0, 0.35, size=n_windows)
            self.walk[b] = np.cumsum(steps) - steps[0]
        self.queue_walk = {
            b: np.cumsum(rng.normal(0.0, 0.4, size=n_windows)) for b in BACKEND_BASE
        }
        self.rtt_walk = np.cumsum(rng.normal(0.0, 0.25, size=n_windows))

    def sample(self, rng: np.random.Generator, window: int) -> Context:
        w = int(np.clip(window, 0, self.n_windows - 1))
        eps, queue = {}, {}
        for b, (e0, q0, _, _, _) in BACKEND_BASE.items():
            eps[b] = float(np.clip(e0 * math.exp(self.walk[b][w] + rng.normal(0, 0.10)),
                                   5e-4, 0.08))
            queue[b] = float(np.clip(q0 * math.exp(self.queue_walk[b][w] + rng.normal(0, 0.25)),
                                     0.02, 60.0))
        return Context(
            window=w,
            eps=eps,
            queue_s=queue,
            rtt_s=float(np.clip(0.03 * math.exp(self.rtt_walk[w] + rng.normal(0, 0.2)),
                                0.005, 1.0)),
            fail_prob=float(rng.uniform(1e-4, 3e-3)),
            deadline_s=float(np.exp(rng.normal(math.log(1.5), 0.9))),
            energy_budget_j=float(rng.uniform(30.0, 400.0)),
            money_budget=float(rng.uniform(0.005, 0.10)),
            volatility=float(rng.uniform(0.01, 0.12)),
            reuse_count=int(np.exp(rng.uniform(0, 7))),
        )


# --------------------------------------------------------------------------
# Utility (plan section 6). Components stored so analysis can re-scalarize.
# --------------------------------------------------------------------------

W_DEADLINE = 2.0
W_INFEASIBLE = 5.0
W_MONEY = 0.5
W_ENERGY = 0.3
MONEY_SCALE = 0.01
ENERGY_SCALE = 50.0


def utility_from_components(comp: dict, drop_terms: tuple = ()) -> float:
    quality = comp["quality_loss"]
    stale = 0.0 if "staleness" in drop_terms else comp["staleness"]
    dl = 0.0 if "deadline" in drop_terms else W_DEADLINE * comp["deadline_miss"]
    fail = 0.0 if "failure" in drop_terms else W_INFEASIBLE * comp["infeasible"]
    money = 0.0 if "money" in drop_terms else W_MONEY * comp["money"] / MONEY_SCALE
    energy = 0.0 if "energy" in drop_terms else W_ENERGY * comp["energy_j"] / ENERGY_SCALE
    lat_terms = stale + dl
    if "latency" in drop_terms:
        lat_terms = 0.0
    return -(quality + lat_terms + fail + money + energy)


def cost_scalar(comp: dict) -> float:
    """Deployment cost (no quality term) used for the amortization ledger."""
    return (comp["delay_s"] + W_MONEY * comp["money"] / MONEY_SCALE
            + W_ENERGY * comp["energy_j"] / ENERGY_SCALE)


# --------------------------------------------------------------------------
# Workload 1: repeated combinatorial control (QAOA on NetworkQBench tasks)
# --------------------------------------------------------------------------

def qaoa_probs(norm: np.ndarray, n: int, params: np.ndarray) -> np.ndarray:
    p = len(params) // 2
    state = np.ones(1 << n, dtype=np.complex128) / math.sqrt(1 << n)
    for i in range(p):
        state = state * np.exp(-1j * params[i] * norm)
        state = _apply_mixer(state, params[p + i], n)
    probs = np.abs(state) ** 2
    return probs / probs.sum()


def _two_qubit_gates(inst: Instance, p: int, compile_level: int) -> int:
    edges = inst.metadata.get("edges", int(round(inst.n_vars * 1.5)))
    return max(1, int(round(edges * p * COMPILE_GATE_FACTOR[compile_level])))


def _nelder_mead(f, x0: np.ndarray, maxiter: int) -> tuple[np.ndarray, int]:
    from scipy.optimize import minimize
    calls = 0

    def wrapped(x):
        nonlocal calls
        calls += 1
        return f(x)

    res = minimize(wrapped, x0, method="Nelder-Mead",
                   options={"maxfev": maxiter, "xatol": 1e-3, "fatol": 1e-5})
    return res.x, calls


_WARM_CACHE: dict = {}
_WARM_PROB_CACHE: dict = {}


def warm_params(task: str, size: int, p: int) -> np.ndarray:
    """Family-level parameters fitted once on noiseless training instances.

    For sizes above 12, parameters are transferred from the size-12 fit and
    lightly refined on one instance, exploiting QAOA parameter
    concentration; full multistart fitting at 2^16+ amplitudes would be
    prohibitive and is unnecessary.
    """
    key = (task, size, p)
    if key in _WARM_CACHE:
        return _WARM_CACHE[key]
    if size > 12:
        x0 = warm_params(task, 12, p)
        inst = GENERATORS[task](size, 9_000)
        nm = _normalized_cost(inst.costs)

        def objective(theta):
            return float(np.dot(qaoa_probs(nm, inst.n_vars, theta), nm))

        best, _ = _nelder_mead(objective, x0, 40)
        best = best if objective(best) < objective(x0) else x0
        _WARM_CACHE[key] = best
        return best
    insts = [GENERATORS[task](size, 9_000 + i) for i in range(3)]
    norms = [_normalized_cost(i.costs) for i in insts]

    def objective(theta):
        return float(np.mean([np.dot(qaoa_probs(nm, insts[0].n_vars, theta), nm)
                              for nm in norms]))

    best_val, best = np.inf, None
    rng = np.random.default_rng(7)
    for _ in range(6):
        x0 = np.concatenate([rng.uniform(0, 2 * np.pi, p), rng.uniform(0, np.pi, p)])
        x, _ = _nelder_mead(objective, x0, 120 * p)
        v = objective(x)
        if v < best_val:
            best_val, best = v, x
    _WARM_CACHE[key] = best
    return best


def eval_qaoa(cfg: dict, inst: Instance, optimum: float, ctx: Context,
              fidelity: int, rng: np.random.Generator) -> dict:
    """Evaluate one configuration on one instance under one context.

    fidelity 0: analytic estimate, no simulation.
    fidelity 1: noiseless statevector + shot-limited sampling.
    fidelity 2: calibration-conditioned depolarizing + shot noise.
    fidelity 3: deployment; fid2 with perturbed calibration and realized
                queue/network draws (held-out truth).
    """
    n = inst.n_vars
    norm = _normalized_cost(inst.costs)
    placement = cfg["placement"]
    p, shots = cfg["p"], cfg["shots"]
    g2 = _two_qubit_gates(inst, p, cfg["compile_level"])
    shots_eff = shots * (MITIGATION_SHOT_MULTIPLIER if cfg["mitigation"] else 1)

    # -- calibration-conditioned depolarizing parameter
    lam = 0.0
    readout_flip = 0.0
    if placement in ("qpu_A", "qpu_B") and fidelity >= 2:
        eps = ctx.eps[placement]
        if fidelity == 3:
            eps = float(np.clip(eps * math.exp(rng.normal(0.0, 0.30)), 5e-4, 0.12))
        elif fidelity == 4:
            # mismatch deployment: heavier calibration tail, misspecified
            # gate-count exponent, and readout bit-flips absent from f2/f3
            eps = float(np.clip(eps * math.exp(rng.normal(0.0, 0.60)), 5e-4, 0.15))
            readout_flip = min(0.04, 2.0 * eps)
        exponent = g2 ** 0.9 if fidelity == 4 else g2
        lam = 1.0 - (1.0 - eps) ** exponent
    lam_train = lam * (MITIGATION_ERROR_MULTIPLIER if cfg["mitigation"] else 1.0)

    n_evals = 1 if cfg["warm_start"] else cfg["opt_iters"]

    # -- latency / energy / money model (all modeled, labeled)
    queue = ctx.queue_s.get(placement, 0.0)
    if fidelity >= 3 and placement in ("qpu_A", "qpu_B"):
        queue = float(queue * math.exp(rng.normal(0.0, 0.35)))
    amps = (1 << n) * p * n
    if placement == "local_sim":
        eval_time = amps / EDGE_FLOPS
        delay = COMPILE_TIME_S[cfg["compile_level"]] + n_evals * eval_time
        energy = POWER_W["local_sim"] * delay
        money = 0.0
        roundtrips = 0
    elif placement == "remote_sim":
        eval_time = amps / REMOTE_FLOPS + 0.005
        roundtrips = n_evals
        delay = (COMPILE_TIME_S[cfg["compile_level"]]
                 + n_evals * eval_time + roundtrips * ctx.rtt_s)
        energy = POWER_W["remote_sim"] * n_evals * eval_time
        money = REMOTE_SIM_MONEY_PER_S * n_evals * eval_time
    elif placement in ("qpu_A", "qpu_B"):
        _, _, per_circ, per_shot, per_shot_money = BACKEND_BASE[placement]
        circ_time = per_circ + shots_eff * per_shot
        roundtrips = n_evals
        delay = (COMPILE_TIME_S[cfg["compile_level"]] + queue
                 + n_evals * circ_time + roundtrips * ctx.rtt_s)
        energy = n_evals * (QPU_ENERGY_PER_CIRCUIT_J + shots_eff * QPU_ENERGY_PER_SHOT_J)
        money = per_shot_money * shots_eff * n_evals
    else:  # classical fallback: annealing from NetworkQBench, measured locally
        res = annealing_solver(inst, rng)
        delay = float(res["runtime_s"]) * 5.0  # modeled edge-CPU slowdown
        energy = POWER_W["classical"] * delay
        money = 0.0
        gap = max(0.0, (res["cost"] - optimum) / (abs(optimum) + 1.0))
        return _finish_qaoa(cfg, ctx, delay, energy, money, gap,
                            bool(res["feasible"]), 0.0, fidelity, rng)

    # -- quality
    if fidelity == 0:
        gap0 = 0.30 / p + 2.0 * lam + 1.2 / math.sqrt(shots) \
            + (0.05 if cfg["warm_start"] else 0.0)
        return _finish_qaoa(cfg, ctx, delay, energy, money, gap0, True,
                            0.0, fidelity, rng, realized_fail=False)

    if cfg["warm_start"]:
        params = warm_params(inst.task, inst.size_label, p)
    else:
        sigma = 0.0 if placement in ("local_sim", "remote_sim") \
            else float(np.std(norm)) / math.sqrt(shots_eff)

        def objective(theta):
            probs = qaoa_probs(norm, n, theta)
            val = (1 - lam_train) * float(np.dot(probs, norm)) + lam_train * float(norm.mean())
            return val + (rng.normal(0.0, sigma) if sigma > 0 else 0.0)

        x0 = warm_params(inst.task, inst.size_label, p)
        params, _ = _nelder_mead(objective, x0 + rng.normal(0, 0.15, size=2 * p),
                                 cfg["opt_iters"])

    if cfg["warm_start"]:
        probability_key = (inst.task, inst.size_label, inst.seed, p,
                           tuple(np.round(params, 12)))
        probs = _WARM_PROB_CACHE.get(probability_key)
        if probs is None:
            probs = qaoa_probs(norm, n, params)
            _WARM_PROB_CACHE[probability_key] = probs
        else:
            probs = probs.copy()
    else:
        probs = qaoa_probs(norm, n, params)
    if lam > 0:
        probs = (1 - lam) * probs + lam / len(probs)
    sampled = rng.choice(len(probs), size=shots, p=probs)
    if readout_flip > 0.0:
        flips = rng.random((len(sampled), n)) < readout_flip
        masks = (flips.astype(np.uint32) << np.arange(n, dtype=np.uint32)) \
            .sum(axis=1)
        sampled = (sampled.astype(np.uint32) ^ masks).astype(int)
    valid = sampled[inst.feasible[sampled]]
    pick = int(valid[np.argmin(inst.costs[valid])]) if len(valid) \
        else int(sampled[np.argmin(inst.costs[sampled])])
    gap = max(0.0, (float(inst.costs[pick]) - optimum) / (abs(optimum) + 1.0))
    feas = bool(inst.feasible[pick])

    p_fail = 1.0 - (1.0 - ctx.fail_prob) ** max(roundtrips, 1) \
        if placement in ("remote_sim", "qpu_A", "qpu_B") else 0.0
    realized_fail = bool(fidelity >= 3 and rng.random() < p_fail)
    if realized_fail:
        res = annealing_solver(inst, rng)
        gap = max(0.0, (res["cost"] - optimum) / (abs(optimum) + 1.0))
        feas = bool(res["feasible"])
        delay += 1.0 + float(res["runtime_s"]) * 5.0
    return _finish_qaoa(cfg, ctx, delay, energy, money, gap, feas, p_fail,
                        fidelity, rng, realized_fail=realized_fail)


def _finish_qaoa(cfg, ctx, delay, energy, money, gap, feas, p_fail, fidelity,
                 rng, realized_fail=False):
    comp = {
        "quality_loss": gap,
        "delay_s": delay,
        "deadline_miss": float(delay > ctx.deadline_s),
        "staleness": ctx.volatility * delay,
        "infeasible": float((not feas) or energy > ctx.energy_budget_j
                            or money > ctx.money_budget),
        "energy_j": energy,
        "money": money,
        "fail_prob": p_fail,
        "realized_fail": float(realized_fail),
        "fidelity": fidelity,
    }
    comp["utility"] = utility_from_components(comp)
    comp["cost_scalar"] = cost_scalar(comp)
    return comp


# --------------------------------------------------------------------------
# Workload 2: small hybrid learning module (variational quantum classifier)
# --------------------------------------------------------------------------

_DATA_CACHE: dict = {}
_VQC_TRAIN_CACHE: dict = {}


def vqc_dataset(seed: int, source: str = "synthetic"):
    """Train/test split for the learning workload.

    "synthetic": a fresh make_classification draw per seed.
    "cancer": the (real) breast-cancer dataset; per-seed 120/80 subsample
    split, with the 6 features selected by ANOVA F-score and standardized
    using TRAINING statistics only (no leakage).
    "digits": the harder 3-versus-8 handwritten-digit task under the same
    leakage-free 120/80 protocol.
    """
    key = (seed, source)
    if key in _DATA_CACHE:
        return _DATA_CACHE[key]
    if source == "synthetic":
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=200, n_features=6, n_informative=4, n_redundant=1,
            flip_y=0.05, class_sep=1.0, random_state=seed)
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        y = 2 * y - 1
        _DATA_CACHE[key] = (X[:120], y[:120], X[120:], y[120:])
    elif source == "cancer":
        from sklearn.datasets import load_breast_cancer
        from sklearn.feature_selection import SelectKBest, f_classif
        data = load_breast_cancer()
        rng = np.random.default_rng(40_000 + seed)
        idx = rng.permutation(len(data.target))[:200]
        X, y = data.data[idx], 2 * data.target[idx] - 1
        Xtr, ytr, Xte, yte = X[:120], y[:120], X[120:], y[120:]
        sel = SelectKBest(f_classif, k=6).fit(Xtr, ytr)
        Xtr, Xte = sel.transform(Xtr), sel.transform(Xte)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xtr = np.clip((Xtr - mu) / sd, -2.5, 2.5)
        Xte = np.clip((Xte - mu) / sd, -2.5, 2.5)
        _DATA_CACHE[key] = (Xtr, ytr, Xte, yte)
    else:
        from sklearn.datasets import load_digits
        from sklearn.feature_selection import SelectKBest, f_classif
        data = load_digits()
        keep = np.isin(data.target, [3, 8])
        Xall = data.data[keep]
        yall = np.where(data.target[keep] == 8, 1, -1)
        rng = np.random.default_rng(50_000 + seed)
        idx = rng.permutation(len(yall))[:200]
        X, y = Xall[idx], yall[idx]
        Xtr, ytr, Xte, yte = X[:120], y[:120], X[120:], y[120:]
        sel = SelectKBest(f_classif, k=6).fit(Xtr, ytr)
        Xtr, Xte = sel.transform(Xtr), sel.transform(Xte)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xtr = np.clip((Xtr - mu) / sd, -2.5, 2.5)
        Xte = np.clip((Xte - mu) / sd, -2.5, 2.5)
        _DATA_CACHE[key] = (Xtr, ytr, Xte, yte)
    return _DATA_CACHE[key]


def _apply_ry_batch(states: np.ndarray, q: int, angles: np.ndarray, n: int):
    stride = 1 << q
    view = states.reshape(states.shape[0], -1, stride << 1)
    a = view[:, :, :stride].copy()
    b = view[:, :, stride:].copy()
    c = np.cos(angles / 2)[:, None, None]
    s = np.sin(angles / 2)[:, None, None]
    view[:, :, :stride] = c * a - s * b
    view[:, :, stride:] = s * a + c * b


def _apply_cz_ring(states: np.ndarray, n: int):
    if n < 2:
        return
    idx = np.arange(states.shape[1])
    bits = (idx[:, None] >> np.arange(n)) & 1
    pairs = [(q, (q + 1) % n) for q in range(n)] if n > 2 else [(0, 1)]
    sign = np.ones(states.shape[1])
    for a, b in pairs:
        sign *= np.where((bits[:, a] == 1) & (bits[:, b] == 1), -1.0, 1.0)
    states *= sign[None, :]


def vqc_forward(X: np.ndarray, theta: np.ndarray, n: int, layers: int,
                reupload: bool) -> np.ndarray:
    """Per-qubit Z expectations (all Z observables commute: one shot yields
    every feature). Returns an array of shape (batch, n)."""
    B = X.shape[0]
    states = np.zeros((B, 1 << n), dtype=np.complex128)
    states[:, 0] = 1.0
    for q in range(n):
        _apply_ry_batch(states, q, X[:, q], n)
    t = theta.reshape(layers, n)
    for layer in range(layers):
        if reupload and layer > 0:
            for q in range(n):
                _apply_ry_batch(states, q, 0.5 * X[:, q], n)
        for q in range(n):
            _apply_ry_batch(states, q, np.full(B, t[layer, q]), n)
        _apply_cz_ring(states, n)
    probs = np.abs(states) ** 2
    idx = np.arange(1 << n)
    feats = np.empty((B, n))
    for q in range(n):
        sign = 1.0 - 2.0 * ((idx >> q) & 1)
        feats[:, q] = probs @ sign
    return feats


def train_vqc(seed: int, cfg: dict, rng: np.random.Generator,
              source: str = "synthetic"):
    """COBYLA circuit training on a mean-Z regression loss, then a classical
    logistic head over the per-qubit Z features (the plan's classical
    postprocessing variable). Returns (theta, head, evals)."""
    cache_key = (seed, source, cfg["n_qubits"], cfg["layers"], cfg["reupload"],
                 cfg["train_iters"], cfg["lr"])
    if cache_key in _VQC_TRAIN_CACHE:
        return _VQC_TRAIN_CACHE[cache_key]
    from scipy.optimize import minimize
    from sklearn.linear_model import LogisticRegression
    Xtr, ytr, _, _ = vqc_dataset(seed, source)
    n, layers = cfg["n_qubits"], cfg["layers"]
    Xn = Xtr[:, :n]
    init_scale = cfg["lr"] * 6.0  # {0.05, 0.15} -> {0.3, 0.9} init spread
    theta = rng.normal(0, init_scale, size=layers * n)
    evals = 0

    def loss(th):
        nonlocal evals
        evals += 1
        f = vqc_forward(Xn, th, n, layers, cfg["reupload"])
        return float(np.mean((f.mean(axis=1) - ytr) ** 2))

    res = minimize(loss, theta, method="COBYLA",
                   options={"maxiter": cfg["train_iters"]})
    theta = res.x
    Ftr = vqc_forward(Xn, theta, n, layers, cfg["reupload"])
    head = LogisticRegression(max_iter=300).fit(Ftr, ytr)
    result = (theta, head, evals)
    _VQC_TRAIN_CACHE[cache_key] = result
    return result


def eval_vqc(cfg: dict, seed: int, ctx: Context, fidelity: int,
             rng: np.random.Generator, source: str = "synthetic") -> dict:
    n, layers = cfg["n_qubits"], cfg["layers"]
    placement = cfg["placement"]
    shots = cfg["shots"]
    shots_eff = shots * (MITIGATION_SHOT_MULTIPLIER if cfg["mitigation"] else 1)
    depth_layers = layers * (2 if cfg["reupload"] else 1)
    g2 = max(1, int(round(n * depth_layers * COMPILE_GATE_FACTOR[cfg["compile_level"]])))

    lam = 0.0
    z_bias_sd = 0.0
    if placement in ("qpu_A", "qpu_B") and fidelity >= 2:
        eps = ctx.eps[placement]
        if fidelity == 3:
            eps = float(np.clip(eps * math.exp(rng.normal(0.0, 0.30)), 5e-4, 0.12))
        elif fidelity == 4:
            eps = float(np.clip(eps * math.exp(rng.normal(0.0, 0.60)), 5e-4, 0.15))
            z_bias_sd = min(0.08, 4.0 * eps)
        exponent = g2 ** 0.9 if fidelity == 4 else g2
        lam = 1.0 - (1.0 - eps) ** exponent
    if cfg["mitigation"]:
        lam *= MITIGATION_ERROR_MULTIPLIER

    # -- per-decision inference latency/energy/money (batched session of 32)
    queue = ctx.queue_s.get(placement, 0.0)
    if fidelity >= 3 and placement in ("qpu_A", "qpu_B"):
        queue = float(queue * math.exp(rng.normal(0.0, 0.35)))
    amps = (1 << n) * depth_layers * n
    if placement == "local_sim":
        delay = amps / EDGE_FLOPS
        energy = POWER_W["local_sim"] * delay
        money = 0.0
        p_fail = 0.0
    elif placement == "remote_sim":
        delay = amps / REMOTE_FLOPS + ctx.rtt_s
        energy = POWER_W["remote_sim"] * (amps / REMOTE_FLOPS)
        money = REMOTE_SIM_MONEY_PER_S * (amps / REMOTE_FLOPS)
        p_fail = ctx.fail_prob
    elif placement in ("qpu_A", "qpu_B"):
        _, _, per_circ, per_shot, per_shot_money = BACKEND_BASE[placement]
        delay = queue / 32.0 + per_circ + shots_eff * per_shot + ctx.rtt_s
        energy = QPU_ENERGY_PER_CIRCUIT_J + shots_eff * QPU_ENERGY_PER_SHOT_J
        money = per_shot_money * shots_eff
        p_fail = ctx.fail_prob
    else:  # classical substitute: logistic regression on same features
        from sklearn.linear_model import LogisticRegression
        Xtr, ytr, Xte, yte = vqc_dataset(seed, source)
        clf = LogisticRegression(max_iter=200).fit(Xtr, ytr)
        acc = float(clf.score(Xte, yte))
        delay = 2.0e-4 * 5.0
        energy = POWER_W["classical"] * delay
        return _finish_vqc(cfg, ctx, delay, energy, 0.0, acc, 0.0, fidelity)

    if fidelity == 0:
        acc0 = 0.55 + 0.10 * min(layers, 2) + 0.03 * (n / 6.0) - 1.5 * lam \
            - 0.3 / math.sqrt(shots)
        return _finish_vqc(cfg, ctx, delay, energy, money, float(np.clip(acc0, 0.4, 0.95)),
                           p_fail, fidelity)

    theta, head, _ = train_vqc(seed, cfg, np.random.default_rng(
        10_000 + seed * 17 + cfg["train_iters"]), source)
    _, _, Xte, yte = vqc_dataset(seed, source)
    z = vqc_forward(Xte[:, :n], theta, n, layers, cfg["reupload"])
    z_dep = (1 - lam) * z
    if z_bias_sd > 0.0:
        z_dep = z_dep + rng.normal(0.0, z_bias_sd, size=n)[None, :] \
            if z_dep.ndim == 2 else z_dep + rng.normal(0.0, z_bias_sd)
    if fidelity >= 1:
        noise_sd = np.sqrt(np.clip(1 - z_dep ** 2, 0.05, 1.0) / shots_eff)
        z_dep = z_dep + rng.normal(0, 1.0, size=z.shape) * noise_sd
    acc = float(np.mean(head.predict(z_dep) == yte))

    realized_fail = bool(fidelity >= 3 and rng.random() < p_fail)
    if realized_fail:
        delay += 1.0
        acc = 0.5
    return _finish_vqc(cfg, ctx, delay, energy, money, acc, p_fail, fidelity,
                       realized_fail=realized_fail)


def _finish_vqc(cfg, ctx, delay, energy, money, acc, p_fail, fidelity,
                realized_fail=False):
    comp = {
        "quality_loss": 1.0 - acc,
        "delay_s": delay,
        "deadline_miss": float(delay > ctx.deadline_s),
        "staleness": ctx.volatility * delay,
        "infeasible": float(energy > ctx.energy_budget_j or money > ctx.money_budget),
        "energy_j": energy,
        "money": money,
        "fail_prob": p_fail,
        "realized_fail": float(realized_fail),
        "fidelity": fidelity,
    }
    comp["utility"] = utility_from_components(comp)
    comp["cost_scalar"] = cost_scalar(comp)
    return comp


# --------------------------------------------------------------------------
# Unified evaluation front-end with a cost ledger (plan section 7.3)
# --------------------------------------------------------------------------

FIDELITY_COST = {0: 0.02, 1: 0.30, 2: 1.00, 3: 1.00, 4: 1.00}


@dataclass
class Workload:
    kind: str            # "qaoa" | "vqc"
    task: str            # networkqbench task for qaoa; dataset tag for vqc
    size: int
    seed: int
    inst: Instance | None = None
    optimum: float = 0.0

    QAOA_TASK = {"qaoa": "routing", "qaoa_channel": "channel",
                 "qaoa_place": "placement", "qaoa_large": "routing",
                 "qaoa_xl": "routing"}

    @staticmethod
    def make(kind: str, seed: int, size: int = 12) -> "Workload":
        if kind.startswith("qaoa"):
            task = Workload.QAOA_TASK[kind]
            if kind == "qaoa_large":
                size = 16
            elif kind == "qaoa_xl":
                size = 20
            inst = GENERATORS[task](size, seed)
            opt = exact_solver(inst)["cost"]
            return Workload(kind, task, size, seed, inst, opt)
        source = {"vqc_cancer": "cancer", "vqc_digits": "digits"}.get(kind, "synthetic")
        return Workload(kind, source, 6, seed)


@dataclass
class Ledger:
    rows: list = field(default_factory=list)
    spent: float = 0.0
    modeled_cost_s: float = 0.0
    evaluations: int = 0
    unique_configs: set = field(default_factory=set)
    fidelity_counts: dict = field(default_factory=dict)

    def charge(self, fidelity: int, comp: dict, cfg: dict | None = None):
        self.spent += FIDELITY_COST[fidelity]
        # modeled evaluation cost: fid0 free-ish, sims charged at modeled time
        self.modeled_cost_s += 0.001 if fidelity == 0 else comp["cost_scalar"]
        self.evaluations += 1
        self.fidelity_counts[fidelity] = self.fidelity_counts.get(fidelity, 0) + 1
        if cfg is not None:
            self.unique_configs.add(tuple(sorted(cfg.items())))


def evaluate(wl: Workload, cfg: dict, ctx: Context, fidelity: int,
             rng: np.random.Generator, ledger: Ledger | None = None) -> dict:
    if wl.kind.startswith("qaoa"):
        comp = eval_qaoa(cfg, wl.inst, wl.optimum, ctx, fidelity, rng)
    else:
        comp = eval_vqc(cfg, wl.seed, ctx, fidelity, rng, source=wl.task)
    if ledger is not None:
        ledger.charge(fidelity, comp, cfg)
    return comp


def mean_utility(wl: Workload, cfg: dict, cm: ContextModel, windows: list[int],
                 rng: np.random.Generator, fidelity: int, draws: int,
                 ledger: Ledger | None = None, drop_terms: tuple = (),
                 cvar_alpha: float | None = None) -> float:
    utils = []
    for _ in range(draws):
        w = windows[rng.integers(len(windows))]
        ctx = cm.sample(rng, w)
        comp = evaluate(wl, cfg, ctx, fidelity, rng, ledger)
        utils.append(utility_from_components(comp, drop_terms))
    utils = np.array(utils)
    if cvar_alpha is not None:
        k = max(1, int(math.ceil(cvar_alpha * len(utils))))
        return float(np.sort(utils)[:k].mean())
    return float(utils.mean())
