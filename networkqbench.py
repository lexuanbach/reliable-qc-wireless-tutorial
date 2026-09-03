"""Core workload and solver implementations for NetworkQBench-Wireless.

The primary quantum routine is an exact statevector simulation of p=1 QAOA;
the generic depth routine supports the explicitly labeled p-sensitivity study.
Nothing
in this module represents physical QPU execution. Deployment delay profiles
are modeled separately and labeled as such in every result row.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp, minimize


@dataclass
class Instance:
    task: str
    size_label: int
    n_vars: int
    seed: int
    costs: np.ndarray
    feasible: np.ndarray
    volatility: float
    deadline_s: float
    metadata: dict


def bit_table(n: int) -> np.ndarray:
    states = np.arange(1 << n, dtype=np.uint32)
    return ((states[:, None] >> np.arange(n, dtype=np.uint32)) & 1).astype(np.int8)


def _normalized_cost(costs: np.ndarray) -> np.ndarray:
    lo, hi = float(costs.min()), float(costs.max())
    if hi <= lo:
        return np.zeros_like(costs, dtype=float)
    return (costs - lo) / (hi - lo)


def make_channel_assignment(size: int, seed: int) -> Instance:
    rng = np.random.default_rng(seed)
    n = size
    bits = bit_table(n)
    edge_mask = rng.random((n, n)) < min(0.6, 3.5 / max(n - 1, 1))
    edge_mask = np.triu(edge_mask, 1)
    weights = np.triu(rng.uniform(0.5, 2.0, size=(n, n)), 1) * edge_mask
    same = bits[:, :, None] == bits[:, None, :]
    interference = (same * weights[None, :, :]).sum(axis=(1, 2))
    imbalance = (bits.sum(axis=1) - n / 2.0) ** 2
    costs = interference + 0.25 * imbalance
    feasible = np.ones(len(costs), dtype=bool)
    return Instance(
        task="channel",
        size_label=size,
        n_vars=n,
        seed=seed,
        costs=costs.astype(float),
        feasible=feasible,
        volatility=float(rng.uniform(0.01, 0.08)),
        deadline_s=float(rng.uniform(0.05, 0.5)),
        metadata={
            "edges": int(edge_mask.sum()),
            "weights": weights,
        },
    )


def _placement_dims(size: int) -> tuple[int, int]:
    return {6: (2, 3), 8: (2, 4), 9: (3, 3), 12: (3, 4),
            14: (2, 7), 16: (4, 4)}[size]


def make_service_placement(size: int, seed: int) -> Instance:
    rng = np.random.default_rng(seed + 10_000)
    functions, nodes = _placement_dims(size)
    n = functions * nodes
    bits = bit_table(n).reshape(-1, functions, nodes)
    demands = rng.integers(1, 4, size=functions)
    # Capacity is tight enough to create structure but always admits at least one assignment.
    cap_base = max(int(math.ceil(demands.sum() / nodes)), int(demands.max()))
    capacities = rng.integers(cap_base, cap_base + 3, size=nodes)
    capacities[0] = max(capacities[0], int(demands.sum()))
    energy = rng.uniform(0.2, 1.5, size=(functions, nodes))
    latency = rng.uniform(0.5, 3.0, size=(nodes, nodes))
    np.fill_diagonal(latency, 0.0)

    exact_one = (bits.sum(axis=2) == 1).all(axis=1)
    load = (bits * demands[None, :, None]).sum(axis=1)
    capacity_ok = (load <= capacities[None, :]).all(axis=1)
    feasible = exact_one & capacity_ok

    base = (bits * energy[None, :, :]).sum(axis=(1, 2))
    for f in range(functions - 1):
        base += np.einsum("bi,ij,bj->b", bits[:, f, :], latency, bits[:, f + 1, :])
    exact_pen = 15.0 * ((bits.sum(axis=2) - 1) ** 2).sum(axis=1)
    overflow = np.maximum(load - capacities[None, :], 0)
    cap_pen = 15.0 * (overflow**2).sum(axis=1)
    costs = base + exact_pen + cap_pen
    return Instance(
        task="placement",
        size_label=size,
        n_vars=n,
        seed=seed,
        costs=costs.astype(float),
        feasible=feasible,
        volatility=float(rng.uniform(0.02, 0.12)),
        deadline_s=float(rng.uniform(0.1, 0.8)),
        metadata={
            "functions": functions,
            "nodes": nodes,
            "demands": demands,
            "capacities": capacities,
            "energy": energy,
            "latency": latency,
        },
    )


def _routing_dims(size: int) -> tuple[int, int]:
    return {6: (2, 3), 8: (2, 4), 9: (3, 3), 12: (3, 4),
            14: (2, 7), 16: (4, 4), 20: (4, 5)}[size]


def make_path_selection(size: int, seed: int) -> Instance:
    rng = np.random.default_rng(seed + 20_000)
    commodities, paths = _routing_dims(size)
    n = commodities * paths
    bits = bit_table(n).reshape(-1, commodities, paths)
    edges = max(4, min(8, size // 2 + 2))
    incidence = rng.random((commodities, paths, edges)) < 0.35
    for c in range(commodities):
        for p in range(paths):
            if not incidence[c, p].any():
                incidence[c, p, rng.integers(edges)] = True
    demand = rng.uniform(0.8, 1.8, size=commodities)
    capacities = rng.uniform(1.6, 3.2, size=edges)
    path_latency = incidence.sum(axis=2) + rng.uniform(0.1, 1.0, size=(commodities, paths))

    exact_one = (bits.sum(axis=2) == 1).all(axis=1)
    load = np.einsum("bcp,cpe,c->be", bits, incidence.astype(float), demand)
    capacity_ok = (load <= capacities[None, :]).all(axis=1)
    feasible = exact_one & capacity_ok
    # Guarantee a feasible reference by relaxing capacities only when the random family is empty.
    if not feasible.any():
        candidates = np.flatnonzero(exact_one)
        violation = np.maximum(load[candidates] - capacities[None, :], 0).sum(axis=1)
        reference = int(candidates[np.argmin(violation)])
        capacities = np.maximum(capacities, load[reference] + 0.1)
        capacity_ok = (load <= capacities[None, :]).all(axis=1)
        feasible = exact_one & capacity_ok

    base = (bits * path_latency[None, :, :]).sum(axis=(1, 2))
    exact_pen = 20.0 * ((bits.sum(axis=2) - 1) ** 2).sum(axis=1)
    overflow = np.maximum(load - capacities[None, :], 0)
    cap_pen = 20.0 * (overflow**2).sum(axis=1)
    costs = base + exact_pen + cap_pen
    return Instance(
        task="routing",
        size_label=size,
        n_vars=n,
        seed=seed,
        costs=costs.astype(float),
        feasible=feasible,
        volatility=float(rng.uniform(0.03, 0.15)),
        deadline_s=float(rng.uniform(0.1, 1.0)),
        metadata={
            "commodities": commodities,
            "paths": paths,
            "edges": edges,
            "incidence": incidence,
            "demand": demand,
            "capacities": capacities,
            "path_latency": path_latency,
        },
    )


GENERATORS: dict[str, Callable[[int, int], Instance]] = {
    "channel": make_channel_assignment,
    "placement": make_service_placement,
    "routing": make_path_selection,
}


def exact_solver(inst: Instance) -> dict:
    start = time.perf_counter()
    valid_cost = np.where(inst.feasible, inst.costs, np.inf)
    idx = int(np.argmin(valid_cost))
    return {
        "solution": idx,
        "cost": float(inst.costs[idx]),
        "feasible": bool(inst.feasible[idx]),
        "runtime_s": time.perf_counter() - start,
        "evals": len(inst.costs),
    }


def random_solver(inst: Instance, rng: np.random.Generator, samples: int = 256) -> dict:
    start = time.perf_counter()
    idxs = rng.integers(0, len(inst.costs), size=samples)
    valid = idxs[inst.feasible[idxs]]
    if len(valid) == 0:
        idx = int(idxs[np.argmin(inst.costs[idxs])])
    else:
        idx = int(valid[np.argmin(inst.costs[valid])])
    return {
        "solution": idx,
        "cost": float(inst.costs[idx]),
        "feasible": bool(inst.feasible[idx]),
        "runtime_s": time.perf_counter() - start,
        "evals": samples,
    }


def _linearized_binary_milp(
    linear: np.ndarray,
    quadratic: dict[tuple[int, int], float],
    base_rows: list[np.ndarray],
    base_lb: list[float],
    base_ub: list[float],
) -> tuple[np.ndarray, int]:
    """Solve a binary quadratic model through exact McCormick linearization."""
    n = len(linear)
    pairs = sorted((min(i, j), max(i, j), value)
                   for (i, j), value in quadratic.items() if abs(value) > 1e-12)
    n_vars = n + len(pairs)
    objective = np.zeros(n_vars)
    objective[:n] = linear
    for k, (_, _, value) in enumerate(pairs):
        objective[n + k] = value

    rows = [np.pad(np.asarray(row, dtype=float), (0, len(pairs)))
            for row in base_rows]
    lb = list(base_lb)
    ub = list(base_ub)
    for k, (i, j, _) in enumerate(pairs):
        y = n + k
        # y <= x_i, y <= x_j, y >= x_i + x_j - 1.
        row = np.zeros(n_vars); row[y] = 1; row[i] = -1
        rows.append(row); lb.append(-np.inf); ub.append(0.0)
        row = np.zeros(n_vars); row[y] = 1; row[j] = -1
        rows.append(row); lb.append(-np.inf); ub.append(0.0)
        row = np.zeros(n_vars); row[i] = 1; row[j] = 1; row[y] = -1
        rows.append(row); lb.append(-np.inf); ub.append(1.0)
    constraints = LinearConstraint(np.vstack(rows), np.asarray(lb), np.asarray(ub)) \
        if rows else None
    result = milp(
        c=objective,
        integrality=np.ones(n_vars),
        bounds=Bounds(np.zeros(n_vars), np.ones(n_vars)),
        constraints=constraints,
        options={"time_limit": 30.0, "mip_rel_gap": 0.0},
    )
    if result.x is None:
        raise RuntimeError(f"SciPy/HiGHS MILP failed with status {result.status}")
    return np.rint(result.x[:n]).astype(np.int8), int(getattr(result, "mip_node_count", 0) or 0)


def milp_solver(inst: Instance) -> dict:
    """Open-source HiGHS MILP baseline on the original network formulation.

    Channel assignment uses exact binary-product linearization; placement and
    routing use their exact assignment/capacity constraints and linearize only
    the consecutive-function latency products. The returned state is always
    checked against the benchmark's independent feasibility mask.
    """
    start = time.perf_counter()
    n = inst.n_vars
    rows: list[np.ndarray] = []
    lb: list[float] = []
    ub: list[float] = []
    quadratic: dict[tuple[int, int], float] = {}

    if "channel" in inst.task:
        weights = np.asarray(inst.metadata["weights"], dtype=float)
        linear = np.full(n, 0.25 * (1.0 - n)) - weights.sum(axis=0) - weights.sum(axis=1)
        for i in range(n):
            for j in range(i + 1, n):
                quadratic[(i, j)] = 0.5 + 2.0 * weights[i, j]
    elif inst.task in ("placement", "abilene_placement"):
        functions = int(inst.metadata["functions"])
        nodes = int(inst.metadata["nodes"])
        demands = np.asarray(inst.metadata["demands"], dtype=float)
        capacities = np.asarray(inst.metadata["capacities"], dtype=float)
        energy = np.asarray(inst.metadata["energy"], dtype=float)
        latency = np.asarray(inst.metadata["latency"], dtype=float)
        linear = energy.reshape(-1).copy()
        for f in range(functions):
            row = np.zeros(n); row[f * nodes:(f + 1) * nodes] = 1.0
            rows.append(row); lb.append(1.0); ub.append(1.0)
        for node in range(nodes):
            row = np.zeros(n)
            for f in range(functions):
                row[f * nodes + node] = demands[f]
            rows.append(row); lb.append(-np.inf); ub.append(capacities[node])
        for f in range(functions - 1):
            for i in range(nodes):
                for j in range(nodes):
                    quadratic[(f * nodes + i, (f + 1) * nodes + j)] = latency[i, j]
    elif inst.task in ("routing", "abilene_routing"):
        commodities = int(inst.metadata["commodities"])
        paths = int(inst.metadata["paths"])
        incidence = np.asarray(inst.metadata["incidence"], dtype=float)
        demand = np.asarray(inst.metadata["demand"], dtype=float)
        capacities = np.asarray(inst.metadata["capacities"], dtype=float)
        linear = np.asarray(inst.metadata["path_latency"], dtype=float).reshape(-1)
        for commodity in range(commodities):
            row = np.zeros(n); row[commodity * paths:(commodity + 1) * paths] = 1.0
            rows.append(row); lb.append(1.0); ub.append(1.0)
        for edge in range(incidence.shape[2]):
            row = np.zeros(n)
            for c in range(commodities):
                for p in range(paths):
                    row[c * paths + p] = demand[c] * incidence[c, p, edge]
            rows.append(row); lb.append(-np.inf); ub.append(capacities[edge])
    else:
        raise ValueError(f"MILP metadata unavailable for task {inst.task}")

    bits, nodes = _linearized_binary_milp(linear, quadratic, rows, lb, ub)
    state = int(np.dot(bits.astype(np.uint64), 1 << np.arange(n, dtype=np.uint64)))
    return {
        "solution": state,
        "cost": float(inst.costs[state]),
        "feasible": bool(inst.feasible[state]),
        "runtime_s": time.perf_counter() - start,
        "evals": nodes,
    }


def task_specific_solver(inst: Instance, beam_width: int = 32) -> dict:
    """Constructive network baseline with a bounded beam for constraints."""
    start = time.perf_counter()
    evals = 0
    if "channel" in inst.task:
        weights = np.asarray(inst.metadata["weights"], dtype=float)
        order = np.argsort(-weights.sum(axis=0) - weights.sum(axis=1))
        x = np.zeros(inst.n_vars, dtype=np.int8)
        assigned: list[int] = []
        for node in order:
            scores = []
            for color in (0, 1):
                interference = sum(weights[min(node, j), max(node, j)]
                                   for j in assigned if color == x[j])
                imbalance = 0.25 * (x[assigned].sum() + color - (len(assigned) + 1) / 2.0) ** 2
                scores.append(interference + imbalance)
                evals += 1
            x[node] = int(np.argmin(scores)); assigned.append(int(node))
        state = int(np.dot(x.astype(np.uint64), 1 << np.arange(inst.n_vars, dtype=np.uint64)))
        for _ in range(2):
            for q in range(inst.n_vars):
                candidate = state ^ (1 << q); evals += 1
                if inst.costs[candidate] < inst.costs[state]:
                    state = candidate
    elif inst.task == "placement":
        functions = int(inst.metadata["functions"]); nodes = int(inst.metadata["nodes"])
        demands = np.asarray(inst.metadata["demands"]); capacities = np.asarray(inst.metadata["capacities"])
        energy = np.asarray(inst.metadata["energy"]); latency = np.asarray(inst.metadata["latency"])
        beam = [(0.0, [], np.zeros(nodes))]
        for f in range(functions):
            expanded = []
            for score, assignment, load in beam:
                for node in range(nodes):
                    evals += 1
                    if load[node] + demands[f] > capacities[node] + 1e-12:
                        continue
                    new_load = load.copy(); new_load[node] += demands[f]
                    inc = energy[f, node] + (latency[assignment[-1], node] if assignment else 0.0)
                    expanded.append((score + inc, assignment + [node], new_load))
            beam = sorted(expanded, key=lambda row: row[0])[:beam_width]
        if not beam:
            return local_search_solver(inst, np.random.default_rng(77), restarts=1)
        assignment = beam[0][1]
        state = sum(1 << (f * nodes + node) for f, node in enumerate(assignment))
    elif inst.task in ("routing", "abilene_routing"):
        commodities = int(inst.metadata["commodities"]); paths = int(inst.metadata["paths"])
        incidence = np.asarray(inst.metadata["incidence"], dtype=float)
        demand = np.asarray(inst.metadata["demand"], dtype=float)
        capacities = np.asarray(inst.metadata["capacities"], dtype=float)
        latency = np.asarray(inst.metadata["path_latency"], dtype=float)
        beam = [(0.0, [], np.zeros_like(capacities))]
        for c in range(commodities):
            expanded = []
            for score, assignment, load in beam:
                for path in np.argsort(latency[c]):
                    evals += 1
                    new_load = load + incidence[c, path] * demand[c]
                    if np.any(new_load > capacities + 1e-12):
                        continue
                    expanded.append((score + latency[c, path], assignment + [int(path)], new_load))
            beam = sorted(expanded, key=lambda row: row[0])[:beam_width]
        if not beam:
            return local_search_solver(inst, np.random.default_rng(79), restarts=1)
        assignment = beam[0][1]
        state = sum(1 << (c * paths + path) for c, path in enumerate(assignment))
    else:
        return local_search_solver(inst, np.random.default_rng(81), restarts=1)
    return {
        "solution": int(state),
        "cost": float(inst.costs[state]),
        "feasible": bool(inst.feasible[state]),
        "runtime_s": time.perf_counter() - start,
        "evals": int(evals),
    }


def local_search_solver(inst: Instance, rng: np.random.Generator, restarts: int = 24) -> dict:
    start = time.perf_counter()
    best = None
    evals = 0
    for _ in range(restarts):
        state = int(rng.integers(len(inst.costs)))
        improved = True
        while improved:
            improved = False
            neighbors = np.array([state ^ (1 << q) for q in range(inst.n_vars)], dtype=int)
            evals += len(neighbors)
            candidate = int(neighbors[np.argmin(inst.costs[neighbors])])
            if inst.costs[candidate] + 1e-12 < inst.costs[state]:
                state = candidate
                improved = True
        if best is None or (
            inst.feasible[state] and not inst.feasible[best]
        ) or (inst.feasible[state] == inst.feasible[best] and inst.costs[state] < inst.costs[best]):
            best = state
    assert best is not None
    return {
        "solution": int(best),
        "cost": float(inst.costs[best]),
        "feasible": bool(inst.feasible[best]),
        "runtime_s": time.perf_counter() - start,
        "evals": evals,
    }


def annealing_solver(
    inst: Instance,
    rng: np.random.Generator,
    sweeps: int = 120,
    restarts: int = 8,
) -> dict:
    start = time.perf_counter()
    norm = _normalized_cost(inst.costs)
    best = None
    evals = 0
    temps = np.geomspace(1.0, 0.01, sweeps)
    for _ in range(restarts):
        state = int(rng.integers(len(inst.costs)))
        for temp in temps:
            q = int(rng.integers(inst.n_vars))
            nxt = state ^ (1 << q)
            delta = norm[nxt] - norm[state]
            evals += 1
            if delta <= 0 or rng.random() < math.exp(-delta / temp):
                state = nxt
            if best is None or (
                inst.feasible[state] and not inst.feasible[best]
            ) or (inst.feasible[state] == inst.feasible[best] and inst.costs[state] < inst.costs[best]):
                best = state
    assert best is not None
    return {
        "solution": int(best),
        "cost": float(inst.costs[best]),
        "feasible": bool(inst.feasible[best]),
        "runtime_s": time.perf_counter() - start,
        "evals": evals,
    }


def mean_field_solver(
    inst: Instance,
    rng: np.random.Generator,
    restarts: int = 8,
    samples: int = 256,
) -> dict:
    """Optimize a classical independent-Bernoulli variational state.

    The method consumes the same dense cost vector as the ideal statevector
    experiment. It is a quantum-inspired small-instance control, not a
    scalable network solver. An analytic score-function gradient minimizes
    expected normalized cost; the product-mode and sampled states are then
    ranked by feasibility and objective value.
    """
    start = time.perf_counter()
    bits = bit_table(inst.n_vars).astype(float)
    norm = _normalized_cost(inst.costs)
    best_state = None
    best_key = None
    evals = 0

    def objective_and_grad(logits: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal evals
        evals += 1
        q = 1.0 / (1.0 + np.exp(-np.clip(logits, -12.0, 12.0)))
        logp = bits @ np.log(q) + (1.0 - bits) @ np.log1p(-q)
        probs = np.exp(logp - np.max(logp))
        probs /= probs.sum()
        value = float(np.dot(probs, norm))
        grad = (bits - q).T @ (probs * norm)
        return value, grad

    powers = 1 << np.arange(inst.n_vars, dtype=np.uint32)
    for _ in range(restarts):
        opt = minimize(
            objective_and_grad,
            rng.normal(0.0, 0.7, size=inst.n_vars),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 80, "ftol": 1e-10},
        )
        q = 1.0 / (1.0 + np.exp(-np.clip(opt.x, -12.0, 12.0)))
        sampled_bits = rng.random((samples, inst.n_vars)) < q
        sampled = (sampled_bits.astype(np.uint32) * powers).sum(axis=1)
        mode = int(np.sum((q >= 0.5).astype(np.uint32) * powers))
        for state in np.concatenate([[mode], sampled.astype(int)]):
            state = int(state)
            key = (not bool(inst.feasible[state]), float(inst.costs[state]))
            if best_key is None or key < best_key:
                best_key, best_state = key, state
    assert best_state is not None
    return {
        "solution": int(best_state),
        "cost": float(inst.costs[best_state]),
        "feasible": bool(inst.feasible[best_state]),
        "runtime_s": time.perf_counter() - start,
        "evals": evals,
    }


def _apply_mixer(state: np.ndarray, beta: float, n: int) -> np.ndarray:
    c, s = math.cos(beta), -1j * math.sin(beta)
    out = state
    for q in range(n):
        stride = 1 << q
        block = stride << 1
        view = out.reshape(-1, block)
        a = view[:, :stride].copy()
        b = view[:, stride:].copy()
        view[:, :stride] = c * a + s * b
        view[:, stride:] = s * a + c * b
    return out


def qaoa_probabilities(costs: np.ndarray, n: int, gamma: float, beta: float) -> np.ndarray:
    norm = _normalized_cost(costs)
    state = np.ones(1 << n, dtype=np.complex128) / math.sqrt(1 << n)
    state *= np.exp(-1j * gamma * norm)
    state = _apply_mixer(state, beta, n)
    probs = np.abs(state) ** 2
    return probs / probs.sum()


def qaoa_probabilities_depth(costs: np.ndarray, n: int,
                             parameters: np.ndarray) -> np.ndarray:
    """Exact statevector probabilities for standard X-mixer QAOA at any p.

    ``parameters`` stores all p cost angles followed by all p mixer angles.
    The primary benchmark remains p=1; this routine makes the depth-sensitivity
    experiment auditable without silently changing that estimand.
    """
    parameters = np.asarray(parameters, dtype=float)
    if parameters.ndim != 1 or len(parameters) == 0 or len(parameters) % 2:
        raise ValueError("parameters must contain p gammas followed by p betas")
    depth = len(parameters) // 2
    norm = _normalized_cost(costs)
    state = np.ones(1 << n, dtype=np.complex128) / math.sqrt(1 << n)
    for layer in range(depth):
        state *= np.exp(-1j * parameters[layer] * norm)
        state = _apply_mixer(state, parameters[depth + layer], n)
    probabilities = np.abs(state) ** 2
    return probabilities / probabilities.sum()


def warm_qaoa_probabilities(
    costs: np.ndarray,
    n: int,
    warm_bits: np.ndarray,
    gamma: float,
    beta: float,
    epsilon: float = 0.25,
) -> np.ndarray:
    """One-layer continuous warm-start QAOA following the biased-product mixer."""
    probabilities = np.where(warm_bits > 0, 1.0 - epsilon, epsilon)
    state = np.array([1.0 + 0.0j])
    for probability in probabilities:
        state = np.kron(np.array([math.sqrt(1.0 - probability), math.sqrt(probability)]), state)
    state *= np.exp(-1j * gamma * _normalized_cost(costs))
    cb, sb = math.cos(beta), math.sin(beta)
    for q, probability in enumerate(probabilities):
        theta = 2.0 * math.asin(math.sqrt(float(probability)))
        hz, hx = math.cos(theta), math.sin(theta)
        stride = 1 << q; block = stride << 1
        view = state.reshape(-1, block)
        a = view[:, :stride].copy(); b = view[:, stride:].copy()
        view[:, :stride] = (cb - 1j * sb * hz) * a - 1j * sb * hx * b
        view[:, stride:] = -1j * sb * hx * a + (cb + 1j * sb * hz) * b
    probs = np.abs(state) ** 2
    return probs / probs.sum()


def warm_start_qaoa_solver(inst: Instance, rng: np.random.Generator,
                           shots: int = 1024, epsilon: float = 0.25) -> dict:
    """Depth-one QAOA warm-started by the task-specific classical baseline."""
    start = time.perf_counter()
    seed_result = task_specific_solver(inst)
    warm_bits = bit_table(inst.n_vars)[seed_result["solution"]]
    norm = _normalized_cost(inst.costs)
    evals = 0

    def objective(theta: np.ndarray) -> float:
        nonlocal evals
        evals += 1
        probs = warm_qaoa_probabilities(inst.costs, inst.n_vars, warm_bits,
                                        float(theta[0]), float(theta[1]), epsilon)
        return float(np.dot(probs, norm))

    gammas = np.linspace(0.0, 2 * np.pi, 7, endpoint=False)
    betas = np.linspace(0.0, np.pi, 7, endpoint=False)
    best_theta = min((np.array([g, b]) for g in gammas for b in betas),
                     key=objective)
    best_value = objective(best_theta)
    opt = minimize(objective, best_theta, method="Nelder-Mead",
                   options={"maxiter": 40, "xatol": 1e-3, "fatol": 1e-5})
    theta = opt.x if opt.fun < best_value else best_theta
    probs = warm_qaoa_probabilities(inst.costs, inst.n_vars, warm_bits,
                                    float(theta[0]), float(theta[1]), epsilon)
    sampled = rng.choice(len(probs), size=shots, p=probs)
    valid = sampled[inst.feasible[sampled]]
    idx = int(valid[np.argmin(inst.costs[valid])]) if len(valid) \
        else int(sampled[np.argmin(inst.costs[sampled])])
    optimum = inst.costs[inst.feasible].min()
    opt_mask = inst.feasible & (inst.costs <= optimum + 1e-9)
    return {
        "solution": idx,
        "cost": float(inst.costs[idx]),
        "feasible": bool(inst.feasible[idx]),
        "runtime_s": time.perf_counter() - start,
        "evals": evals,
        "shots": shots,
        "success_prob": float(probs[opt_mask].sum()),
        "gamma": float(theta[0]),
        "beta": float(theta[1]),
        "expected_norm_cost": float(np.dot(probs, norm)),
        "classical_seed_evals": seed_result["evals"],
        "classical_seed_runtime_s": seed_result["runtime_s"],
    }


def qaoa_solver(inst: Instance, rng: np.random.Generator, shots: int = 1024) -> dict:
    start = time.perf_counter()
    norm = _normalized_cost(inst.costs)
    evals = 0

    def objective(theta: np.ndarray) -> float:
        nonlocal evals
        evals += 1
        p = qaoa_probabilities(inst.costs, inst.n_vars, float(theta[0]), float(theta[1]))
        return float(np.dot(p, norm))

    # Coarse deterministic coverage avoids seed-sensitive initialization.
    gammas = np.linspace(0.0, 2 * np.pi, 7, endpoint=False)
    betas = np.linspace(0.0, np.pi, 7, endpoint=False)
    best_theta = None
    best_value = np.inf
    for gamma in gammas:
        for beta in betas:
            value = objective(np.array([gamma, beta]))
            if value < best_value:
                best_value, best_theta = value, np.array([gamma, beta])
    assert best_theta is not None
    opt = minimize(
        objective,
        best_theta,
        method="Nelder-Mead",
        options={"maxiter": 40, "xatol": 1e-3, "fatol": 1e-5},
    )
    theta = opt.x if opt.fun < best_value else best_theta
    probs = qaoa_probabilities(inst.costs, inst.n_vars, float(theta[0]), float(theta[1]))
    sampled = rng.choice(len(probs), size=shots, p=probs)
    valid = sampled[inst.feasible[sampled]]
    if len(valid):
        idx = int(valid[np.argmin(inst.costs[valid])])
    else:
        idx = int(sampled[np.argmin(inst.costs[sampled])])
    success_prob = float(probs[np.where(inst.feasible & (inst.costs <= inst.costs[inst.feasible].min() + 1e-9))].sum())
    return {
        "solution": idx,
        "cost": float(inst.costs[idx]),
        "feasible": bool(inst.feasible[idx]),
        "runtime_s": time.perf_counter() - start,
        "evals": evals,
        "shots": shots,
        "success_prob": success_prob,
        "gamma": float(theta[0]),
        "beta": float(theta[1]),
        "expected_norm_cost": float(np.dot(probs, norm)),
    }


def operational_metrics(
    inst: Instance,
    result: dict,
    optimum: float,
    algorithm: str,
    profile: str,
) -> dict:
    if algorithm.startswith("qaoa"):
        # Explicitly modeled remote-QPU session profiles. These are not measured hardware values.
        cfg = {
            # queue, per-circuit overhead, per-shot time, classical round trip
            "optimistic": (0.01, 0.0005, 0.00001, 0.002),
            "nominal": (0.5, 0.005, 0.00005, 0.02),
            "stressed": (5.0, 0.02, 0.00020, 0.10),
        }[profile]
        queue_s, per_eval_s, per_shot_s, roundtrip_s = cfg
        delay = queue_s + result["evals"] * (
            per_eval_s + roundtrip_s + result.get("shots", 1024) * per_shot_s
        )
        evidence = "modeled_qpu"
    else:
        delay = float(result["runtime_s"])
        evidence = "measured_local"
    gap = max(0.0, (float(result["cost"]) - optimum) / (abs(optimum) + 1.0))
    infeasible_penalty = 5.0 if not result["feasible"] else 0.0
    stale = inst.volatility * delay
    missed = delay > inst.deadline_s
    deadline_penalty = 2.0 if missed else 0.0
    utility = -(gap + infeasible_penalty + stale + deadline_penalty)
    return {
        "delay_s": delay,
        "gap": gap,
        "staleness_loss": stale,
        "deadline_miss": bool(missed),
        "utility": utility,
        "timing_evidence": evidence,
    }
