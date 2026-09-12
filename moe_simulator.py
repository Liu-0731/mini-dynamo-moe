"""A dependency-free, single-card simulation of a sparse MoE layer.

The code follows the real data path: router logits -> softmax -> Top-K ->
expert capacity/drop policy -> expert MLP -> weighted combination.  Logical
experts are all executed by the same CPU process; no CUDA or model download is
required.
"""

import math
import random
from typing import Optional

from experiment_logger import append_experiment

Vector = list[float]
Matrix = list[list[float]]
Route = tuple[int, float]  # (expert id, normalized router weight)


def linear(x: Vector, weights: Matrix, bias: Vector) -> Vector:
    return [sum(x[j] * row[j] for j in range(len(x))) + bias[i]
            for i, row in enumerate(weights)]


def softmax(values: Vector) -> Vector:
    largest = max(values)
    exps = [math.exp(v - largest) for v in values]
    total = sum(exps)
    return [v / total for v in exps]


def top_indices(values: Vector, count: int) -> list[int]:
    return sorted(range(len(values)), key=lambda i: values[i], reverse=True)[:count]


def make_matrix(rows: int, cols: int, rng: random.Random, scale: float = 0.5) -> Matrix:
    return [[rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]


def make_experts(num_experts: int, model_dim: int, hidden_dim: int,
                 rng: random.Random) -> list[tuple[Matrix, Vector, Matrix, Vector]]:
    return [(make_matrix(hidden_dim, model_dim, rng), [0.0] * hidden_dim,
             make_matrix(model_dim, hidden_dim, rng), [0.0] * model_dim)
            for _ in range(num_experts)]


def expert_forward(x: Vector, expert: tuple[Matrix, Vector, Matrix, Vector]) -> Vector:
    w1, b1, w2, b2 = expert
    hidden = [max(0.0, v) for v in linear(x, w1, b1)]
    return linear(hidden, w2, b2)


def make_router(
    num_experts: int,
    model_dim: int,
    rng: random.Random,
    hot_expert: int = 0,
    hot_bias: float = 1.5,
) -> tuple[Matrix, Vector]:
    weights = make_matrix(num_experts, model_dim, rng, scale=0.25)
    bias = [0.0] * num_experts
    bias[hot_expert] = hot_bias  # create a controllable hot expert/straggler
    return weights, bias


def router_candidates(tokens: list[Vector], router: tuple[Matrix, Vector]):
    weights, bias = router
    probabilities, ranked = [], []
    for token in tokens:
        probs = softmax(linear(token, weights, bias))
        probabilities.append(probs)
        ranked.append(top_indices(probs, len(probs)))
    return probabilities, ranked


def capacity_for(total_tokens: int, num_experts: int, top_k: int, factor: float) -> int:
    return max(1, math.ceil(factor * total_tokens * top_k / num_experts))


def _try_assign_with_backtracking(
    token_id: int,
    expert_id: int,
    score: float,
    selected: list[list[Route]],
    loads: list[int],
    probabilities: list[Vector],
    ranked_experts: list[list[int]],
    capacity: int,
    depth: int,
) -> bool:
    """Assign one route, optionally moving a lower-score victim.

    This is a bounded local search, not an NP-hard global optimizer.  When the
    target expert is full, it tries to move a lower-score token on that expert
    to one of the token's later candidates.  A snapshot makes every failed
    attempt reversible, which is the important backtracking behavior for this
    teaching simulator.
    """
    if expert_id in {expert for expert, _ in selected[token_id]}:
        return False
    if loads[expert_id] < capacity:
        selected[token_id].append((expert_id, score))
        loads[expert_id] += 1
        return True
    if depth <= 0:
        return False

    victims: list[tuple[float, int, int]] = []
    for victim_id, routes in enumerate(selected):
        for route_index, (victim_expert, victim_score) in enumerate(routes):
            if victim_expert == expert_id and victim_id != token_id:
                victims.append((victim_score, victim_id, route_index))
    victims.sort(key=lambda item: item[0])

    for _, victim_id, route_index in victims:
        snapshot = [routes[:] for routes in selected]
        load_snapshot = loads[:]
        _, victim_score = selected[victim_id].pop(route_index)
        loads[expert_id] -= 1

        alternatives = [
            (probabilities[victim_id][candidate], candidate)
            for candidate in ranked_experts[victim_id]
            if candidate != expert_id
            and candidate not in {expert for expert, _ in selected[victim_id]}
        ]
        alternatives.sort(reverse=True)
        for alternative_score, alternative_expert in alternatives:
            if _try_assign_with_backtracking(
                victim_id,
                alternative_expert,
                alternative_score,
                selected,
                loads,
                probabilities,
                ranked_experts,
                capacity,
                depth - 1,
            ):
                if loads[expert_id] < capacity:
                    selected[token_id].append((expert_id, score))
                    loads[expert_id] += 1
                    return True
            selected[:] = [routes[:] for routes in snapshot]
            loads[:] = load_snapshot
        selected[:] = [routes[:] for routes in snapshot]
        loads[:] = load_snapshot
    return False


def _priority_backtracking_routes(
    probabilities: list[Vector],
    ranked_experts: list[list[int]],
    top_k: int,
    capacity: int,
    reservation_fraction: float = 0.25,
    backtrack_depth: int = 2,
) -> tuple[list[list[Route]], list[int]]:
    """Capacity-aware routing with priority, reservation, and local repair.

    The policy mirrors the decisions described in the paper at a small scale:

    1. process each token's first candidate before lower-ranked candidates;
    2. reserve a modest part of each expert's capacity for second candidates;
    3. fill remaining slots with later candidates; and
    4. locally backtrack when a high-priority route meets a full expert.

    Reservation is deliberately a tunable heuristic rather than a claim of the
    paper's exact production scheduler.  It makes the scheduling trade-off
    visible while remaining deterministic and easy to inspect.
    """
    num_tokens = len(probabilities)
    num_experts = len(probabilities[0])
    selected: list[list[Route]] = [[] for _ in range(num_tokens)]
    loads = [0] * num_experts

    secondary_demand = [0] * num_experts
    if top_k > 1:
        for candidates in ranked_experts:
            if len(candidates) > 1:
                secondary_demand[candidates[1]] += 1
    reserve = [
        min(max(capacity - 1, 0), math.ceil(demand * reservation_fraction))
        for demand in secondary_demand
    ]
    primary_capacity = [max(1, capacity - amount) for amount in reserve]

    # High-confidence primary routes go first.  In a tie, tokens with a more
    # concentrated choice distribution are handled first.
    primary_order = sorted(
        range(num_tokens),
        key=lambda token_id: (
            probabilities[token_id][ranked_experts[token_id][0]],
            probabilities[token_id][ranked_experts[token_id][0]]
            - probabilities[token_id][ranked_experts[token_id][1]]
            if len(ranked_experts[token_id]) > 1 else 1.0,
        ),
        reverse=True,
    )
    for token_id in primary_order:
        primary = ranked_experts[token_id][0]
        _try_assign_with_backtracking(
            token_id,
            primary,
            probabilities[token_id][primary],
            selected,
            loads,
            probabilities,
            ranked_experts,
            primary_capacity[primary],
            backtrack_depth,
        )

    # Release the reservations and fill each token's remaining Top-K slots in
    # candidate order.  Tokens with fewer viable candidates are attempted
    # first so they do not lose their only useful fallback.
    second_order = sorted(
        range(num_tokens),
        key=lambda token_id: len(ranked_experts[token_id]),
    )
    for token_id in second_order:
        for candidate in ranked_experts[token_id]:
            if len(selected[token_id]) >= top_k:
                break
            if candidate in {expert for expert, _ in selected[token_id]}:
                continue
            _try_assign_with_backtracking(
                token_id,
                candidate,
                probabilities[token_id][candidate],
                selected,
                loads,
                probabilities,
                ranked_experts,
                capacity,
                backtrack_depth,
            )

    return selected, loads


def choose_routes(
    probabilities,
    ranked_experts,
    top_k: int,
    capacity: Optional[int],
    strategy: str,
    reservation_fraction: float = 0.25,
    backtrack_depth: int = 2,
):
    num_tokens, num_experts = len(probabilities), len(probabilities[0])
    selected: list[list[Route]] = [[] for _ in range(num_tokens)]
    if strategy == "baseline" or capacity is None:
        for t in range(num_tokens):
            selected[t] = [(e, probabilities[t][e]) for e in ranked_experts[t][:top_k]]
    elif strategy == "token_drop":
        by_expert = [[] for _ in range(num_experts)]
        for t in range(num_tokens):
            for e in ranked_experts[t][:top_k]:
                by_expert[e].append((probabilities[t][e], t))
        for e, routes in enumerate(by_expert):
            for score, t in sorted(routes, reverse=True)[:capacity]:
                selected[t].append((e, score))
    elif strategy == "expanded_drop":
        pool = [(probabilities[t][e], t, e)
                for t in range(num_tokens) for e in ranked_experts[t]]
        loads = [0] * num_experts
        for score, t, e in sorted(pool, reverse=True):
            if len(selected[t]) < top_k and loads[e] < capacity:
                selected[t].append((e, score)); loads[e] += 1
    elif strategy == "priority_backtrack":
        if capacity is None:
            raise ValueError("priority_backtrack requires a capacity")
        selected, _ = _priority_backtracking_routes(
            probabilities,
            ranked_experts,
            top_k,
            capacity,
            reservation_fraction=reservation_fraction,
            backtrack_depth=backtrack_depth,
        )
    else:
        raise ValueError(
            "strategy must be baseline, token_drop, expanded_drop, or priority_backtrack"
        )
    for t, routes in enumerate(selected):
        total = sum(w for _, w in routes)
        if total:
            selected[t] = [(e, w / total) for e, w in routes]
    loads = [0] * num_experts
    for routes in selected:
        for e, _ in routes: loads[e] += 1
    return selected, loads


def moe_forward(
    tokens,
    router,
    experts,
    top_k: int,
    capacity_factor: float,
    strategy: str,
    reservation_fraction: float = 0.25,
    backtrack_depth: int = 2,
):
    probabilities, ranked = router_candidates(tokens, router)
    capacity = capacity_for(len(tokens), len(experts), top_k, capacity_factor)
    routes, loads = choose_routes(probabilities, ranked, top_k,
                                  None if strategy == "baseline" else capacity,
                                  strategy,
                                  reservation_fraction=reservation_fraction,
                                  backtrack_depth=backtrack_depth)
    outputs = []
    for token, token_routes in zip(tokens, routes):
        if not token_routes:
            outputs.append(token[:]); continue  # residual if all routes dropped
        output = [0.0] * len(token)
        for e, weight in token_routes:
            value = expert_forward(token, experts[e])
            for i, v in enumerate(value): output[i] += weight * v
        outputs.append(output)
    return outputs, routes, loads, capacity


def main() -> None:
    rng = random.Random(7)
    n, e, d, h, top_k = 32, 4, 8, 16, 2
    tokens = [[rng.uniform(-1, 1) for _ in range(d)] for _ in range(n)]
    router, experts = make_router(e, d, rng), make_experts(e, d, h, rng)
    print("Mini MoE forward-pass simulation")
    print(f"tokens={n}, experts={e}, top_k={top_k}")
    rows = []
    for strategy in ("baseline", "token_drop", "expanded_drop", "priority_backtrack"):
        outputs, routes, loads, capacity = moe_forward(tokens, router, experts,
                                                        top_k, 1.0, strategy)
        kept = sum(len(r) for r in routes)
        rows.append((strategy, capacity, loads, kept, n * top_k))
        print(f"\n[{strategy}] capacity={capacity}; loads={loads}; routes kept={kept}/{n*top_k}")
        print(f"token 0 routes={routes[0]}; output[:3]={[round(v, 4) for v in outputs[0][:3]]}")
    table = (
        "| 策略 | 容量 | 专家负载 | 保留路由 |\n"
        "|---|---:|---|---:|\n"
        + "\n".join(
            f"| {strategy} | {capacity} | {loads} | {kept}/{requested} |"
            for strategy, capacity, loads, kept, requested in rows
        )
    )
    append_experiment(
        title="独立 MoE 前向实验：四种路由策略",
        question="在同一批 token 上，容量限制和局部回溯如何改变专家负载与路由保留率？",
        setup="T=32，E=4，model_dim=8，hidden_dim=16，top_k=2，capacity_factor=1.0，seed=7",
        result_markdown=table,
        analysis=[
            "baseline 保留全部路由，但热点专家可能超过平均负载。",
            "token_drop 用直接丢弃换取确定性容量上限。",
            "expanded_drop 和 priority_backtrack 尝试用候选专家补位，减少路由损失。",
            "priority_backtrack 不是严格全局最优匹配，而是带预留和有限回溯的可解释启发式。",
        ],
        limitations=[
            "专家 MLP 在单进程中执行，没有真实跨 GPU 通信。",
            "该脚本只验证路由和容量机制，不代表真实端到端延迟。",
        ],
    )
    print("实验结果已追加到 research_log.md")


if __name__ == "__main__":
    main()
