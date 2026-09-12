"""Research step 3: stress the router with an increasingly hot expert."""

import random

from experiment_logger import append_experiment
from moe_simulator import make_experts, make_router, moe_forward


def make_tokens(num_tokens: int, model_dim: int, seed: int) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.uniform(-1.0, 1.0) for _ in range(model_dim)] for _ in range(num_tokens)]


def run_one(seed: int, strategy: str, hot_bias: float) -> dict[str, float | int | str]:
    num_tokens, num_experts, model_dim, hidden_dim, top_k = 32, 4, 8, 16, 2
    tokens = make_tokens(num_tokens, model_dim, seed)
    rng = random.Random(seed + 1000)
    router = make_router(num_experts, model_dim, rng, hot_bias=hot_bias)
    experts = make_experts(num_experts, model_dim, hidden_dim, rng)
    _, routes, loads, capacity = moe_forward(
        tokens,
        router,
        experts,
        top_k=top_k,
        capacity_factor=1.0,
        strategy=strategy,
    )
    requested = num_tokens * top_k
    kept = sum(len(item) for item in routes)
    average = sum(loads) / num_experts
    imbalance = max(loads, default=0) / average if average else 0.0
    return {
        "strategy": strategy,
        "hot_bias": hot_bias,
        "capacity": capacity,
        "kept_ratio": kept / requested,
        "max_load": max(loads, default=0),
        "imbalance": imbalance,
    }


def main() -> None:
    strategies = ("baseline", "token_drop", "expanded_drop", "priority_backtrack")
    hot_biases = (1.5, 2.5, 4.0)
    seeds = range(10)
    rows = [
        run_one(seed, strategy, hot_bias)
        for hot_bias in hot_biases
        for strategy in strategies
        for seed in seeds
    ]

    summaries: dict[tuple[float, str], dict[str, float]] = {}
    print("Research question: what happens when one expert becomes increasingly hot?")
    print("Fixed: T=32, E=4, top_k=2, capacity_factor=1.0, 10 seeds")
    print()
    print(f"{'hot_bias':>9s} {'strategy':20s} {'kept%':>8s} {'max_load':>10s} {'imbalance':>12s}")
    print("-" * 67)
    for hot_bias in hot_biases:
        for strategy in strategies:
            group = [
                row for row in rows
                if row["hot_bias"] == hot_bias and row["strategy"] == strategy
            ]
            kept = sum(float(row["kept_ratio"]) for row in group) / len(group) * 100
            max_load = sum(float(row["max_load"]) for row in group) / len(group)
            imbalance = sum(float(row["imbalance"]) for row in group) / len(group)
            summaries[(hot_bias, strategy)] = {
                "kept": kept,
                "max_load": max_load,
                "imbalance": imbalance,
            }
            print(f"{hot_bias:9.1f} {strategy:20s} {kept:7.2f}% {max_load:10.2f} {imbalance:12.3f}")

    analysis = []
    for hot_bias in hot_biases:
        expanded = summaries[(hot_bias, "expanded_drop")]
        priority = summaries[(hot_bias, "priority_backtrack")]
        analysis.append(
            f"hot_bias={hot_bias:.1f} 时，Expanded Drop 保留率 {expanded['kept']:.2f}%，"
            f"Priority Backtracking 保留率 {priority['kept']:.2f}%。"
        )
        analysis.append(
            f"hot_bias={hot_bias:.1f} 时，两种候选补位策略的不均衡比分别为 "
            f"{expanded['imbalance']:.3f} 和 {priority['imbalance']:.3f}。"
        )
    analysis.extend([
        "热点越强，baseline 的最大负载会继续增加；容量策略则把最大负载限制在容量上限附近。",
        "如果 Priority Backtracking 的优势只在强热点下出现，说明它主要解决的是候选冲突，而不是普通负载。",
        "如果两种候选补位策略结果接近，下一步应增加更小容量或构造更多候选冲突，才能区分调度器质量。",
    ])

    table = (
        "| hot_bias | 策略 | 平均保留路由 | 平均最大负载 | 平均不均衡比 |\n"
        "|---:|---|---:|---:|---:|\n"
        + "\n".join(
            f"| {bias:.1f} | {strategy} | {summaries[(bias, strategy)]['kept']:.2f}% | "
            f"{summaries[(bias, strategy)]['max_load']:.2f} | "
            f"{summaries[(bias, strategy)]['imbalance']:.3f} |"
            for bias in hot_biases
            for strategy in strategies
        )
    )
    append_experiment(
        title="实验 3：热点专家压力测试",
        question="当一个专家越来越热门时，候选补位和局部回溯能否保持负载与路由质量？",
        setup="T=32，E=4，top_k=2，capacity_factor=1.0，hot_bias=1.5/2.5/4.0，随机种子 0-9",
        result_markdown=table,
        analysis=analysis,
        limitations=[
            "hot_bias 是合成 Router 偏置，不等于真实语言模型训练出来的专家偏好。",
            "仍未测量真实跨 GPU 通信、权重搬运和模型准确率。",
        ],
    )
    print()
    print("实验结果已追加到 research_log.md")


if __name__ == "__main__":
    main()
