"""Research step 2: sweep the expert capacity factor gamma."""

import random

from experiment_logger import append_experiment
from moe_simulator import make_experts, make_router, moe_forward


def make_tokens(num_tokens: int, model_dim: int, seed: int) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.uniform(-1.0, 1.0) for _ in range(model_dim)] for _ in range(num_tokens)]


def run_one(seed: int, strategy: str, capacity_factor: float) -> dict[str, float | int | str]:
    num_tokens, num_experts, model_dim, hidden_dim, top_k = 32, 4, 8, 16, 2
    tokens = make_tokens(num_tokens, model_dim, seed)
    rng = random.Random(seed + 1000)
    router = make_router(num_experts, model_dim, rng)
    experts = make_experts(num_experts, model_dim, hidden_dim, rng)
    _, routes, loads, capacity = moe_forward(
        tokens,
        router,
        experts,
        top_k=top_k,
        capacity_factor=capacity_factor,
        strategy=strategy,
    )
    requested = num_tokens * top_k
    kept = sum(len(item) for item in routes)
    average = sum(loads) / num_experts
    imbalance = max(loads, default=0) / average if average else 0.0
    return {
        "strategy": strategy,
        "capacity_factor": capacity_factor,
        "capacity": capacity,
        "kept_ratio": kept / requested,
        "max_load": max(loads, default=0),
        "imbalance": imbalance,
    }


def main() -> None:
    strategies = ("baseline", "token_drop", "expanded_drop", "priority_backtrack")
    capacity_factors = (0.5, 0.8, 1.0, 1.5)
    seeds = range(10)
    rows = [
        run_one(seed, strategy, capacity_factor)
        for capacity_factor in capacity_factors
        for strategy in strategies
        for seed in seeds
    ]

    summaries: dict[tuple[float, str], dict[str, float]] = {}
    print("Research question: how does gamma change route retention and load balance?")
    print("Fixed: T=32, E=4, top_k=2, 10 seeds")
    print()
    print(f"{'gamma':>6s} {'strategy':20s} {'C':>4s} {'kept%':>8s} {'max_load':>10s} {'imbalance':>12s}")
    print("-" * 68)
    for capacity_factor in capacity_factors:
        for strategy in strategies:
            group = [
                row for row in rows
                if row["capacity_factor"] == capacity_factor and row["strategy"] == strategy
            ]
            capacity = int(group[0]["capacity"])
            kept = sum(float(row["kept_ratio"]) for row in group) / len(group) * 100
            max_load = sum(float(row["max_load"]) for row in group) / len(group)
            imbalance = sum(float(row["imbalance"]) for row in group) / len(group)
            summaries[(capacity_factor, strategy)] = {
                "capacity": capacity,
                "kept": kept,
                "max_load": max_load,
                "imbalance": imbalance,
            }
            print(f"{capacity_factor:6.1f} {strategy:20s} {capacity:4d} {kept:7.2f}% {max_load:10.2f} {imbalance:12.3f}")

    analysis = []
    for strategy in strategies:
        low = summaries[(0.5, strategy)]
        high = summaries[(1.5, strategy)]
        analysis.append(
            f"{strategy} 的容量从 {low['capacity']} 增加到 {high['capacity']} 时，平均保留率从 "
            f"{low['kept']:.2f}% 变为 {high['kept']:.2f}%。"
        )
    analysis.extend([
        "capacity_factor 越小，专家容量越严格，通常会增加路由丢弃或备用专家冲突。",
        "capacity_factor 越大，路由保留率更高，但容量上限和潜在计算量也更高。",
        f"在 γ=1.5 时 priority_backtrack 的平均不均衡比为 {summaries[(1.5, 'priority_backtrack')]['imbalance']:.3f}，说明容量上限变宽并不自动等于完全均衡。",
        "baseline 不受容量因子影响，因为它本身不启用容量限制；它只作为无约束对照。",
        "下一轮应在更强热点专家偏置下重复本实验，确认结论不是由当前随机路由分布造成的。",
    ])

    table = (
        "| γ | 策略 | 容量 C | 平均保留路由 | 平均最大负载 | 平均不均衡比 |\n"
        "|---:|---|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {gamma:.1f} | {strategy} | {summaries[(gamma, strategy)]['capacity']} | "
            f"{summaries[(gamma, strategy)]['kept']:.2f}% | "
            f"{summaries[(gamma, strategy)]['max_load']:.2f} | "
            f"{summaries[(gamma, strategy)]['imbalance']:.3f} |"
            for gamma in capacity_factors
            for strategy in strategies
        )
    )
    append_experiment(
        title="实验 2：capacity_factor γ 扫描",
        question="容量因子如何影响专家负载、路由保留率和容量感知策略的权衡？",
        setup="T=32，E=4，top_k=2，γ=0.5/0.8/1.0/1.5，随机种子 0-9",
        result_markdown=table,
        analysis=analysis,
        limitations=[
            "仍然是单卡逻辑模拟，不能把保留率变化直接换算成真实 GPU 加速比。",
            "当前容量是按 expert 计算，不包含真实 EP device 映射和 All-to-All 通信。",
        ],
    )
    print()
    print("实验结果已追加到 research_log.md")


if __name__ == "__main__":
    main()
