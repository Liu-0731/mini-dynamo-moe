"""Research step 7: scan top-k and compare capacity-aware routing policies."""

import random

from experiment_logger import append_experiment
from moe_simulator import make_experts, make_router, moe_forward


def make_tokens(num_tokens: int, model_dim: int, seed: int) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.uniform(-1.0, 1.0) for _ in range(model_dim)] for _ in range(num_tokens)]


def main() -> None:
    strategies = ("baseline", "token_drop", "expanded_drop", "priority_backtrack")
    top_k_values = (1, 2, 3, 4)
    seeds = range(10)
    rows = []
    for top_k in top_k_values:
        for strategy in strategies:
            kept_values = []
            imbalance_values = []
            capacity_values = []
            for seed in seeds:
                num_tokens, num_experts, model_dim, hidden_dim = 32, 4, 8, 16
                tokens = make_tokens(num_tokens, model_dim, seed)
                rng = random.Random(seed + 1000)
                router = make_router(num_experts, model_dim, rng)
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
                kept_values.append(sum(len(item) for item in routes) / requested)
                average = sum(loads) / num_experts
                imbalance_values.append(max(loads, default=0) / average if average else 0.0)
                capacity_values.append(capacity)
            rows.append({
                "top_k": top_k,
                "strategy": strategy,
                "capacity": sum(capacity_values) / len(capacity_values),
                "kept": sum(kept_values) / len(kept_values),
                "imbalance": sum(imbalance_values) / len(imbalance_values),
            })

    print("Research question: how does top_k change route pressure and capacity-aware behavior?")
    print("Fixed: T=32, E=4, capacity_factor=1.0, 10 random seeds")
    print()
    print(f"{'top_k':>6s} {'strategy':20s} {'C':>7s} {'kept%':>9s} {'imbalance':>12s}")
    print("-" * 62)
    for row in rows:
        print(
            f"{row['top_k']:6d} {row['strategy']:20s} {row['capacity']:7.2f} "
            f"{row['kept'] * 100:8.2f}% {row['imbalance']:12.3f}"
        )

    analysis = [
        "top_k 增大意味着每个 token 请求更多专家路由，因此总路由数和专家容量都会按公式同步增加。",
        "baseline 始终保留全部 Top-K，但不受容量约束，热点专家仍可能过载。",
        "容量感知策略的保留率取决于候选专家是否足够分散；top_k 越大不代表一定更均衡。",
    ]
    for top_k in top_k_values:
        group = [row for row in rows if row["top_k"] == top_k]
        best = max(group, key=lambda row: row["kept"] if row["strategy"] != "baseline" else -1)
        analysis.append(
            f"top_k={top_k}：容量感知策略中 {best['strategy']} 平均保留率最高，"
            f"为 {best['kept'] * 100:.2f}%。"
        )

    table = (
        "| top_k | 策略 | 平均容量 | 平均保留路由 | 平均不均衡比 |\n"
        "|---:|---|---:|---:|---:|\n"
        + "\n".join(
            f"| {row['top_k']} | {row['strategy']} | {row['capacity']:.2f} | "
            f"{row['kept'] * 100:.2f}% | {row['imbalance']:.3f} |"
            for row in rows
        )
    )
    append_experiment(
        title="实验 7：top_k 扫描",
        question="每个 token 选择更多专家时，容量压力、路由保留率和负载均衡如何变化？",
        setup="T=32，E=4，hidden_dim=16，top_k=1/2/3/4，capacity_factor=1.0，随机种子 0-9",
        result_markdown=table,
        analysis=analysis,
        limitations=[
            "top_k 只改变路由数量，没有测量真实模型精度变化。",
            "实验仍是单卡逻辑模拟，不代表真实分布式通信性能。",
        ],
    )
    print("实验结果已追加到 research_log.md")


if __name__ == "__main__":
    main()
