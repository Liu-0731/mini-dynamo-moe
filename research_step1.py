"""Research step 1: compare routing strategies under repeated seeds.

This is deliberately a small, reproducible study rather than a production
benchmark.  It asks one question: does priority_backtrack reduce imbalance
without dropping more routes than the simpler policies?
"""

import random

from experiment_logger import append_experiment
from moe_simulator import make_experts, make_router, moe_forward


def make_tokens(num_tokens: int, model_dim: int, seed: int) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.uniform(-1.0, 1.0) for _ in range(model_dim)] for _ in range(num_tokens)]


def run_one(seed: int, strategy: str) -> dict[str, float | int | str]:
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
        capacity_factor=1.0,
        strategy=strategy,
    )
    requested = num_tokens * top_k
    kept = sum(len(item) for item in routes)
    average = sum(loads) / num_experts
    imbalance = max(loads, default=0) / average if average else 0.0
    return {
        "strategy": strategy,
        "seed": seed,
        "capacity": capacity,
        "kept_ratio": kept / requested,
        "max_load": max(loads, default=0),
        "imbalance": imbalance,
    }


def main() -> None:
    strategies = ("baseline", "token_drop", "expanded_drop", "priority_backtrack")
    seeds = range(10)
    rows = [run_one(seed, strategy) for strategy in strategies for seed in seeds]

    print("Research question: does priority_backtrack improve balance without extra route loss?")
    print("Fixed: T=32, E=4, hidden_dim=16, top_k=2, capacity_factor=1.0, 10 seeds")
    print()
    print(f"{'strategy':20s} {'kept%':>8s} {'max_load':>10s} {'imbalance':>12s}")
    print("-" * 56)
    summaries: dict[str, dict[str, float]] = {}
    for strategy in strategies:
        group = [row for row in rows if row["strategy"] == strategy]
        kept = sum(float(row["kept_ratio"]) for row in group) / len(group) * 100
        max_load = sum(float(row["max_load"]) for row in group) / len(group)
        imbalance = sum(float(row["imbalance"]) for row in group) / len(group)
        summaries[strategy] = {
            "kept": kept,
            "max_load": max_load,
            "imbalance": imbalance,
        }
        print(f"{strategy:20s} {kept:7.2f}% {max_load:10.2f} {imbalance:12.3f}")

    print()
    print("Interpretation guide:")
    print("- kept% 越高越好：越少丢弃路由。")
    print("- max_load 越低越好：热点专家的工作上限越低。")
    print("- imbalance 越接近 1 越好：各专家越均衡。")
    print("- 这轮只验证路由/容量趋势，不代表真实 GPU 延迟。")

    capacity_aware = ["token_drop", "expanded_drop", "priority_backtrack"]
    best_keep = max(capacity_aware, key=lambda name: summaries[name]["kept"])
    best_balance = min(summaries, key=lambda name: summaries[name]["imbalance"])
    analysis = [
        "baseline 的 100% 保留率是没有容量约束的结果，不能和容量感知策略直接按保留率比较。",
        f"在三个容量感知策略中，{best_keep} 的平均路由保留率最高（{summaries[best_keep]['kept']:.2f}%）。",
        f"{best_balance} 的平均负载最均衡（不均衡比 {summaries[best_balance]['imbalance']:.3f}）。",
        f"Token Drop 的平均最大负载为 {summaries['token_drop']['max_load']:.2f}，但会损失约 {100 - summaries['token_drop']['kept']:.2f}% 的路由。",
        "priority_backtrack 是启发式局部回溯，不等于严格的全局最优匹配；下一轮应在更小容量和更强热点下继续验证。",
    ]
    table = (
        "| 策略 | 平均保留路由 | 平均最大负载 | 平均不均衡比 |\n"
        "|---|---:|---:|---:|\n"
        + "\n".join(
            f"| {name} | {summaries[name]['kept']:.2f}% | "
            f"{summaries[name]['max_load']:.2f} | {summaries[name]['imbalance']:.3f} |"
            for name in strategies
        )
    )
    append_experiment(
        title="实验 1：四种容量路由策略重复比较",
        question="priority_backtrack 是否能在不额外丢失路由的情况下改善专家负载均衡？",
        setup="T=32，E=4，hidden_dim=16，top_k=2，capacity_factor=1.0，随机种子 0-9",
        result_markdown=table,
        analysis=analysis,
        limitations=[
            "这是单卡逻辑模拟，没有测量真实 GPU、All-to-All 或 NVLink 延迟。",
            "10 个随机种子只能说明当前合成路由分布下的趋势，不能代表所有模型。",
        ],
    )
    print("实验结果已追加到 research_log.md")


if __name__ == "__main__":
    main()
