"""Research step 4: controlled candidate-conflict scheduling cases."""

from experiment_logger import append_experiment
from moe_simulator import choose_routes


def make_case(groups: list[tuple[int, list[int]]], num_experts: int = 4):
    """Create deterministic router probabilities from ranked candidates."""
    rank_scores = (0.70, 0.20, 0.08, 0.02)
    probabilities = []
    ranked = []
    for count, order in groups:
        for _ in range(count):
            probs = [0.0] * num_experts
            for rank, expert in enumerate(order):
                probs[expert] = rank_scores[rank]
            probabilities.append(probs)
            ranked.append(order[:])
    return probabilities, ranked


def run_case(name: str, groups, capacity: int):
    probabilities, ranked = make_case(groups)
    top_k = 2
    strategies = ("baseline", "token_drop", "expanded_drop", "priority_backtrack")
    rows = []
    for strategy in strategies:
        routes, loads = choose_routes(
            probabilities,
            ranked,
            top_k=top_k,
            capacity=None if strategy == "baseline" else capacity,
            strategy=strategy,
        )
        requested = len(probabilities) * top_k
        kept = sum(len(item) for item in routes)
        average = sum(loads) / len(loads)
        imbalance = max(loads, default=0) / average if average else 0.0
        # Besides counting how many routes survive, measure route quality:
        # did each token keep its first-choice expert, and how deep in the
        # candidate list are the selected experts on average?
        primary_hits = sum(
            any(expert_id == ranked[token_id][0] for expert_id, _ in token_routes)
            for token_id, token_routes in enumerate(routes)
        )
        rank_values = [
            ranked[token_id].index(expert_id)
            for token_id, token_routes in enumerate(routes)
            for expert_id, _ in token_routes
        ]
        rows.append({
            "case": name,
            "strategy": strategy,
            "capacity": capacity,
            "loads": loads,
            "kept": kept,
            "requested": requested,
            "kept_ratio": kept / requested,
            "imbalance": imbalance,
            "primary_rate": primary_hits / len(probabilities),
            "mean_rank": sum(rank_values) / len(rank_values) if rank_values else 0.0,
        })
    return rows


def main() -> None:
    cases = {
        "shared_fallback": (
            [(12, [0, 1, 2, 3]), (8, [0, 2, 1, 3]), (4, [0, 3, 1, 2])],
            8,
        ),
        "crossed_primary": (
            [(8, [0, 1, 2, 3]), (8, [1, 0, 2, 3]), (8, [0, 1, 3, 2])],
            8,
        ),
        "scarce_backup": (
            [(12, [0, 1, 2, 3]), (6, [0, 1, 2, 3]), (6, [1, 0, 2, 3])],
            10,
        ),
    }
    all_rows = []
    print("Research question: can reservation and local backtracking resolve candidate conflicts?")
    print("Synthetic setup: each token has an explicit ranked candidate list; top_k=2")
    print()
    print(f"{'case':20s} {'strategy':20s} {'C':>4s} {'loads':18s} {'kept':>10s} {'primary%':>10s} {'rank':>8s} {'imbalance':>12s}")
    print("-" * 118)
    for name, (groups, capacity) in cases.items():
        rows = run_case(name, groups, capacity)
        all_rows.extend(rows)
        for row in rows:
            print(
                f"{name:20s} {row['strategy']:20s} {row['capacity']:4d} "
                f"{str(row['loads']):18s} {row['kept']:>4d}/{row['requested']:<5d} "
                f"{row['primary_rate'] * 100:9.1f}% {row['mean_rank']:8.3f} "
                f"{row['imbalance']:12.3f}"
            )

    analysis = []
    for name in cases:
        group = [row for row in all_rows if row["case"] == name]
        expanded = next(row for row in group if row["strategy"] == "expanded_drop")
        priority = next(row for row in group if row["strategy"] == "priority_backtrack")
        analysis.append(
            f"{name}：Expanded Drop 保留 {expanded['kept']}/{expanded['requested']}，"
            f"Priority Backtracking 保留 {priority['kept']}/{priority['requested']}。"
        )
        analysis.append(
            f"{name}：两种策略的不均衡比分别为 {expanded['imbalance']:.3f} 和 {priority['imbalance']:.3f}。"
        )
        analysis.append(
            f"{name}：第一候选覆盖率分别为 {expanded['primary_rate'] * 100:.1f}% 和 "
            f"{priority['primary_rate'] * 100:.1f}%；平均候选排名分别为 "
            f"{expanded['mean_rank']:.3f} 和 {priority['mean_rank']:.3f}（越接近 0 越好）。"
        )
    analysis.extend([
        "这轮实验固定了候选顺序，因此比单纯提高 hot_bias 更能检验冲突处理逻辑。",
        "如果 Priority Backtracking 只在部分冲突图中有优势，说明它是有条件的启发式，而不是总能胜过全局候选池排序。",
        "后续可以继续调节 reservation_fraction 和 backtrack_depth，做真正的消融实验。",
    ])

    table = (
        "| 场景 | 策略 | 容量 | 专家负载 | 保留路由 | 第一候选覆盖率 | 平均候选排名 | 不均衡比 |\n"
        "|---|---|---:|---|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {row['case']} | {row['strategy']} | {row['capacity']} | {row['loads']} | "
            f"{row['kept']}/{row['requested']} | {row['primary_rate'] * 100:.1f}% | "
            f"{row['mean_rank']:.3f} | {row['imbalance']:.3f} |"
            for row in all_rows
        )
    )
    append_experiment(
        title="实验 4：候选专家冲突与局部回溯",
        question="当多个 token 争抢同一专家且备用专家不同，容量预留和局部回溯能否改善调度？",
        setup="3 个合成冲突场景；E=4；top_k=2；每个 token 显式给定候选顺序",
        result_markdown=table,
        analysis=analysis,
        limitations=[
            "候选分数和冲突图是人工构造的，不代表某个真实模型的 Router 分布。",
            "实验验证的是调度逻辑，不包含专家 MLP 质量和真实通信时间。",
        ],
    )
    print()
    print("实验结果已追加到 research_log.md")


if __name__ == "__main__":
    main()
