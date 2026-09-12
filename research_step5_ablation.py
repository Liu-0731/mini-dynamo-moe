"""Research step 5: ablate reservation and backtracking parameters."""

from experiment_logger import append_experiment
from moe_simulator import choose_routes
from research_step4_conflict import make_case


def evaluate(groups, capacity: int, reservation_fraction: float, backtrack_depth: int):
    probabilities, ranked = make_case(groups)
    routes, loads = choose_routes(
        probabilities,
        ranked,
        top_k=2,
        capacity=capacity,
        strategy="priority_backtrack",
        reservation_fraction=reservation_fraction,
        backtrack_depth=backtrack_depth,
    )
    requested = len(probabilities) * 2
    kept = sum(len(item) for item in routes)
    primary_hits = sum(
        any(expert_id == ranked[token_id][0] for expert_id, _ in token_routes)
        for token_id, token_routes in enumerate(routes)
    )
    rank_values = [
        ranked[token_id].index(expert_id)
        for token_id, token_routes in enumerate(routes)
        for expert_id, _ in token_routes
    ]
    average = sum(loads) / len(loads)
    return {
        "kept": kept,
        "requested": requested,
        "kept_ratio": kept / requested,
        "loads": loads,
        "imbalance": max(loads, default=0) / average if average else 0.0,
        "primary_rate": primary_hits / len(probabilities),
        "mean_rank": sum(rank_values) / len(rank_values) if rank_values else 0.0,
    }


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
    reservation_values = (0.0, 0.25, 0.5)
    depth_values = (0, 1, 2, 3)
    rows = []
    print("Research question: how sensitive is priority backtracking to its two heuristic parameters?")
    print("Lower mean_rank is better; primary% is the fraction of tokens retaining their first choice.")
    print()
    print(f"{'case':20s} {'reserve':>8s} {'depth':>6s} {'kept':>10s} {'primary%':>10s} {'rank':>8s} {'imbalance':>12s}")
    print("-" * 86)
    for name, (groups, capacity) in cases.items():
        for reservation_fraction in reservation_values:
            for backtrack_depth in depth_values:
                result = evaluate(groups, capacity, reservation_fraction, backtrack_depth)
                row = {
                    "case": name,
                    "capacity": capacity,
                    "reserve": reservation_fraction,
                    "depth": backtrack_depth,
                    **result,
                }
                rows.append(row)
                print(
                    f"{name:20s} {reservation_fraction:8.2f} {backtrack_depth:6d} "
                    f"{result['kept']:>4d}/{result['requested']:<5d} "
                    f"{result['primary_rate'] * 100:9.1f}% {result['mean_rank']:8.3f} "
                    f"{result['imbalance']:12.3f}"
                )

    analysis = [
        "本实验只改变 priority_backtrack 的 reservation_fraction 和 backtrack_depth，其他条件保持不变。",
        "reserve 越大，越早为第二候选保留容量；它可能减少第一候选的可用空间，也可能降低后续冲突。",
        "depth 越大，局部回溯搜索越深，通常能修复更多冲突，但调度开销也会增加。",
    ]
    for name in cases:
        group = [row for row in rows if row["case"] == name]
        best_rank = min(group, key=lambda row: (row["mean_rank"], -row["kept"]))
        best_primary = max(group, key=lambda row: (row["primary_rate"], row["kept_ratio"]))
        analysis.append(
            f"{name}：平均候选排名最优为 reserve={best_rank['reserve']:.2f}, "
            f"depth={best_rank['depth']}（rank={best_rank['mean_rank']:.3f}）；"
            f"第一候选覆盖率最优为 reserve={best_primary['reserve']:.2f}, "
            f"depth={best_primary['depth']}（{best_primary['primary_rate'] * 100:.1f}%）。"
        )

    table = (
        "| 场景 | 预留比例 | 回溯深度 | 保留路由 | 第一候选覆盖率 | 平均候选排名 | 不均衡比 |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {row['case']} | {row['reserve']:.2f} | {row['depth']} | "
            f"{row['kept']}/{row['requested']} | {row['primary_rate'] * 100:.1f}% | "
            f"{row['mean_rank']:.3f} | {row['imbalance']:.3f} |"
            for row in rows
        )
    )
    append_experiment(
        title="实验 5：预留比例与局部回溯深度消融",
        question="priority_backtrack 的效果是否依赖预留比例和回溯深度？",
        setup="3 个人工候选冲突场景；reservation_fraction=0/0.25/0.5；backtrack_depth=0/1/2/3",
        result_markdown=table,
        analysis=analysis,
        limitations=[
            "参数扫描仍基于人工构造的候选冲突图，不能替代真实模型 Router 分布。",
            "回溯深度增加带来的计算开销只在算法步骤上体现，没有换算成真实 GPU 延迟。",
        ],
    )
    print()
    print("实验结果已追加到 research_log.md")


if __name__ == "__main__":
    main()
