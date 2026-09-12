"""Research step 6: scan batch size and compare TP/EP preferences."""

from experiment_logger import append_experiment
from simulator import estimate_step_latency


def make_loads(batch_size: int, pattern: str, top_k: int = 2, experts: int = 4) -> list[int]:
    """Build deterministic logical expert loads for a toy decoding batch.

    The two patterns deliberately separate the effects discussed in the paper:
    a small batch can be irregular/hot, while a large batch can be well mixed.
    This is still a cost-model experiment, not a real GPU measurement.
    """
    total_routes = batch_size * top_k
    if pattern == "hotspot":
        # Small/irregular workload: most tokens choose the same expert.
        hot = max(1, round(total_routes * 0.70))
        rest = total_routes - hot
        return [hot, rest, 0, 0]
    if pattern == "balanced":
        base, remainder = divmod(total_routes, experts)
        return [base + (index < remainder) for index in range(experts)]
    raise ValueError("pattern must be hotspot or balanced")


def main() -> None:
    batch_sizes = (1, 2, 4, 8, 16, 32, 64, 128)
    rows = []
    print("Research question: how do batch size and expert-load shape affect TP/EP choice?")
    print()
    print(f"{'pattern':10s} {'batch':>7s} {'loads':18s} {'TP(ms)':>10s} {'EP(ms)':>10s} {'winner':>8s}")
    print("-" * 72)
    for pattern in ("hotspot", "balanced"):
        for batch_size in batch_sizes:
            loads = make_loads(batch_size, pattern)
            tp = estimate_step_latency(loads, top_k=2, mode="TP")
            ep = estimate_step_latency(loads, top_k=2, mode="EP")
            winner = "TP" if tp < ep else "EP"
            row = {
                "pattern": pattern,
                "batch": batch_size,
                "loads": loads,
                "tp": tp,
                "ep": ep,
                "winner": winner,
            }
            rows.append(row)
            print(
                f"{pattern:10s} {batch_size:7d} {str(loads):18s} "
                f"{tp:10.3f} {ep:10.3f} {winner:>8s}"
            )

    analysis = [
        "本实验把 batch size 与专家负载形状分开观察：hotspot 表示路由集中，balanced 表示路由充分混合。",
        "在 hotspot 场景中，EP 的最忙专家和不均衡惩罚会随 batch 增大，TP 通常更稳健。",
        "在 balanced 场景中，EP 能直接让不同逻辑专家处理各自 token，随着 batch 增大，其通信和计算效率更有优势。",
        "因此切换条件不能只看 batch size，还应结合专家负载不均衡度；这也是论文采用运行时请求率和负载信息做策略选择的原因。",
    ]
    for pattern in ("hotspot", "balanced"):
        group = [row for row in rows if row["pattern"] == pattern]
        tp_wins = sum(row["winner"] == "TP" for row in group)
        ep_wins = sum(row["winner"] == "EP" for row in group)
        analysis.append(f"{pattern}：TP 胜出 {tp_wins} 次，EP 胜出 {ep_wins} 次。")

    table = (
        "| 负载形状 | batch size | 专家负载 | TP(ms) | EP(ms) | 推荐策略 |\n"
        "|---|---:|---|---:|---:|---|\n"
        + "\n".join(
            f"| {row['pattern']} | {row['batch']} | {row['loads']} | {row['tp']:.3f} | "
            f"{row['ep']:.3f} | {row['winner']} |"
            for row in rows
        )
    )
    append_experiment(
        title="实验 6：Batch Size 与 TP/EP 策略选择扫描",
        question="batch size 和专家负载形状如何影响 TP/EP 的选择？",
        setup="batch size=1/2/4/8/16/32/64/128；top_k=2；4 个逻辑专家；hotspot 与 balanced 两种负载形状",
        result_markdown=table,
        analysis=analysis,
        limitations=[
            "延迟来自 simulator.py 的教学成本模型，不是真实 GPU、NVLink 或 All-to-All 测量。",
            "hotspot 和 balanced 是人工构造的负载形状，不代表所有真实模型和输入分布。",
        ],
    )
    print()
    print("实验结果已追加到 research_log.md")


if __name__ == "__main__":
    main()
