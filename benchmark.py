"""Run all demonstrations for the Mini-Dynamo-MoE project."""

import random

from controller import AdaptiveParallelismController
from experiment_logger import append_experiment
from moe_simulator import make_experts, make_router, moe_forward
from simulator import CostModel, dynamic_latency_trace, estimate_step_latency


def make_tokens(num_tokens: int, model_dim: int, seed: int) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.uniform(-1.0, 1.0) for _ in range(model_dim)] for _ in range(num_tokens)]


def run_moe_demo() -> list[dict[str, object]]:
    rng = random.Random(7)
    num_tokens, num_experts, model_dim, hidden_dim, top_k = 32, 4, 8, 16, 2
    tokens = make_tokens(num_tokens, model_dim, 7)
    router = make_router(num_experts, model_dim, rng)
    experts = make_experts(num_experts, model_dim, hidden_dim, rng)

    print("=== 真实 MoE 前向流程（单卡模拟） ===")
    rows: list[dict[str, object]] = []
    for strategy in (
        "baseline",
        "token_drop",
        "expanded_drop",
        "priority_backtrack",
    ):
        outputs, routes, loads, capacity = moe_forward(
            tokens, router, experts, top_k, capacity_factor=1.0, strategy=strategy
        )
        kept = sum(len(item) for item in routes)
        rows.append({
            "strategy": strategy,
            "capacity": capacity,
            "loads": loads,
            "kept": kept,
            "requested": num_tokens * top_k,
        })
        print(
            f"{strategy:14s} capacity={capacity:2d} "
            f"loads={loads} routes={kept}/{num_tokens * top_k} "
            f"output0={tuple(round(x, 3) for x in outputs[0][:3])}"
        )
    return rows


def run_parallelism_demo() -> list[dict[str, object]]:
    print("\n=== TP / EP 延迟模型 ===")
    examples = {
        "低负载": [4, 3, 2, 1],
        "高负载且不均衡": [32, 12, 13, 7],
        "高负载且已均衡": [16, 16, 16, 16],
    }
    rows: list[dict[str, object]] = []
    for name, loads in examples.items():
        tp = estimate_step_latency(loads, top_k=2, mode="TP")
        ep = estimate_step_latency(loads, top_k=2, mode="EP")
        rows.append({"name": name, "loads": loads, "tp": tp, "ep": ep})
        print(f"{name:12s} loads={loads} TP={tp:.2f} ms EP={ep:.2f} ms")
    return rows


def run_dynamic_demo() -> list[dict[str, object]]:
    print("\n=== 动态 TP / EP 切换 ===")
    request_rates = [4, 8, 32, 40, 50, 50, 8, 4, 2]
    # As request rate increases, the decoding batch grows.
    loads_by_interval = [
        [4, 3, 2, 1],
        [8, 5, 2, 1],
        [32, 12, 13, 7],
        [32, 12, 13, 7],
        [32, 12, 13, 7],
        [32, 12, 13, 7],
        [8, 5, 2, 1],
        [4, 3, 2, 1],
        [4, 3, 2, 1],
    ]
    trace = dynamic_latency_trace(
        request_rates, loads_by_interval, top_k=2, threshold=20, counter_bits=2
    )
    for item in trace:
        event = f"  [{item['event']}]" if item["event"] else ""
        print(
            f"RPS={item['request_rate']:>4} C={item['counter']} "
            f"mode={item['mode']} latency={item['latency_ms']:.2f} ms{event}"
        )
    return trace


def main() -> None:
    moe_rows = run_moe_demo()
    parallel_rows = run_parallelism_demo()
    trace = run_dynamic_demo()

    moe_table = (
        "| 策略 | 容量 | 专家负载 | 保留路由 |\n"
        "|---|---:|---|---:|\n"
        + "\n".join(
            f"| {row['strategy']} | {row['capacity']} | {row['loads']} | "
            f"{row['kept']}/{row['requested']} |"
            for row in moe_rows
        )
    )
    parallel_table = (
        "| 负载场景 | TP(ms) | EP(ms) |\n"
        "|---|---:|---:|\n"
        + "\n".join(
            f"| {row['name']} | {row['tp']:.2f} | {row['ep']:.2f} |"
            for row in parallel_rows
        )
    )
    switch_events = [item["event"] for item in trace if item["event"]]
    result = (
        "### MoE 路由\n\n"
        + moe_table
        + "\n\n### TP / EP 成本模型\n\n"
        + parallel_table
        + "\n\n### 动态切换\n\n"
        + "| RPS | counter | mode | latency(ms) | event |\n"
        + "|---:|---:|---|---:|---|\n"
        + "\n".join(
            f"| {item['request_rate']} | {item['counter']} | {item['mode']} | "
            f"{item['latency_ms']:.2f} | {item['event'] or '-'} |"
            for item in trace
        )
    )
    analysis = [
        "容量感知策略把热点专家的最大负载压到容量上限，priority_backtrack 同时尝试保留更多路由。",
        "在当前成本模型中，负载严重不均衡时 EP 会受到最忙专家和不均衡惩罚影响。",
        f"本次请求率序列触发了 {len(switch_events)} 次模式切换：{', '.join(switch_events) if switch_events else '无'}。",
        "切换瞬间包含额外切换成本，因此不能只比较 TP/EP 的稳态延迟。",
    ]
    append_experiment(
        title="综合实验：MoE 路由、TP/EP 延迟与动态切换",
        question="容量感知路由和请求率驱动的 TP/EP 策略是否能改善不同负载下的系统行为？",
        setup="benchmark.py 默认配置；单卡逻辑专家；TP/EP 使用教学成本模型",
        result_markdown=result,
        analysis=analysis,
        limitations=[
            "TP/EP 延迟是人为成本模型，不是 RTX 5060 或多 GPU 的真实测量。",
            "当前综合实验没有加载真实语言模型，也没有测量模型准确率。",
        ],
    )
    print("实验结果已追加到 research_log.md")


if __name__ == "__main__":
    main()
