"""Research step 8: compare hysteresis counter widths for TP/EP switching."""

from experiment_logger import append_experiment
from simulator import dynamic_latency_trace


def main() -> None:
    request_rates = [4, 8, 32, 40, 50, 50, 8, 4, 2]
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
    rows = []
    print("Research question: how does hysteresis counter width affect switching stability?")
    print()
    print(f"{'bits':>6s} {'range':>12s} {'switches':>10s} {'total_ms':>12s} {'modes':30s}")
    print("-" * 76)
    for counter_bits in (1, 2, 3, 4):
        trace = dynamic_latency_trace(
            request_rates,
            loads_by_interval,
            top_k=2,
            threshold=20,
            counter_bits=counter_bits,
        )
        events = [item["event"] for item in trace if item["event"]]
        total_latency = sum(float(item["latency_ms"]) for item in trace)
        modes = "→".join(str(item["mode"]) for item in trace)
        maximum = (2**counter_bits) - 1
        row = {
            "bits": counter_bits,
            "range": f"0..{maximum}",
            "switches": len(events),
            "total": total_latency,
            "events": events,
            "modes": modes,
        }
        rows.append(row)
        print(
            f"{counter_bits:6d} {row['range']:>12s} {row['switches']:10d} "
            f"{row['total']:12.3f} {modes:30s}"
        )

    analysis = [
        "n 位计数器的状态范围是 0 到 2^n-1；只有计数器达到两端才触发模式切换。",
        "位数越大，系统需要连续更多次的高负载或低负载观测才能切换，因此抗瞬时抖动能力更强。",
        "位数过大也会增加策略滞后：工作负载已经改变，系统仍可能暂时停留在旧模式。",
        "本序列中切换次数可能相同，但切换发生时刻和切换带来的总成本会不同，不能只看次数。",
    ]
    table = (
        "| 计数器位数 | 状态范围 | 切换次数 | 总延迟(ms) | 模式序列 |\n"
        "|---:|---|---:|---:|---|\n"
        + "\n".join(
            f"| {row['bits']} | {row['range']} | {row['switches']} | {row['total']:.3f} | "
            f"{row['modes']} |"
            for row in rows
        )
    )
    append_experiment(
        title="实验 8：n 位滞回计数器比较",
        question="计数器位数如何影响 TP/EP 切换的稳定性和响应速度？",
        setup="同一请求率序列；threshold=20；counter_bits=1/2/3/4；切换成本纳入总延迟",
        result_markdown=table,
        analysis=analysis,
        limitations=[
            "请求率序列是人工构造的，不能代表所有在线流量。",
            "计数器只模拟策略决策，不包含真实服务队列和用户级 SLA。",
        ],
    )
    print("实验结果已追加到 research_log.md")


if __name__ == "__main__":
    main()
