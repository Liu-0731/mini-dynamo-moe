"""Research step 9: model overlap between expert-weight transfer and compute."""

from experiment_logger import append_experiment
from simulator import (
    combine_compute_and_transfer_ms,
    estimate_step_latency,
    estimate_weight_transfer_ms,
)


def main() -> None:
    scenarios = {
        "balanced_EP": [16, 16, 16, 16],
        "hotspot_EP": [32, 12, 13, 7],
    }
    bandwidths = (16.0, 32.0, 64.0)
    missed_experts_values = (1, 2, 4)
    expert_weight_mb = 512.0
    synchronization_ms = 0.02
    rows = []
    print("Research question: how much can concurrent expert-weight transfer be hidden by computation?")
    print(f"Fixed expert weight={expert_weight_mb:.0f} MB; synchronization={synchronization_ms:.2f} ms")
    print()
    print(f"{'scenario':15s} {'BW(Gbps)':>9s} {'missed':>8s} {'compute':>10s} {'transfer':>10s} {'serial':>10s} {'overlap':>10s} {'masked%':>9s}")
    print("-" * 100)
    for scenario, loads in scenarios.items():
        compute = estimate_step_latency(loads, top_k=2, mode="EP")
        for bandwidth in bandwidths:
            for missed in missed_experts_values:
                transfer = estimate_weight_transfer_ms(
                    missed,
                    expert_weight_mb=expert_weight_mb,
                    link_bandwidth_gbps=bandwidth,
                )
                serial = combine_compute_and_transfer_ms(compute, transfer, overlap=False)
                overlapped = combine_compute_and_transfer_ms(
                    compute,
                    transfer,
                    overlap=True,
                    synchronization_ms=synchronization_ms,
                )
                masked = max(0.0, min(1.0, (serial - overlapped) / transfer)) if transfer else 0.0
                row = {
                    "scenario": scenario,
                    "bandwidth": bandwidth,
                    "missed": missed,
                    "compute": compute,
                    "transfer": transfer,
                    "serial": serial,
                    "overlap": overlapped,
                    "masked": masked,
                }
                rows.append(row)
                print(
                    f"{scenario:15s} {bandwidth:9.0f} {missed:8d} {compute:10.3f} "
                    f"{transfer:10.3f} {serial:10.3f} {overlapped:10.3f} {masked * 100:8.1f}%"
                )

    analysis = [
        "serial 是先搬运完专家权重、再计算；overlap 是让权重搬运和已有计算并行，阶段耗时近似取两者较大值，再加同步开销。",
        "链路带宽越高，transfer 时间越短，越容易被计算阶段隐藏。",
        "当 transfer 小于 compute 时，重叠后的主要成本仍由计算决定；这对应论文中‘搬运成本可被并发计算掩盖’的直觉。",
        "当缺失专家很多或权重很大时，transfer 可能超过 compute，此时即使重叠也不能完全隐藏搬运成本。",
    ]
    table = (
        "| 场景 | 带宽(Gbps) | 缺失专家数 | 计算(ms) | 搬运(ms) | 串行(ms) | 重叠(ms) | 搬运隐藏比例 |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {row['scenario']} | {row['bandwidth']:.0f} | {row['missed']} | {row['compute']:.3f} | "
            f"{row['transfer']:.3f} | {row['serial']:.3f} | {row['overlap']:.3f} | {row['masked'] * 100:.1f}% |"
            for row in rows
        )
    )
    append_experiment(
        title="实验 9：计算与专家权重搬运重叠成本模型",
        question="并发计算能否隐藏加载缺失专家权重的额外成本？",
        setup="balanced/hotspot 两种 EP 负载；专家权重 512 MB；带宽 16/32/64 Gbps；缺失专家数 1/2/4",
        result_markdown=table,
        analysis=analysis,
        limitations=[
            "这是串行与 max(compute, transfer) 的教学模型，不执行真实 DMA、CUDA Stream 或 GPU 拷贝。",
            "权重大小和带宽是人为参数，不能直接当作特定硬件的实测结果。",
        ],
    )
    print("实验结果已追加到 research_log.md")


if __name__ == "__main__":
    main()
