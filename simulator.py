"""Single-card cost model for the distributed parts of Dynamo-MoE.

The model uses logical devices.  It does not claim to measure real NVLink or
InfiniBand performance; it makes the paper's trade-offs visible on one GPU.
"""

from dataclasses import dataclass


@dataclass
class CostModel:
    """Toy costs, expressed in milliseconds."""

    compute_per_route_ms: float = 0.08
    tp_collective_per_token_ms: float = 0.025
    ep_dispatch_per_route_ms: float = 0.012
    ep_imbalance_penalty_ms: float = 0.15
    switch_per_layer_ms: float = 0.40
    moe_layers: int = 4


def estimate_step_latency(
    expert_loads: list[int],
    top_k: int,
    mode: str,
    num_devices: int = 4,
    cost: CostModel | None = None,
) -> float:
    """Estimate one MoE step from route counts and a TP/EP strategy."""
    if cost is None:
        cost = CostModel()
    if mode not in {"TP", "EP"}:
        raise ValueError("mode must be TP or EP")

    total_routes = sum(expert_loads)
    total_tokens = max(1, total_routes // max(top_k, 1))
    if mode == "TP":
        # TP shares work evenly, then pays a collective communication cost.
        compute = total_routes * cost.compute_per_route_ms / max(num_devices, 1)
        communication = total_tokens * cost.tp_collective_per_token_ms
    else:
        # EP is governed by the busiest logical device/expert.
        compute = max(expert_loads, default=0) * cost.compute_per_route_ms
        communication = total_routes * cost.ep_dispatch_per_route_ms
        average = total_routes / max(len(expert_loads), 1)
        imbalance = max(expert_loads, default=0) / max(average, 1e-9)
        communication += max(0.0, imbalance - 1.0) * cost.ep_imbalance_penalty_ms
    return compute + communication


def estimate_switch_cost(cost: CostModel | None = None) -> float:
    if cost is None:
        cost = CostModel()
    return cost.switch_per_layer_ms * cost.moe_layers


def estimate_weight_transfer_ms(
    missed_experts: int,
    expert_weight_mb: float = 128.0,
    link_bandwidth_gbps: float = 32.0,
) -> float:
    """Estimate host-to-device transfer time for missing experts.

    This is a bandwidth-only teaching model.  It assumes each missing expert
    has the same weight size and that the link is fully utilized.  It does not
    model PCIe/NVLink contention, DMA setup details, or real CUDA streams.
    """
    if missed_experts <= 0:
        return 0.0
    if expert_weight_mb < 0:
        raise ValueError("expert_weight_mb must be non-negative")
    if link_bandwidth_gbps <= 0:
        raise ValueError("link_bandwidth_gbps must be positive")
    total_bits = missed_experts * expert_weight_mb * 8.0
    return total_bits / (link_bandwidth_gbps * 1000.0)


def combine_compute_and_transfer_ms(
    compute_ms: float,
    transfer_ms: float,
    overlap: bool,
    synchronization_ms: float = 0.0,
) -> float:
    """Combine compute and weight-transfer stages under two schedules.

    Serial execution adds both durations.  With overlap, transfer can be
    hidden by computation and the stage takes approximately the longer one,
    plus any explicit synchronization cost.
    """
    if compute_ms < 0 or transfer_ms < 0 or synchronization_ms < 0:
        raise ValueError("costs must be non-negative")
    if overlap:
        return max(compute_ms, transfer_ms) + synchronization_ms
    return compute_ms + transfer_ms


def dynamic_latency_trace(
    request_rates: list[float],
    expert_loads_by_interval: list[list[int]],
    top_k: int,
    threshold: float = 20.0,
    counter_bits: int = 2,
    num_devices: int = 4,
    cost: CostModel | None = None,
) -> list[dict[str, object]]:
    """Run the adaptive policy over a sequence of workload intervals."""
    from controller import AdaptiveParallelismController

    controller = AdaptiveParallelismController(threshold, counter_bits)
    trace: list[dict[str, object]] = []
    previous_mode = controller.mode
    for request_rate, loads in zip(request_rates, expert_loads_by_interval):
        state = controller.observe(request_rate)
        latency = estimate_step_latency(loads, top_k, state["mode"], num_devices, cost)
        if state["mode"] != previous_mode:
            latency += estimate_switch_cost(cost)
        trace.append({**state, "latency_ms": latency, "loads": loads})
        previous_mode = state["mode"]
    return trace
