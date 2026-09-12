"""Adaptive TP/EP controller based on the paper's hysteresis counter."""


class AdaptiveParallelismController:
    """Switch between TP and EP without reacting to one noisy measurement."""

    def __init__(
        self,
        threshold: float = 20.0,
        counter_bits: int = 2,
        initial_mode: str = "TP",
    ) -> None:
        if counter_bits < 1:
            raise ValueError("counter_bits must be positive")
        if initial_mode not in {"TP", "EP"}:
            raise ValueError("initial_mode must be TP or EP")
        self.threshold = threshold
        self.maximum = (2**counter_bits) - 1
        self.counter = 0 if initial_mode == "EP" else self.maximum
        self.mode = initial_mode

    def observe(self, request_rate: float) -> dict[str, object]:
        """Observe one interval and return the new state plus a switch event."""
        old_mode = self.mode
        if request_rate > self.threshold:
            self.counter = min(self.counter + 1, self.maximum)
        elif request_rate < self.threshold:
            self.counter = max(self.counter - 1, 0)

        if self.counter == self.maximum:
            self.mode = "EP"
        elif self.counter == 0:
            self.mode = "TP"

        return {
            "request_rate": request_rate,
            "counter": self.counter,
            "mode": self.mode,
            "switched": self.mode != old_mode,
            "event": f"{old_mode} -> {self.mode}" if self.mode != old_mode else "",
        }


if __name__ == "__main__":
    controller = AdaptiveParallelismController(threshold=20, counter_bits=2)
    for rate in [4, 8, 32, 40, 50, 8, 4, 2]:
        print(controller.observe(rate))
