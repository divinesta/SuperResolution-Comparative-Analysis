"""Warm-up and repeated wall-clock timing utilities."""

from collections.abc import Callable
from dataclasses import asdict, dataclass
from statistics import mean, median, pstdev
from time import perf_counter_ns
from typing import TypeVar


ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class TimingStats:
    """Summary of repeated single-image runtime measurements."""

    latency_mean_ms: float
    latency_median_ms: float
    latency_std_ms: float
    latency_min_ms: float
    latency_max_ms: float
    warmup_runs: int
    timed_runs: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def measure_runtime(
    operation: Callable[[], ResultT],
    *,
    warmup_runs: int = 3,
    timed_runs: int = 10,
) -> tuple[ResultT, TimingStats]:
    """Run an operation repeatedly and return its final result and timing summary."""
    if warmup_runs < 0:
        raise ValueError("warmup_runs cannot be negative.")
    if timed_runs <= 0:
        raise ValueError("timed_runs must be greater than zero.")

    for _ in range(warmup_runs):
        operation()

    durations_ms: list[float] = []
    result: ResultT | None = None
    for _ in range(timed_runs):
        start_ns = perf_counter_ns()
        result = operation()
        durations_ms.append((perf_counter_ns() - start_ns) / 1_000_000.0)

    stats = TimingStats(
        latency_mean_ms=mean(durations_ms),
        latency_median_ms=median(durations_ms),
        latency_std_ms=pstdev(durations_ms),
        latency_min_ms=min(durations_ms),
        latency_max_ms=max(durations_ms),
        warmup_runs=warmup_runs,
        timed_runs=timed_runs,
    )
    return result, stats  # type: ignore[return-value]
