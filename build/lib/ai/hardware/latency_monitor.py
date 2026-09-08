"""Per-stage latency monitor for the real-time ANC demo.

Timestamps each stage of the processing pipeline and provides rolling
statistics for display on demo slides:

    capture → network → ANC → AI → playback

Usage::

    monitor = LatencyMonitor()
    monitor.mark("capture")
    # ... capture audio ...
    monitor.mark("network")
    # ... receive over UDP ...
    monitor.mark("anc")
    # ... run ANC ...
    monitor.mark("ai")
    # ... run AI enhancement ...
    monitor.mark("playback")
    print(monitor.summary())
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class LatencySnapshot:
    """One measurement of per-stage latencies."""
    stages: dict[str, float] = field(default_factory=dict)
    total_ms: float = 0.0
    timestamp: float = 0.0


class LatencyMonitor:
    """Rolling per-stage latency tracker.

    Parameters
    ----------
    history_size : int
        Number of snapshots to keep for averaging (default 50).
    """

    def __init__(self, history_size: int = 50) -> None:
        self._history: deque[LatencySnapshot] = deque(maxlen=history_size)
        self._marks: list[tuple[str, float]] = []
        self._current_start: float | None = None

    def begin(self) -> None:
        """Start a new measurement cycle."""
        self._marks = []
        self._current_start = time.perf_counter()
        self._marks.append(("begin", self._current_start))

    def mark(self, stage_name: str) -> None:
        """Record a timestamp for the named stage completion."""
        self._marks.append((stage_name, time.perf_counter()))

    def end(self) -> LatencySnapshot:
        """Finalize the current cycle and store the snapshot.

        Returns
        -------
        LatencySnapshot
            Per-stage latencies in milliseconds.
        """
        if not self._marks:
            return LatencySnapshot()

        stages: dict[str, float] = {}
        for i in range(1, len(self._marks)):
            prev_name, prev_time = self._marks[i - 1]
            curr_name, curr_time = self._marks[i]
            stages[curr_name] = (curr_time - prev_time) * 1000.0  # ms

        total = (self._marks[-1][1] - self._marks[0][1]) * 1000.0

        snapshot = LatencySnapshot(
            stages=stages,
            total_ms=total,
            timestamp=time.time(),
        )
        self._history.append(snapshot)
        self._marks = []
        return snapshot

    @property
    def latest(self) -> LatencySnapshot | None:
        """Most recent snapshot, or None."""
        return self._history[-1] if self._history else None

    @property
    def average_ms(self) -> dict[str, float]:
        """Rolling average of per-stage latencies in ms."""
        if not self._history:
            return {}

        all_stages: dict[str, list[float]] = {}
        for snap in self._history:
            for stage, ms in snap.stages.items():
                all_stages.setdefault(stage, []).append(ms)

        return {
            stage: sum(values) / len(values)
            for stage, values in all_stages.items()
        }

    @property
    def average_total_ms(self) -> float:
        """Rolling average total latency in ms."""
        if not self._history:
            return 0.0
        return sum(s.total_ms for s in self._history) / len(self._history)

    def summary(self) -> str:
        """Human-readable summary of recent latency."""
        avg = self.average_ms
        total = self.average_total_ms

        lines = [f"Latency (avg over {len(self._history)} cycles):"]
        for stage, ms in avg.items():
            lines.append(f"  {stage:>12s}: {ms:6.2f} ms")
        lines.append(f"  {'TOTAL':>12s}: {total:6.2f} ms")
        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all history."""
        self._history.clear()
        self._marks = []

    def export_distribution_report(
        self,
        path: str | None = None,
    ) -> dict[str, object]:
        """Export a full latency distribution report.

        Aggregates all history into per-stage and total distributions
        with min/median/mean/max/percentiles — the evidence artifact
        for the latency budget in the final report.

        Parameters
        ----------
        path : str, optional
            If provided, writes the report as JSON to this path.

        Returns
        -------
        dict
            Full distribution report.
        """
        import json
        from pathlib import Path

        if not self._history:
            return {"error": "No latency data collected."}

        # Collect all per-stage values
        all_stages: dict[str, list[float]] = {}
        all_totals: list[float] = []

        for snap in self._history:
            all_totals.append(snap.total_ms)
            for stage, ms in snap.stages.items():
                all_stages.setdefault(stage, []).append(ms)

        def _stats(values: list[float]) -> dict[str, float]:
            import numpy as np
            arr = np.array(values)
            return {
                "count": len(arr),
                "min_ms": float(np.min(arr)),
                "p5_ms": float(np.percentile(arr, 5)),
                "p25_ms": float(np.percentile(arr, 25)),
                "median_ms": float(np.median(arr)),
                "mean_ms": float(np.mean(arr)),
                "p75_ms": float(np.percentile(arr, 75)),
                "p95_ms": float(np.percentile(arr, 95)),
                "p99_ms": float(np.percentile(arr, 99)),
                "max_ms": float(np.max(arr)),
                "std_ms": float(np.std(arr)),
            }

        report: dict[str, object] = {
            "num_measurements": len(self._history),
            "total": _stats(all_totals),
            "per_stage": {
                stage: _stats(values) for stage, values in all_stages.items()
            },
        }

        if path is not None:
            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(report, indent=2) + "\n",
                encoding="utf-8",
            )

        return report
