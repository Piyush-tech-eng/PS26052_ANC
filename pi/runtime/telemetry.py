"""Lightweight telemetry monitor for Raspberry Pi 3."""

from __future__ import annotations

import time
import os


class PiTelemetry:
    """Collects low-overhead CPU/RAM metrics and frame drop counters."""

    def __init__(self) -> None:
        self.frames_sent = 0
        self.frames_dropped = 0
        self.start_time = time.time()

    def get_stats(self) -> dict[str, float]:
        cpu_percent = 0.0
        try:
            load1, _, _ = os.getloadavg()
            cpu_percent = load1 * 25.0  # 4 cores on Pi 3
        except Exception:
            pass

        return {
            "uptime_s": time.time() - self.start_time,
            "frames_sent": self.frames_sent,
            "frames_dropped": self.frames_dropped,
            "estimated_cpu_percent": min(cpu_percent, 100.0),
        }
