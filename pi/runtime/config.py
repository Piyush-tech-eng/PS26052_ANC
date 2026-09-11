"""Raspberry Pi 3 runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PiConfig:
    """Configuration for Pi 3 edge audio node."""

    target_host: str = "192.168.1.100"
    target_port: int = 5005
    sample_rate: int = 16_000
    channels: int = 2
    frame_ms: int = 20
    device_index: int | None = None
    ch0_is_reference: bool = False
    mode: str = "capture_only"  # capture_only, local_anc, full_edge

    @property
    def samples_per_frame(self) -> int:
        return int(self.sample_rate * self.frame_ms / 1000)

    @property
    def flags(self) -> int:
        f = 0
        if self.ch0_is_reference:
            f |= 1
        return f
