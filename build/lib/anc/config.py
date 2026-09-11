"""Centralized configuration for the PS26052 ANC system.

Provides a single ``ANCConfig`` dataclass that encapsulates all tuneable
parameters for every execution profile (development / prototype / edge).
Configuration sources, in priority order:

1. Constructor keyword arguments (highest)
2. Environment variables (``ANC_SAMPLE_RATE``, ``ANC_PROFILE``, etc.)
3. Configuration file (YAML or JSON)
4. Built-in defaults (lowest)

Usage::

    from anc.config import ANCConfig, load_config

    # Defaults
    cfg = ANCConfig()

    # From file
    cfg = load_config("configs/prototype.yaml")

    # Override at construction
    cfg = ANCConfig(sample_rate=48000, frame_ms=10)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any


class ExecutionProfile(str, Enum):
    """Execution profile — controls which features are available."""

    DEVELOPMENT = "development"
    PROTOTYPE = "prototype"
    EDGE = "edge"


@dataclass
class ANCConfig:
    """Complete system configuration.

    All numeric defaults target a 16 kHz, 20 ms-frame laptop prototype.
    """

    # ── Execution profile ────────────────────────────────────────────
    profile: ExecutionProfile = ExecutionProfile.PROTOTYPE

    # ── Audio ────────────────────────────────────────────────────────
    sample_rate: int = 16_000
    frame_ms: int = 20
    channels: int = 2

    # ── Microphone channel mapping ───────────────────────────────────
    reference_channel: int = 1    # Mic facing the noise source
    error_channel: int = 0        # Mic at the cancellation point

    # ── ANC (classical) ─────────────────────────────────────────────
    anc_enabled: bool = True
    filter_length: int = 64
    step_size: float = 0.01
    epsilon: float = 1e-6
    algorithm: str = "fxnlms"

    # ── AI enhancement ───────────────────────────────────────────────
    ai_enabled: bool = True
    model_name: str = "auto"
    model_precision: str = "fp32"   # fp32, fp16, int8

    # ── Transport / networking ───────────────────────────────────────
    udp_host: str = "0.0.0.0"
    udp_port: int = 5005
    jitter_buffer_depth: int = 1

    # ── Dashboard ────────────────────────────────────────────────────
    dashboard_port: int = 8080
    telemetry_rate_hz: float = 10.0
    waveform_rate_hz: float = 30.0

    # ── Buffers ──────────────────────────────────────────────────────
    input_buffer_frames: int = 100
    output_buffer_frames: int = 200
    ring_buffer_seconds: float = 2.0

    # ── Playback ─────────────────────────────────────────────────────
    playback_block_size: int = 512

    # ── Telemetry / status ───────────────────────────────────────────
    status_file: str = "results/demo_status.json"

    # ── Calibration ──────────────────────────────────────────────────
    calibration_duration_s: float = 3.0

    # ── Derived (computed) ───────────────────────────────────────────

    @property
    def frame_samples(self) -> int:
        """Number of samples per frame."""
        return int(self.sample_rate * self.frame_ms / 1000)

    @property
    def frame_duration_s(self) -> float:
        """Frame duration in seconds."""
        return self.frame_ms / 1000.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        d: dict[str, Any] = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, Enum):
                val = val.value
            d[f.name] = val
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ANCConfig":
        """Construct from a dictionary, ignoring unknown keys."""
        known = {f.name for f in fields(cls)}
        filtered = {}
        for k, v in data.items():
            if k not in known:
                continue
            if k == "profile":
                v = ExecutionProfile(v)
            filtered[k] = v
        return cls(**filtered)


def _str_to_bool(s: str) -> bool:
    """Convert common boolean string representations."""
    return s.lower() in ("1", "true", "yes", "on")


def load_config(path: str | Path | None = None) -> ANCConfig:
    """Load configuration from file, env vars, and defaults.

    Parameters
    ----------
    path : str or Path, optional
        Path to a YAML or JSON config file.  If None, only env vars
        and built-in defaults are used.

    Returns
    -------
    ANCConfig
    """
    data: dict[str, Any] = {}

    # 1. File
    if path is not None:
        p = Path(path)
        if p.exists():
            text = p.read_text(encoding="utf-8")
            if p.suffix in (".yaml", ".yml"):
                try:
                    import yaml  # type: ignore[import-untyped]
                    data = yaml.safe_load(text) or {}
                except ImportError:
                    # Fall back to JSON-like subset
                    pass
            else:
                data = json.loads(text)

    # 2. Environment variable overrides (ANC_SAMPLE_RATE, ANC_PROFILE, etc.)
    _env_overrides = {
        "ANC_PROFILE": ("profile", str),
        "ANC_SAMPLE_RATE": ("sample_rate", int),
        "ANC_FRAME_MS": ("frame_ms", int),
        "ANC_FILTER_LENGTH": ("filter_length", int),
        "ANC_STEP_SIZE": ("step_size", float),
        "ANC_MODEL": ("model_name", str),
        "ANC_UDP_PORT": ("udp_port", int),
        "ANC_DASHBOARD_PORT": ("dashboard_port", int),
        "ANC_REFERENCE_CHANNEL": ("reference_channel", int),
        "ANC_ERROR_CHANNEL": ("error_channel", int),
        "ANC_ANC_ENABLED": ("anc_enabled", _str_to_bool),
        "ANC_AI_ENABLED": ("ai_enabled", _str_to_bool),
    }
    for env_key, (field_name, converter) in _env_overrides.items():
        env_val = os.environ.get(env_key)
        if env_val is not None:
            try:
                data[field_name] = converter(env_val)
            except (ValueError, TypeError):
                pass

    return ANCConfig.from_dict(data)
