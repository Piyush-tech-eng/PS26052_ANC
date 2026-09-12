"""Execution profile validation for PS26052 ANC.

Three profiles control which features are available at runtime:

``development``
    Full research environment: training, experiments, datasets,
    benchmarking, model conversion, all visualization.

``prototype``
    Pi 3 capture → laptop processing → live dashboard → continuous
    playback.  No training dependencies.

``edge``
    All real-time processing on Raspberry Pi / Jetson-class device.
    No laptop dependency, no training dependencies.

Usage::

    from anc.profiles import validate_profile, ExecutionProfile

    validate_profile(ExecutionProfile.PROTOTYPE)
"""

from __future__ import annotations

import importlib
import os
import warnings
from dataclasses import dataclass, field
from typing import Any

from anc.config import ExecutionProfile


def get_active_profile() -> ExecutionProfile:
    """Return the active execution profile.

    Reads from the ``ANC_PROFILE`` environment variable if set
    (``development``, ``prototype``, or ``edge``). Defaults to
    ``ExecutionProfile.PROTOTYPE``.
    """
    env_val = os.environ.get("ANC_PROFILE", "").lower().strip()
    for prof in ExecutionProfile:
        if prof.value == env_val:
            return prof
    return ExecutionProfile.PROTOTYPE


@dataclass
class ProfileSpec:
    """Feature and dependency specification for one execution profile."""

    name: str
    required_packages: list[str] = field(default_factory=list)
    optional_packages: list[str] = field(default_factory=list)
    features: dict[str, bool] = field(default_factory=dict)


_PROFILES: dict[ExecutionProfile, ProfileSpec] = {
    ExecutionProfile.DEVELOPMENT: ProfileSpec(
        name="development",
        required_packages=["numpy", "scipy", "matplotlib"],
        optional_packages=[
            "torch", "torchaudio", "onnx", "onnxruntime",
            "pystoi", "soundfile", "sounddevice", "pyaudio",
        ],
        features={
            "training": True,
            "experiments": True,
            "datasets": True,
            "benchmarking": True,
            "model_conversion": True,
            "hardware_capture": True,
            "local_anc": True,
            "local_ai": True,
            "dashboard": True,
            "playback": True,
        },
    ),
    ExecutionProfile.PROTOTYPE: ProfileSpec(
        name="prototype",
        required_packages=["numpy", "scipy"],
        optional_packages=["onnxruntime", "sounddevice"],
        features={
            "training": False,
            "experiments": False,
            "datasets": False,
            "benchmarking": False,
            "model_conversion": False,
            "hardware_capture": True,
            "local_anc": True,
            "local_ai": True,
            "dashboard": True,
            "playback": True,
        },
    ),
    ExecutionProfile.EDGE: ProfileSpec(
        name="edge",
        required_packages=["numpy"],
        optional_packages=["scipy", "onnxruntime", "pyaudio"],
        features={
            "training": False,
            "experiments": False,
            "datasets": False,
            "benchmarking": False,
            "model_conversion": False,
            "hardware_capture": True,
            "local_anc": True,
            "local_ai": True,
            "dashboard": False,
            "playback": True,
        },
    ),
}


@dataclass
class ValidationResult:
    """Result of profile validation."""

    profile: ExecutionProfile
    valid: bool
    missing_required: list[str] = field(default_factory=list)
    missing_optional: list[str] = field(default_factory=list)
    available_features: dict[str, bool] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        lines = [f"Profile: {self.profile.value}"]
        if self.valid:
            lines.append("  Status: [OK] All required dependencies satisfied")
        else:
            lines.append("  Status: [X] Missing required dependencies")
            for pkg in self.missing_required:
                lines.append(f"    - {pkg}")
        if self.missing_optional:
            lines.append("  Optional (missing):")
            for pkg in self.missing_optional:
                lines.append(f"    - {pkg}")
        lines.append("  Features:")
        for feat, available in self.available_features.items():
            mark = "[OK]" if available else "[X]"
            lines.append(f"    {mark} {feat}")
        return "\n".join(lines)


def _check_package(name: str) -> bool:
    """Check if a Python package is importable."""
    try:
        importlib.import_module(name)
        return True
    except (ImportError, OSError):
        return False


def validate_profile(profile: ExecutionProfile) -> ValidationResult:
    """Validate that the current environment satisfies a profile's requirements.

    Parameters
    ----------
    profile : ExecutionProfile
        The target execution profile.

    Returns
    -------
    ValidationResult
        Detailed validation outcome.

    Raises
    ------
    ValueError
        If required packages are missing (only in strict mode — callers
        can instead check ``result.valid``).
    """
    spec = _PROFILES[profile]

    missing_req = [p for p in spec.required_packages if not _check_package(p)]
    missing_opt = [p for p in spec.optional_packages if not _check_package(p)]

    # Determine effective features — features requiring missing optional
    # packages are degraded.
    feature_pkg_map: dict[str, list[str]] = {
        "training": ["torch", "torchaudio"],
        "local_ai": ["onnxruntime"],
        "playback": ["sounddevice"],
        "hardware_capture": ["pyaudio"],
        "dashboard": [],
        "local_anc": [],
        "experiments": ["matplotlib"],
        "datasets": ["soundfile"],
        "benchmarking": ["matplotlib"],
        "model_conversion": ["onnx", "onnxruntime"],
    }

    available_features: dict[str, bool] = {}
    for feat, enabled in spec.features.items():
        if not enabled:
            available_features[feat] = False
            continue
        required_for_feat = feature_pkg_map.get(feat, [])
        has_deps = all(_check_package(p) for p in required_for_feat)
        available_features[feat] = has_deps
        if enabled and not has_deps:
            dep_list = ", ".join(required_for_feat)
            warnings.warn(
                f"[profile] Feature '{feat}' requires [{dep_list}] but "
                f"some are not installed. Feature will be unavailable.",
                stacklevel=2,
            )

    return ValidationResult(
        profile=profile,
        valid=len(missing_req) == 0,
        missing_required=missing_req,
        missing_optional=missing_opt,
        available_features=available_features,
    )


def get_profile_spec(profile: ExecutionProfile) -> ProfileSpec:
    """Return the specification for a profile."""
    return _PROFILES[profile]
