"""Configuration-driven Module 3 adaptive-filter laboratory helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from anc.adaptive.convergence import ConvergenceAnalysisResult, analyze_lms_convergence
from anc.adaptive.lms import LMSFilter
from anc.adaptive.nlms import NLMSFilter
from anc.adaptive.wiener_comparison import (
    WienerComparisonResult,
    compare_against_wiener,
)


@dataclass(frozen=True)
class AdaptiveExperimentConfig:
    """Reproducible configuration shared by LMS and NLMS experiments."""

    filter_length: int
    step_size: float
    algorithm: str = "lms"
    epsilon: float = 1e-8
    sampling_rate_hz: int = 8_000
    experiment_seed: int | None = None
    learning_window: int = 256

    def __post_init__(self) -> None:
        if not isinstance(self.filter_length, (int, np.integer)) or isinstance(
            self.filter_length,
            bool,
        ):
            raise TypeError("filter_length must be an integer.")
        if self.filter_length <= 0:
            raise ValueError("filter_length must be positive.")
        if self.algorithm not in {"lms", "nlms"}:
            raise ValueError("algorithm must be 'lms' or 'nlms'.")
        if not np.isfinite(float(self.step_size)) or self.step_size <= 0.0:
            raise ValueError("step_size must be finite and positive.")
        if not np.isfinite(float(self.epsilon)) or self.epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive.")
        if not isinstance(self.sampling_rate_hz, (int, np.integer)) or isinstance(
            self.sampling_rate_hz,
            bool,
        ):
            raise TypeError("sampling_rate_hz must be an integer.")
        if self.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive.")
        if not isinstance(self.learning_window, (int, np.integer)) or isinstance(
            self.learning_window,
            bool,
        ):
            raise TypeError("learning_window must be an integer.")
        if self.learning_window <= 0:
            raise ValueError("learning_window must be positive.")
        if self.experiment_seed is not None and not isinstance(
            self.experiment_seed,
            (int, np.integer),
        ):
            raise TypeError("experiment_seed must be an integer or None.")


@dataclass(frozen=True)
class AdaptiveExperimentResult:
    """Complete online histories and diagnostics from one Module 3 run."""

    output: np.ndarray
    error: np.ndarray
    squared_error: np.ndarray
    coefficient_history: np.ndarray
    final_coefficients: np.ndarray
    mse_learning_curve: np.ndarray
    convergence: ConvergenceAnalysisResult
    wiener_comparison: WienerComparisonResult | None
    config: AdaptiveExperimentConfig


def run_adaptive_experiment(
    reference: np.ndarray,
    desired: np.ndarray,
    config: AdaptiveExperimentConfig,
    *,
    initial_coefficients: np.ndarray | None = None,
    wiener_coefficients: np.ndarray | None = None,
) -> AdaptiveExperimentResult:
    """Run LMS or NLMS and attach standard convergence diagnostics."""

    if not isinstance(config, AdaptiveExperimentConfig):
        raise TypeError("config must be an AdaptiveExperimentConfig.")

    if config.algorithm == "lms":
        filter_instance = LMSFilter(
            filter_length=config.filter_length,
            step_size=config.step_size,
            initial_coefficients=initial_coefficients,
        )
        raw_result = filter_instance.adapt(reference, desired)
        squared_error = raw_result.error ** 2
    else:
        filter_instance = NLMSFilter(
            filter_length=config.filter_length,
            step_size=config.step_size,
            epsilon=config.epsilon,
            initial_coefficients=initial_coefficients,
        )
        raw_result = filter_instance.adapt(reference, desired)
        squared_error = raw_result.squared_error

    convergence = analyze_lms_convergence(
        raw_result.error,
        raw_result.coefficient_history,
        learning_window=config.learning_window,
        stability_window=config.learning_window,
    )

    comparison = None
    if wiener_coefficients is not None:
        comparison = compare_against_wiener(
            raw_result.coefficient_history,
            wiener_coefficients,
        )

    return AdaptiveExperimentResult(
        output=raw_result.output,
        error=raw_result.error,
        squared_error=squared_error,
        coefficient_history=raw_result.coefficient_history,
        final_coefficients=raw_result.final_coefficients,
        mse_learning_curve=convergence.learning_curve,
        convergence=convergence,
        wiener_comparison=comparison,
        config=config,
    )
