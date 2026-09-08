"""Statistics and correlation utilities for Module 2."""
from .correlation import (
    autocorrelation,
    cross_correlation,
)

from .basic import (
    Module1Handoff,
    build_m2_input_summary,
    load_module1_handoff,
    save_m2_input_summary,
)

from .wiener import (
    build_correlation_matrix,
    build_cross_correlation_vector,
    solve_wiener_hopf,
)

from .wiener_apply import (
    apply_fir,
    mean_squared_error,
    normalized_correlation,
    root_mean_squared_error,
)

from .performance import (
    explained_power_fraction,
    orthogonality_residual,
    relative_residual_power,
    signal_power,
)

from .system_comparison import (
    align_impulse_responses,
    coefficient_correlation,
    coefficient_error,
    coefficient_mse,
    coefficient_rmse,
    relative_coefficient_error,
)

from .visualization import (
    plot_sequence,
    plot_signal,
    plot_two_sequences,
    plot_two_signals,
)

__all__ = [
    "Module1Handoff",
    "build_m2_input_summary",
    "load_module1_handoff",
    "save_m2_input_summary",
    "autocorrelation",
    "cross_correlation",
    "build_autocorrelation_matrix",
    "build_cross_correlation_vector",
    "solve_wiener_hopf",
    "apply_fir",
    "mean_squared_error",
    "normalized_correlation",
    "root_mean_squared_error",
    "explained_power_fraction",
    "orthogonality_residual",
    "relative_residual_power",
    "signal_power",
    "align_impulse_responses",
    "coefficient_correlation",
    "coefficient_error",
    "coefficient_mse",
    "coefficient_rmse",
    "relative_coefficient_error",
    "plot_sequence",
    "plot_signal",
    "plot_two_sequences",
    "plot_two_signals",
]
