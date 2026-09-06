"""M6.02 — Calibration and Explicit Signal Alignment.

This experiment validates the Module 6 timing-calibration layer.

Pipeline:

    calibration reference
            ↓
    known timing offset
            ↓
    simulated measured target
            ↓
    delay estimation
            ↓
    explicit alignment
            ↓
    verification

The experiment is intentionally separate from the ANC controller.

Timing calibration should happen before recorded/measured signals are
passed into the ANC replay pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from anc.calibration import (
    align_signals,
    apply_known_delay,
    estimate_delay,
)


def correlation(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """Compute normalized zero-mean correlation."""

    first_centered = (
        first
        - np.mean(first)
    )

    second_centered = (
        second
        - np.mean(second)
    )

    denominator = (
        np.linalg.norm(
            first_centered
        )
        * np.linalg.norm(
            second_centered
        )
    )

    if denominator <= np.finfo(
        np.float64
    ).eps:
        return 0.0

    return float(
        np.dot(
            first_centered,
            second_centered,
        )
        / denominator
    )


def main() -> None:

    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    results_dir = (
        project_root
        / "results"
        / "m6_02_calibration_alignment"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------
    # Reproducibility
    # -------------------------------------------------

    seed = 26052

    rng = np.random.default_rng(
        seed
    )

    # -------------------------------------------------
    # Signal configuration
    # -------------------------------------------------

    sampling_rate_hz = 8_000

    duration_seconds = 3.0

    num_samples = int(
        sampling_rate_hz
        * duration_seconds
    )

    # -------------------------------------------------
    # Calibration reference
    #
    # A coloured broadband signal is used so the timing
    # estimation has identifiable temporal structure.
    # -------------------------------------------------

    white = rng.normal(
        0.0,
        0.25,
        size=num_samples,
    )

    reference = np.empty(
        num_samples,
        dtype=np.float64,
    )

    previous = 0.0

    alpha = 0.86

    for index, sample in enumerate(
        white
    ):

        previous = (
            alpha
            * previous
            + (
                1.0
                - alpha
            )
            * sample
        )

        reference[index] = (
            previous
        )

    # -------------------------------------------------
    # Simulated measured signal
    #
    # In a real experiment this would come from a
    # microphone/recording path.
    # -------------------------------------------------

    true_delay_samples = 137

    delayed = apply_known_delay(
        reference,
        delay_samples=(
            true_delay_samples
        ),
    )

    gain = 0.72

    measurement_noise = rng.normal(
        0.0,
        0.003,
        size=num_samples,
    )

    measured_target = (
        gain
        * delayed
        + measurement_noise
    )

    # -------------------------------------------------
    # Delay estimation
    # -------------------------------------------------

    estimate = estimate_delay(
        reference,
        measured_target,
        max_delay_samples=500,
        minimum_overlap_samples=1_000,
        allow_polarity_inversion=True,
    )

    # -------------------------------------------------
    # Explicit alignment
    # -------------------------------------------------

    alignment = align_signals(
        reference,
        measured_target,
        delay=estimate,
        minimum_overlap_samples=1_000,
    )

    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------

    before_length = min(
        len(reference),
        len(measured_target),
    )

    before_correlation = (
        correlation(
            reference[
                :before_length
            ],
            measured_target[
                :before_length
            ],
        )
    )

    after_correlation = (
        correlation(
            alignment.reference,
            alignment.target,
        )
    )

    delay_error_samples = (
        estimate.lag_samples
        - true_delay_samples
    )

    delay_error_seconds = (
        delay_error_samples
        / sampling_rate_hz
    )

    estimated_delay_seconds = (
        estimate.lag_samples
        / sampling_rate_hz
    )

    true_delay_seconds = (
        true_delay_samples
        / sampling_rate_hz
    )

    # -------------------------------------------------
    # Acceptance
    # -------------------------------------------------

    if (
        estimate.lag_samples
        != true_delay_samples
    ):
        raise RuntimeError(
            "Calibration failed: estimated delay "
            "does not match the known experiment "
            "delay."
        )

    if (
        after_correlation
        <= before_correlation
    ):
        raise RuntimeError(
            "Calibration failed: alignment did not "
            "improve signal correlation."
        )

    # -------------------------------------------------
    # Save numerical artifacts
    # -------------------------------------------------

    np.save(
        results_dir
        / "reference.npy",
        reference,
    )

    np.save(
        results_dir
        / "measured_target.npy",
        measured_target,
    )

    np.save(
        results_dir
        / "aligned_reference.npy",
        alignment.reference,
    )

    np.save(
        results_dir
        / "aligned_target.npy",
        alignment.target,
    )

    # -------------------------------------------------
    # Plot raw timing mismatch
    # -------------------------------------------------

    display_samples = 1_500

    time = (
        np.arange(
            display_samples
        )
        / sampling_rate_hz
    )

    plt.figure(
        figsize=(11, 5)
    )

    plt.plot(
        time,
        reference[
            :display_samples
        ],
        label="Reference",
    )

    plt.plot(
        time,
        measured_target[
            :display_samples
        ],
        label="Measured target",
        alpha=0.8,
    )

    plt.title(
        "M6.02 — Before Calibration"
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Amplitude"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        results_dir
        / "before_alignment.png",
        dpi=150,
    )

    plt.close()

    # -------------------------------------------------
    # Plot aligned signals
    # -------------------------------------------------

    aligned_display = min(
        display_samples,
        alignment.overlap_samples,
    )

    aligned_time = (
        np.arange(
            aligned_display
        )
        / sampling_rate_hz
    )

    plt.figure(
        figsize=(11, 5)
    )

    plt.plot(
        aligned_time,
        alignment.reference[
            :aligned_display
        ],
        label="Aligned reference",
    )

    plt.plot(
        aligned_time,
        alignment.target[
            :aligned_display
        ],
        label="Aligned target",
        alpha=0.8,
    )

    plt.title(
        "M6.02 — After Explicit Alignment"
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Amplitude"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        results_dir
        / "after_alignment.png",
        dpi=150,
    )

    plt.close()

    # -------------------------------------------------
    # Save summary
    # -------------------------------------------------

    summary = {
        "module": "M6.02",
        "experiment": (
            "calibration_alignment"
        ),
        "seed": seed,
        "sampling_rate_hz": (
            sampling_rate_hz
        ),
        "num_samples": (
            num_samples
        ),
        "true_delay": {
            "samples": (
                true_delay_samples
            ),
            "seconds": (
                true_delay_seconds
            ),
        },
        "estimated_delay": {
            "samples": (
                estimate.lag_samples
            ),
            "seconds": (
                estimated_delay_seconds
            ),
        },
        "delay_error": {
            "samples": (
                delay_error_samples
            ),
            "seconds": (
                delay_error_seconds
            ),
        },
        "correlation": {
            "before_alignment": (
                before_correlation
            ),
            "after_alignment": (
                after_correlation
            ),
        },
        "normalized_peak_correlation": (
            estimate.normalized_peak_correlation
        ),
        "alignment": {
            "reference_start": (
                alignment.reference_start
            ),
            "target_start": (
                alignment.target_start
            ),
            "overlap_samples": (
                alignment.overlap_samples
            ),
        },
        "acceptance": {
            "delay_exactly_recovered": True,
            "alignment_improved_correlation": True,
        },
    }

    with (
        results_dir
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

        file.write(
            "\n"
        )

    # -------------------------------------------------
    # Console report
    # -------------------------------------------------

    print()

    print(
        "M6.02 — Calibration and "
        "Explicit Signal Alignment"
    )

    print(
        "----------------------------------------"
    )

    print()

    print(
        f"True delay: "
        f"{true_delay_samples} samples "
        f"({true_delay_seconds:.6f} s)"
    )

    print(
        f"Estimated delay: "
        f"{estimate.lag_samples} samples "
        f"({estimated_delay_seconds:.6f} s)"
    )

    print(
        f"Delay error: "
        f"{delay_error_samples} samples"
    )

    print()

    print(
        f"Correlation before alignment: "
        f"{before_correlation:.6f}"
    )

    print(
        f"Correlation after alignment: "
        f"{after_correlation:.6f}"
    )

    print()

    print(
        "M6.02 acceptance passed."
    )

    print()

    print(
        f"Results saved to:\n"
        f"{results_dir}"
    )


if __name__ == "__main__":
    main()