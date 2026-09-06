"""Preparation of recorded signals for offline ANC replay."""

from __future__ import annotations

from anc.calibration import (
    align_signals,
)

from anc.io.recordings import (
    ANCReplayInputs,
    SignalRecording,
)


def prepare_replay_inputs(
    reference: SignalRecording,
    measured: SignalRecording,
    *,
    align: bool = True,
    max_delay_samples: int | None = None,
    minimum_overlap_samples: int = 32,
) -> ANCReplayInputs:
    """Prepare recorded signals for an ANC replay experiment.

    The function:

    1. verifies equal sampling rates,
    2. optionally estimates timing offset,
    3. explicitly aligns the recordings,
    4. returns equal-length standardized replay inputs.

    It does not modify the ANC controller.
    """

    if (
        reference.sampling_rate_hz
        != measured.sampling_rate_hz
    ):
        raise ValueError(
            "Sampling-rate mismatch between "
            "reference and measured recordings."
        )

    if not align:

        overlap = min(
            reference.num_samples,
            measured.num_samples,
        )

        return ANCReplayInputs(
            reference=SignalRecording(
                samples=reference.samples[
                    :overlap
                ],
                sampling_rate_hz=(
                    reference.sampling_rate_hz
                ),
                role="reference",
                source=(
                    reference.source
                ),
                selected_channel=(
                    reference.selected_channel
                ),
                total_channels=(
                    reference.total_channels
                ),
            ),
            measured=SignalRecording(
                samples=measured.samples[
                    :overlap
                ],
                sampling_rate_hz=(
                    measured.sampling_rate_hz
                ),
                role="measured",
                source=(
                    measured.source
                ),
                selected_channel=(
                    measured.selected_channel
                ),
                total_channels=(
                    measured.total_channels
                ),
            ),
            alignment_applied=False,
            delay_samples=None,
            overlap_samples=overlap,
        )

    result = align_signals(
        reference.samples,
        measured.samples,
        max_delay_samples=(
            max_delay_samples
        ),
        minimum_overlap_samples=(
            minimum_overlap_samples
        ),
        allow_polarity_inversion=True,
    )

    aligned_reference = SignalRecording(
        samples=result.reference,
        sampling_rate_hz=(
            reference.sampling_rate_hz
        ),
        role="reference",
        source=reference.source,
        selected_channel=(
            reference.selected_channel
        ),
        total_channels=(
            reference.total_channels
        ),
    )

    aligned_measured = SignalRecording(
        samples=result.target,
        sampling_rate_hz=(
            measured.sampling_rate_hz
        ),
        role="measured",
        source=measured.source,
        selected_channel=(
            measured.selected_channel
        ),
        total_channels=(
            measured.total_channels
        ),
    )

    return ANCReplayInputs(
        reference=aligned_reference,
        measured=aligned_measured,
        alignment_applied=True,
        delay_samples=(
            result.delay.lag_samples
        ),
        overlap_samples=(
            result.overlap_samples
        ),
    )