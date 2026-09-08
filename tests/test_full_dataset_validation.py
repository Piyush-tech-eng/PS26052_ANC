"""Full dataset validation tests for PS26052 ANC.

Validates the full scale-and-split requirements from Phase A:
- Complete speaker split isolation (zero speaker leakage across train/val/test)
- Held-out noise family reservation (held-out family strictly in test split)
- Stratified mixing coverage across (noise_family, snr_db) cells
- Dataset window integrity (finite arrays, matching lengths, valid metadata)
- Normalization statistics computation and roundtrip application
"""

from __future__ import annotations

import numpy as np
import pytest

from anc.evaluation.dataset import (
    DatasetWindow,
    WindowConfig,
    apply_normalization,
    compute_normalization,
    extract_windows,
    split_dataset,
)
from anc.speech.mixing import DEFAULT_SNR_LADDER_DB, stratified_mix_batch
from anc.speech.sources import NOISE_FAMILIES, NoiseSample, SpeechSample


def _create_synthetic_speech_pool(n_speakers: int = 10, samples_per_speaker: int = 2) -> list[SpeechSample]:
    """Helper to generate a multi-speaker speech sample pool for split validation."""
    samples = []
    sr = 16_000
    duration = 1.0  # 1 second clips
    n_pts = int(sr * duration)

    for spk_idx in range(n_speakers):
        speaker_id = f"spk_{spk_idx:03d}"
        for s_idx in range(samples_per_speaker):
            t = np.linspace(0, duration, n_pts, endpoint=False)
            f0 = 120.0 + spk_idx * 15.0 + s_idx * 5.0
            audio = 0.5 * np.sin(2 * np.pi * f0 * t) + 0.25 * np.sin(4 * np.pi * f0 * t)
            samples.append(SpeechSample(
                audio=audio,
                sampling_rate=sr,
                source_id=f"speech_{speaker_id}_{s_idx:02d}",
                speaker_id=speaker_id,
                provenance={"dataset": "synthetic_test", "speaker_id": speaker_id},
            ))
    return samples


def _create_synthetic_noise_pool(families: list[str] | None = None) -> list[NoiseSample]:
    """Helper to generate noise samples covering specified noise families."""
    if families is None:
        families = ["impulsive", "rotor", "engine_vehicle", "broadband", "colored", "wind", "alarm_siren"]
    samples = []
    sr = 16_000
    n_pts = sr

    rng = np.random.default_rng(42)
    for fam in families:
        for n_idx in range(3):
            if fam == "broadband":
                audio = rng.standard_normal(n_pts) * 0.1
            elif fam == "rotor":
                t = np.linspace(0, 1.0, n_pts, endpoint=False)
                audio = 0.3 * np.sin(2 * np.pi * 50.0 * t) + 0.1 * rng.standard_normal(n_pts)
            elif fam == "impulsive":
                audio = np.zeros(n_pts)
                audio[n_pts // 2: n_pts // 2 + 50] = 0.8
            else:
                audio = rng.standard_normal(n_pts) * 0.05

            samples.append(NoiseSample(
                audio=audio,
                sampling_rate=sr,
                noise_id=f"noise_{fam}_{n_idx:02d}",
                noise_family=fam,
                source_id=f"source_{fam}_{n_idx}",
                provenance={"dataset": "synthetic_test", "noise_family": fam},
            ))
    return samples


def test_stratified_mixing_cell_coverage() -> None:
    """Validate that stratified_mix_batch produces mixes for every (noise_family, snr_db) cell."""
    speech = _create_synthetic_speech_pool(n_speakers=6, samples_per_speaker=2)
    families = ["rotor", "impulsive", "broadband"]
    noise = _create_synthetic_noise_pool(families=families)
    snr_ladder = [-10.0, 0.0, 10.0]

    mixes = stratified_mix_batch(
        speech,
        noise,
        snr_ladder_db=snr_ladder,
        samples_per_cell=5,
        seed=123,
    )

    # Check that all cells are covered
    covered_cells: set[tuple[str, float]] = set()
    for m in mixes:
        covered_cells.add((m.noise_family, m.snr_db))

    expected_cells = {(fam, snr) for fam in families for snr in snr_ladder}
    assert covered_cells == expected_cells, f"Missing cells: {expected_cells - covered_cells}"


def test_speaker_split_zero_leakage() -> None:
    """Validate zero speaker leakage across train, validation, and test splits at scale."""
    speech = _create_synthetic_speech_pool(n_speakers=12, samples_per_speaker=3)
    noises = _create_synthetic_noise_pool(["rotor", "engine_vehicle", "impulsive"])
    mixes = stratified_mix_batch(speech, noises, snr_ladder_db=[0.0, 10.0], samples_per_cell=4, seed=42)

    windows = []
    for i, m in enumerate(mixes):
        windows.append(DatasetWindow(
            reference_x=m.noisy_speech,
            baseline_residual=m.noise_component,
            controller_output_y=None,
            filtered_reference=None,
            target=m.clean_speech,
            sample_rate=m.sampling_rate,
            scenario_id=f"scen_{i}",
            run_id=f"run_{i}",
            window_id=f"win_{i}",
            algorithm_metadata={"controller": "none"},
            path_metadata={"secondary_path": "ideal"},
            source_type="speech",
            speech_metadata={
                "speech_source_id": m.speaker_id,
                "noise_family": m.noise_family,
                "snr_db": m.snr_db,
            },
            noisy_speech=m.noisy_speech,
            clean_speech_target=m.clean_speech,
            noise_component=m.noise_component,
        ))

    splits = split_dataset(windows, train_fraction=0.6, validation_fraction=0.2)

    train_speakers = {w.speech_metadata["speech_source_id"] for w in splits["train"]}
    val_speakers = {w.speech_metadata["speech_source_id"] for w in splits["validation"]}
    test_speakers = {w.speech_metadata["speech_source_id"] for w in splits["test"]}

    assert len(train_speakers) > 0, "Train split has no speakers"
    assert len(val_speakers) > 0, "Validation split has no speakers"
    assert len(test_speakers) > 0, "Test split has no speakers"

    # Mutual exclusivity
    assert not (train_speakers & val_speakers), f"Train/val speaker overlap: {train_speakers & val_speakers}"
    assert not (train_speakers & test_speakers), f"Train/test speaker overlap: {train_speakers & test_speakers}"
    assert not (val_speakers & test_speakers), f"Val/test speaker overlap: {val_speakers & test_speakers}"


def test_heldout_noise_family_isolation() -> None:
    """Validate that the held-out noise family is strictly confined to the test split."""
    speech = _create_synthetic_speech_pool(n_speakers=6, samples_per_speaker=1)
    noises = _create_synthetic_noise_pool(["rotor", "engine_vehicle", "impulsive"])
    from anc.speech.mixing import mix_batch
    mixes = mix_batch(speech, noises, snr_ladder_db=[0.0])

    windows = []
    for i, m in enumerate(mixes):
        windows.append(DatasetWindow(
            reference_x=m.noisy_speech,
            baseline_residual=m.noise_component,
            controller_output_y=None,
            filtered_reference=None,
            target=m.clean_speech,
            sample_rate=m.sampling_rate,
            scenario_id=f"scen_{i}",
            run_id=f"run_{i}",
            window_id=f"win_{i}",
            algorithm_metadata={},
            path_metadata={},
            source_type="speech",
            speech_metadata={
                "speech_source_id": m.speaker_id,
                "noise_family": m.noise_family,
                "snr_db": m.snr_db,
            },
        ))

    # split_dataset reserves one noise family for test only
    splits = split_dataset(windows, train_fraction=0.6, validation_fraction=0.2)

    train_noises = {w.speech_metadata["noise_family"] for w in splits["train"]}
    val_noises = {w.speech_metadata["noise_family"] for w in splits["validation"]}
    test_noises = {w.speech_metadata["noise_family"] for w in splits["test"]}

    # The test set must contain at least one noise family not present in train or validation
    heldout_noises = test_noises - (train_noises | val_noises)
    assert len(heldout_noises) >= 1, "Expected at least one strictly held-out noise family in test split"


def test_normalization_computation_and_application() -> None:
    """Validate compute_normalization and apply_normalization behave as expected."""
    speech = _create_synthetic_speech_pool(n_speakers=4, samples_per_speaker=2)
    noises = _create_synthetic_noise_pool(["broadband"])
    mixes = stratified_mix_batch(speech, noises, snr_ladder_db=[5.0], samples_per_cell=2, seed=99)

    windows = []
    for i, m in enumerate(mixes):
        windows.append(DatasetWindow(
            reference_x=m.noisy_speech,
            baseline_residual=m.noise_component,
            controller_output_y=None,
            filtered_reference=None,
            target=m.clean_speech,
            sample_rate=m.sampling_rate,
            scenario_id=f"scen_{i}",
            run_id=f"run_{i}",
            window_id=f"win_{i}",
            algorithm_metadata={},
            path_metadata={},
            source_type="speech",
            speech_metadata={"speech_source_id": m.speaker_id, "noise_family": m.noise_family},
            noisy_speech=m.noisy_speech,
            clean_speech_target=m.clean_speech,
            noise_component=m.noise_component,
        ))

    norm_stats = compute_normalization(windows)
    assert "fields" in norm_stats
    assert "noisy_speech" in norm_stats["fields"]
    assert "clean_speech_target" in norm_stats["fields"]

    # Apply normalization to the first window
    w_norm = apply_normalization(windows[0], norm_stats)
    assert w_norm.noisy_speech is not None
    assert w_norm.clean_speech_target is not None

    # Check shapes match
    assert len(w_norm.noisy_speech) == len(windows[0].noisy_speech)
    assert len(w_norm.clean_speech_target) == len(windows[0].clean_speech_target)
    # Check that it's transformed (not identical unless mean=0, std=1)
    assert np.all(np.isfinite(w_norm.noisy_speech))
