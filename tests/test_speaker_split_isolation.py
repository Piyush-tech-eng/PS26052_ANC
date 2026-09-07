"""Tests for speaker isolation in the dataset split logic."""

import numpy as np
import pytest

from anc.evaluation.dataset import DatasetWindow, split_dataset


def test_split_isolates_speakers() -> None:
    # Create windows with different speakers and noise families
    # We need at least 3 distinct speakers to split train/val/test
    # and we want to verify no speaker appears in more than one split
    
    windows = []
    
    # 5 speakers, 3 noise families
    for i in range(20):
        speaker_id = f"spk_{i % 5}"
        noise_family = f"noise_{i % 3}"
        
        windows.append(DatasetWindow(
            reference_x=np.zeros(10),
            baseline_residual=np.zeros(10),
            controller_output_y=None,
            filtered_reference=None,
            target=None,
            sample_rate=8000,
            scenario_id=f"scen_{i}",
            run_id=f"run_{i}",
            window_id=f"win_{i}",
            algorithm_metadata={"controller": "none"},
            path_metadata={"secondary_path_model": "ideal", "path_condition": "ideal"},
            source_type="speech",
            speech_metadata={"speech_source_id": speaker_id, "noise_family": noise_family, "snr_db": 5.0}
        ))
        
    splits = split_dataset(windows, train_fraction=0.6, validation_fraction=0.2)
    
    assert "train" in splits
    assert "validation" in splits
    assert "test" in splits
    
    train_speakers = {w.speech_metadata["speech_source_id"] for w in splits["train"]}
    val_speakers = {w.speech_metadata["speech_source_id"] for w in splits["validation"]}
    test_speakers = {w.speech_metadata["speech_source_id"] for w in splits["test"]}
    
    # Ensure mutually exclusive speakers
    assert not train_speakers.intersection(val_speakers), "Train and validation share speakers!"
    assert not train_speakers.intersection(test_speakers), "Train and test share speakers!"
    assert not val_speakers.intersection(test_speakers), "Validation and test share speakers!"
    
    # Check that at least 2 speakers are in test
    assert len(test_speakers) >= 2, f"Test split should have >= 2 speakers, but has {len(test_speakers)}"
    
    train_noise = {w.speech_metadata["noise_family"] for w in splits["train"]}
    val_noise = {w.speech_metadata["noise_family"] for w in splits["validation"]}
    test_noise = {w.speech_metadata["noise_family"] for w in splits["test"]}
    
    # Check that at least 1 noise family is exclusively in test
    # (Since there are only 3 noise families, and we reserve 1 for test)
    # The current logic forces the reserved test noise families to only appear in the test set
    reserved_in_test = test_noise - (train_noise.union(val_noise))
    assert len(reserved_in_test) >= 1, "At least one noise family should be strictly in the test set."
