"""Dataset generation combining speech and noise, building scenarios, and saving splits.

Generates the `results/m7_speech_ai_handoff/` dataset for Phase 5 AI model integration.
This script demonstrates the end-to-end pipeline of mixing speech and noise, generating
baseline ANC residuals, splitting the dataset into train/val/test, and persisting to disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from anc.evaluation.dataset import (
    WindowConfig, extract_windows, split_dataset, compute_normalization, save_dataset
)
from anc.evaluation.scenarios import build_scenario_matrix, run_scenario
from anc.plant import ANCExperimentConfig
from anc.scenarios import ANCScenario

from anc.speech.mixing import mix_at_snr
from anc.speech.sources import SpeechSample, NoiseSample
from anc.signals.generators import generate_sine, generate_white_noise

RESULTS_DIR = Path("results/m7_speech_ai_handoff")


def generate_synthetic_data() -> tuple[list[SpeechSample], list[NoiseSample]]:
    """Generate synthetic speech and noise samples if real data is missing."""
    speech_samples = []
    noise_samples = []
    sr = 8000
    duration = 5.0
    
    # Generate 5 speakers (synthetic representation)
    for i in range(5):
        # A mixture of tones to represent a voice
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        f0 = 100 + i * 50
        audio = np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(2 * np.pi * f0 * 2 * t)
        
        speech_samples.append(SpeechSample(
            audio=audio,
            sampling_rate=sr,
            source_id=f"synth_speech_{i:02d}",
            speaker_id=f"spk_{i}",
            provenance={"synthetic_approximation": True}
        ))
        
    # Generate 3 noise families (rotor, impulsive, broadband)
    for family in ["rotor", "impulsive", "broadband"]:
        if family == "rotor":
            noise_audio = generate_sine(sr, duration, frequency=50.0).samples
        elif family == "impulsive":
            noise_audio = np.zeros(int(sr * duration))
            noise_audio[sr:sr+100] = 1.0  # impulse at 1s
            noise_audio[3*sr:3*sr+100] = 0.8
        else:
            noise_audio = generate_white_noise(sr, duration).samples
            
        noise_samples.append(NoiseSample(
            audio=noise_audio,
            sampling_rate=sr,
            noise_id=f"noise_{family}_01",
            noise_family=family,
            source_id=f"synth_{family}_01",
            provenance={"synthetic_approximation": True}
        ))
        
    return speech_samples, noise_samples


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Generating speech/noise dataset...")
    speech_samples, noise_samples = generate_synthetic_data()
    
    config = ANCExperimentConfig(
        algorithm="nlms",
        step_size=0.01,
        filter_length=64,
        learning_window=8000
    )
    
    snr_levels = [0.0, 10.0]
    definitions = build_scenario_matrix(
        signal_classes=["speech"],
        input_levels=["nominal"],
        path_conditions=["ideal"],
        source_types=["speech"],
        controllers=["nlms", "none"],
        secondary_path_models=["ideal"],
        speech_source_ids=[s.speaker_id for s in speech_samples],
        noise_families=[n.noise_family for n in noise_samples],
        snr_dbs=snr_levels,
        config_snapshot={"algorithm": "nlms", "learning_rate": 0.01},
        seed=42
    )
    
    print(f"Generated {len(definitions)} scenario definitions.")
    
    results = []
    # Note: For a real pipeline, we'd mix them into a physical path. 
    # For synthetic demonstration, we'll directly construct an ANCScenario.
    
    for df in definitions:
        # Find matching source
        speech = next(s for s in speech_samples if s.speaker_id == df.speech_source_id)
        noise = next(n for n in noise_samples if n.noise_family == df.noise_family)
        
        mix_result = mix_at_snr(speech, noise, df.snr_db)
        mixed_audio = mix_result.noisy_speech
        
        # We need a measured scenario to pass to run_scenario
        from anc.scenarios import recording_from_array
        from anc.secondary_path import SecondaryPathModel
        
        ref_rec = recording_from_array(mixed_audio, sampling_rate_hz=speech.sampling_rate, role="reference", source="synthetic")
        err_rec = recording_from_array(mixed_audio, sampling_rate_hz=speech.sampling_rate, role="measured", source="synthetic")
        ideal_path = np.zeros(64)
        ideal_path[0] = 1.0
        model = SecondaryPathModel(impulse_response=ideal_path, sampling_rate_hz=speech.sampling_rate, source_id="ideal", generation_method="ideal")
        
        scenario = ANCScenario(
            scenario_id=df.scenario_id,
            reference=ref_rec,
            error_input=err_rec,
            secondary_path_model=model,
            true_secondary_path=ideal_path,
        )
        
        # Target is the clean speech
        target = mix_result.clean_speech
        
        res = run_scenario(df, scenario, config, target=target)
        # Manually attach speech metadata arrays to the result definition for extract_windows?
        # DatasetWindow has noisy_speech, clean_speech_target. We'll add them after extract_windows.
        results.append((res, mixed_audio, target, mix_result.noise_component))
        
    print("Extracting windows...")
    window_config = WindowConfig(window_length=4000, hop_length=2000)
    
    # We run extract_windows on the pure results
    windows = extract_windows([r[0] for r in results], window_config)
    
    # We need to map the noisy/clean arrays into the DatasetWindows
    # We can do this by run_id
    run_arrays = {r[0].run_id: (r[1], r[2], r[3]) for r in results}
    
    enriched_windows = []
    for w in windows:
        run_id = w.run_id
        mixed_audio, clean_audio, noise_comp = run_arrays[run_id]
        
        # Get start/end indices. extract_windows uses the index in window_id (e.g. `__w00001`)
        idx_str = w.window_id.split("__w")[-1]
        idx = int(idx_str)
        start = idx * window_config.hop_length
        end = start + window_config.window_length
        
        # In Python dataclasses, if it's frozen we must use object.__setattr__
        # But for script simplicity, we create a new instance with the fields
        
        enriched_windows.append(type(w)(
            reference_x=w.reference_x,
            baseline_residual=w.baseline_residual,
            controller_output_y=w.controller_output_y,
            filtered_reference=w.filtered_reference,
            target=w.target,
            sample_rate=w.sample_rate,
            scenario_id=w.scenario_id,
            run_id=w.run_id,
            window_id=w.window_id,
            algorithm_metadata=w.algorithm_metadata,
            path_metadata=w.path_metadata,
            source_type=w.source_type,
            speech_metadata=w.speech_metadata,
            noisy_speech=mixed_audio[start:end],
            clean_speech_target=clean_audio[start:end],
            noise_component=noise_comp[start:end]
        ))
        
    print(f"Total windows extracted: {len(enriched_windows)}")
    
    splits = split_dataset(enriched_windows, train_fraction=0.6, validation_fraction=0.2)
    
    print("Splits created:")
    for k, v in splits.items():
        print(f"  {k}: {len(v)} windows")
        if len(v) > 0:
            speakers = {w.speech_metadata['speech_source_id'] for w in v}
            noises = {w.speech_metadata['noise_family'] for w in v}
            print(f"    Speakers: {speakers}")
            print(f"    Noises: {noises}")

    norm_stats = compute_normalization(splits["train"] if splits["train"] else splits["test"])
    
    print("Saving dataset...")
    save_dataset(splits, norm_stats, RESULTS_DIR)
    
    # Save a manifest
    manifest = {
        "num_windows": {k: len(v) for k, v in splits.items()},
        "speakers_by_split": {k: list({w.speech_metadata['speech_source_id'] for w in v}) for k, v in splits.items()},
        "noise_families_by_split": {k: list({w.speech_metadata['noise_family'] for w in v}) for k, v in splits.items()},
    }
    with open(RESULTS_DIR / "dataset_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Done. Dataset saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
