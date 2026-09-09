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

from anc.speech.augmentation import convolve_with_rir, generate_synthetic_rirs, load_rir_corpus
from anc.speech.corpus_manifest import CorpusEntry, CorpusManifest, write_manifest
from anc.speech.mixing import DEFAULT_SNR_LADDER_DB, mix_at_snr
from anc.speech.sources import (
    NoiseSample,
    SpeechSample,
    load_esc50_noise,
    load_librispeech,
    load_musan_noise,
    load_vctk,
)
from anc.signals.generators import generate_sine, generate_white_noise

RESULTS_DIR = Path("results/m7_speech_ai_handoff")
CORPORA_DIR = Path("data/corpora")


def load_real_or_synthetic_data() -> tuple[list[SpeechSample], list[NoiseSample], CorpusManifest]:
    """Load real corpora if present under data/corpora/, else fall back to synthetic data."""
    manifest = CorpusManifest(description="PS26052 ANC speech and noise corpus manifest")
    speech_samples: list[SpeechSample] = []
    noise_samples: list[NoiseSample] = []

    # Try real speech corpora
    librispeech_dir = CORPORA_DIR / "librispeech"
    vctk_dir = CORPORA_DIR / "vctk"

    if librispeech_dir.exists():
        loaded = load_librispeech(librispeech_dir, max_samples=50)
        speech_samples.extend(loaded)
        for s in loaded:
            manifest.add_entry(CorpusEntry(
                file_path=str(s.provenance.get("original_path", s.source_id)),
                source_url="https://www.openslr.org/12/",
                license="CC-BY-4.0",
                collector="m7_02_pipeline",
                dataset_name="librispeech",
                speaker_id=s.speaker_id,
                duration_seconds=s.duration,
            ))

    if vctk_dir.exists():
        loaded = load_vctk(vctk_dir, max_samples=50)
        speech_samples.extend(loaded)
        for s in loaded:
            manifest.add_entry(CorpusEntry(
                file_path=str(s.provenance.get("original_path", s.source_id)),
                source_url="https://datashare.ed.ac.uk/handle/10283/3443",
                license="CC-BY-4.0",
                collector="m7_02_pipeline",
                dataset_name="vctk",
                speaker_id=s.speaker_id,
                duration_seconds=s.duration,
            ))

    # Try real noise corpora
    musan_dir = CORPORA_DIR / "musan"
    esc50_dir = CORPORA_DIR / "esc50"

    if musan_dir.exists():
        loaded = load_musan_noise(musan_dir, max_per_family=10)
        noise_samples.extend(loaded)
        for n in loaded:
            manifest.add_entry(CorpusEntry(
                file_path=str(n.provenance.get("original_path", n.source_id)),
                source_url="https://www.openslr.org/17/",
                license="Open (see MUSAN)",
                collector="m7_02_pipeline",
                dataset_name="musan",
                noise_family=n.noise_family,
                duration_seconds=n.duration,
            ))

    if esc50_dir.exists():
        loaded = load_esc50_noise(esc50_dir, max_per_family=10)
        noise_samples.extend(loaded)
        for n in loaded:
            manifest.add_entry(CorpusEntry(
                file_path=str(n.provenance.get("original_path", n.source_id)),
                source_url="https://github.com/karolpiczak/ESC-50",
                license="CC-BY-NC-3.0",
                collector="m7_02_pipeline",
                dataset_name="esc50",
                noise_family=n.noise_family,
                duration_seconds=n.duration,
            ))

    # If real data is not available, generate synthetic fixtures
    if not speech_samples or not noise_samples:
        print("Real corpora not found in data/corpora/; using synthetic data generators.")
        synth_speech, synth_noise = generate_synthetic_data()
        speech_samples = synth_speech
        noise_samples = synth_noise
        for s in synth_speech:
            manifest.add_entry(CorpusEntry(
                file_path=s.source_id,
                source_url="internal://synthetic",
                license="MIT",
                collector="synthetic_generator",
                dataset_name="synthetic_speech",
                speaker_id=s.speaker_id,
                duration_seconds=s.duration,
            ))
        for n in synth_noise:
            manifest.add_entry(CorpusEntry(
                file_path=n.source_id,
                source_url="internal://synthetic",
                license="MIT",
                collector="synthetic_generator",
                dataset_name="synthetic_noise",
                noise_family=n.noise_family,
                duration_seconds=n.duration,
            ))

    return speech_samples, noise_samples, manifest


def generate_synthetic_data() -> tuple[list[SpeechSample], list[NoiseSample]]:
    """Generate synthetic speech and noise samples if real data is missing."""
    speech_samples = []
    noise_samples = []
    sr = 8000
    duration = 5.0
    
    # Generate 5 speakers (synthetic representation)
    for i in range(5):
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
    
    print("Loading speech and noise samples...")
    speech_samples, noise_samples, corpus_manifest = load_real_or_synthetic_data()
    write_manifest(corpus_manifest, RESULTS_DIR / "corpus_manifest.json")
    print(f"Corpus manifest written to {RESULTS_DIR / 'corpus_manifest.json'}")

    # Prepare RIRs for reverberation augmentation (Phase A.4)
    rir_dir = CORPORA_DIR / "rir"
    rirs = load_rir_corpus(rir_dir) if rir_dir.exists() else []
    if not rirs:
        rirs = generate_synthetic_rirs(n_rirs=5, sampling_rate=speech_samples[0].sampling_rate, seed=42)
    
    
    snr_levels = [0.0, 10.0]
    unique_speech = speech_samples[:4]
    unique_speech_ids = [s.source_id for s in unique_speech]
    unique_families = sorted(list(set(n.noise_family for n in noise_samples)))

    definitions = build_scenario_matrix(
        signal_classes=["speech"],
        input_levels=["nominal"],
        path_conditions=["ideal"],
        source_types=["speech"],
        controllers=["nlms", "none"],
        secondary_path_models=["ideal"],
        speech_source_ids=unique_speech_ids,
        noise_families=unique_families,
        snr_dbs=snr_levels,
        config_snapshot={"algorithm": "nlms", "learning_rate": 0.01},
        seed=42
    )
    
    print(f"Generated {len(definitions)} scenario definitions.")
    
    speech_lookup = {s.source_id: s for s in speech_samples}
    noise_lookup: dict[str, list[NoiseSample]] = {}
    for n in noise_samples:
        noise_lookup.setdefault(n.noise_family, []).append(n)

    results = []
    
    for idx, df in enumerate(definitions):
        # Find matching source
        speech = speech_lookup[df.speech_source_id]
        matching_noises = noise_lookup[df.noise_family]
        noise = matching_noises[idx % len(matching_noises)]
        
        # Trim to 3.0s for uniform windowing and fast scenario simulation
        target_len = int(speech.sampling_rate * 3.0)
        s_audio = speech.audio[:target_len] if len(speech.audio) >= target_len else np.pad(speech.audio, (0, target_len - len(speech.audio)))
        n_audio = noise.audio[:target_len] if len(noise.audio) >= target_len else np.pad(noise.audio, (0, target_len - len(noise.audio)))
        s_trimmed = SpeechSample(audio=s_audio, sampling_rate=speech.sampling_rate, source_id=speech.source_id, speaker_id=speech.speaker_id, provenance=speech.provenance)
        n_trimmed = NoiseSample(audio=n_audio, sampling_rate=noise.sampling_rate, noise_id=noise.noise_id, noise_family=noise.noise_family, source_id=noise.source_id, provenance=noise.provenance)

        mix_result = mix_at_snr(s_trimmed, n_trimmed, df.snr_db)
        mixed_audio = mix_result.noisy_speech
        clean_target = mix_result.clean_speech

        # Phase A.4: Apply RIR reverberation augmentation to ~30% of scenarios
        is_augmented = (idx % 3 == 0) and len(rirs) > 0
        aug_type = None
        if is_augmented:
            rir = rirs[idx % len(rirs)]
            mixed_audio = convolve_with_rir(mixed_audio, rir)
            clean_target = convolve_with_rir(clean_target, rir)
            aug_type = "rir_convolution"
        
        # We need a measured scenario to pass to run_scenario
        from anc.scenarios import recording_from_array
        from anc.secondary_path import SecondaryPathModel
        
        ref_rec = recording_from_array(mixed_audio, sampling_rate_hz=speech.sampling_rate, source_type="synthetic")
        err_rec = recording_from_array(mixed_audio, sampling_rate_hz=speech.sampling_rate, source_type="synthetic")
        ideal_path = np.zeros(64)
        ideal_path[0] = 1.0
        model = SecondaryPathModel(
            impulse_response=ideal_path,
            sampling_rate_hz=speech.sampling_rate,
            metadata={"source_id": "ideal", "generation_method": "ideal"},
        )
        
        scenario = ANCScenario(
            scenario_id=df.scenario_id,
            reference=ref_rec,
            error_input=err_rec,
            secondary_path_model=model,
            true_secondary_path=ideal_path,
        )
        
        # Target is the clean speech
        target = clean_target
        
        cfg = ANCExperimentConfig(
            sampling_rate_hz=speech.sampling_rate,
            algorithm=df.controller,
            step_size=0.01,
            filter_length=64,
            learning_window=speech.sampling_rate,
        )
        res = run_scenario(df, scenario, cfg, target=target)
        results.append((res, mixed_audio, target, mix_result.noise_component, is_augmented, aug_type))
        if (idx + 1) % 10 == 0 or (idx + 1) == len(definitions):
            print(f"  Processed {idx + 1}/{len(definitions)} scenarios...", flush=True)
        
    print("Extracting windows...")
    window_config = WindowConfig(window_length=4000, hop_length=2000)
    
    windows = extract_windows([r[0] for r in results], window_config)
    run_arrays = {r[0].run_id: (r[1], r[2], r[3], r[4], r[5]) for r in results}
    
    enriched_windows = []
    for w in windows:
        run_id = w.run_id
        mixed_audio, clean_audio, noise_comp, is_aug, aug_tp = run_arrays[run_id]
        
        idx_str = w.window_id.split("__w")[-1]
        idx = int(idx_str)
        start = idx * window_config.hop_length
        end = start + window_config.window_length
        
        meta = dict(w.speech_metadata) if w.speech_metadata is not None else {}
        meta["augmented"] = is_aug
        if aug_tp is not None:
            meta["augmentation_type"] = aug_tp
        
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
            speech_metadata=meta,
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
            n_aug = sum(1 for w in v if w.speech_metadata.get("augmented"))
            print(f"    Speakers: {speakers}")
            print(f"    Noises: {noises}")
            print(f"    Augmented: {n_aug}/{len(v)} windows")

    norm_stats = compute_normalization(splits["train"] if splits["train"] else splits["test"])
    
    print("Saving dataset...")
    save_dataset(RESULTS_DIR, splits, norm_stats, window_config)
    
    # Save a summary manifest
    summary = {
        "num_windows": {k: len(v) for k, v in splits.items()},
        "augmented_windows": {k: sum(1 for w in v if w.speech_metadata.get("augmented")) for k, v in splits.items()},
        "speakers_by_split": {k: list({w.speech_metadata['speech_source_id'] for w in v}) for k, v in splits.items()},
        "noise_families_by_split": {k: list({w.speech_metadata['noise_family'] for w in v}) for k, v in splits.items()},
    }
    with open(RESULTS_DIR / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"Done. Dataset saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
