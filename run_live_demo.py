#!/usr/bin/env python3
"""PS26052 ANC — Unified Live Demonstration Entrypoint.

Demonstrates the two-microphone adaptive noise cancellation system across 4
operational modes with real-time telemetry streaming to the operator dashboard.

Demonstration Modes:
1. offline (file):
   Process WAV files through the streaming pipeline frame-by-frame,
   recording telemetry and saving enhanced audio.
2. laptop-live (loopback):
   Continuous capture from laptop microphone -> Hybrid Engine -> speakers/headphones.
3. pi-prototype (live):
   Receive dual-channel UDP stream from Raspberry Pi ReSpeaker 2-Mic HAT,
   execute two-stage classical FxNLMS + AI enhancement on laptop, output audio.
4. pi-edge (simulated):
   Edge demonstration running on-device FxNLMS + quantized INT8 AI.

Usage::

    # 1. Offline file processing
    python run_live_demo.py --mode offline --input sample.wav --output enhanced.wav

    # 2. Laptop live microphone demonstration
    python run_live_demo.py --mode laptop-live --duration 10

    # 3. Raspberry Pi 3 hardware demonstration
    python run_live_demo.py --mode pi-prototype --port 5005

    # 4. On-Pi Edge demonstration
    python run_live_demo.py --mode pi-edge --duration 10
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Ensure repo root and src are on path
repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from ai.models import get_best_available_model
from ai.streaming.hybrid_engine import HybridEngine
from anc.realtime.audio_input import AudioInput, AudioFrame, WAVInput, LaptopMicInput, RaspberryPiInput
from anc.realtime.audio_output import AudioOutput, WAVOutput, LaptopSpeakerOutput
from anc.realtime.streaming_pipeline import StreamingPipeline
from anc.realtime.telemetry import TelemetryCollector, PipelineTelemetry


def _find_default_input_wav() -> Path | None:
    """Find a suitable default demo WAV file in the workspace."""
    candidates = [
        repo_root / "results" / "demo_assets" / "wav" / "demo_0_rotor_snr+0_noisy.wav",
        repo_root / "results" / "demo_assets" / "wav" / "demo_1_alarm_snr+5_noisy.wav",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def run_demo(
    mode: str = "offline",
    input_file: str | None = None,
    output_file: str | None = None,
    host: str = "0.0.0.0",
    port: int = 5005,
    model_name: str = "auto",
    no_anc: bool = False,
    sample_rate: int = 16_000,
    duration: float = 0.0,
    status_file: str | None = "results/demo_status.json",
    ch0_is_reference: bool = False,
    calibrate: bool = False,
) -> None:
    """Execute the ANC demonstration in the specified mode."""
    # Normalize mode aliases
    mode_map = {
        "file": "offline",
        "loopback": "laptop-live",
        "live": "pi-prototype",
        "simulated": "pi-edge",
    }
    canon_mode = mode_map.get(mode.lower(), mode.lower())

    ref_channel = 0 if ch0_is_reference else 1
    err_channel = 1 if ch0_is_reference else 0

    print(f"\n{'='*75}")
    print(f"PS26052 ANC - LIVE DEMONSTRATION ORCHESTRATOR")
    print(f"{'='*75}")
    print(f"  Mode:             {canon_mode.upper()} (requested: {mode})")
    print(f"  Sample Rate:      {sample_rate} Hz (20 ms / 320 samples per frame)")
    print(f"  Channel Mapping:  Ref = Ch{ref_channel}, Err = Ch{err_channel}")
    print(f"  Classical ANC:    {'DISABLED (AI-only)' if no_anc else 'ENABLED (FxNLMS 64-tap)'}")
    print(f"  AI Model:         {model_name.upper()}")
    if status_file:
        print(f"  Dashboard Status: {status_file}")
    print(f"{'='*75}\n")

    # Select AI Enhancement model
    model = get_best_available_model(prefer=model_name, sample_rate=sample_rate)
    print(f"[Engine] Selected AI Model: {model.name}")

    # Build Hybrid Engine
    from ai.streaming.frame_anc import FrameANCConfig
    anc_cfg = None
    sec_true = None
    sec_model = None
    if not no_anc:
        anc_cfg = FrameANCConfig(filter_length=64, step_size=0.01)
        sec_true = np.zeros(64, dtype=np.float64)
        sec_true[0] = 1.0
        sec_model = sec_true.copy()

    engine = HybridEngine(
        model=model,
        anc_config=anc_cfg,
        secondary_path_true=sec_true,
        secondary_path_model=sec_model,
        sample_rate=sample_rate,
    )

    # Build Telemetry Collector
    telemetry = TelemetryCollector(
        mode=canon_mode,
        sample_rate=sample_rate,
        frame_size=int(sample_rate * 0.02),
        status_file=Path(status_file) if status_file else None,
        write_interval=0.25,
    )
    model_path_str = str(getattr(model, "_model_path", ""))
    telemetry.set_model_precision("int8" if "quantized" in model_path_str else "fp32")

    # Construct AudioInput and AudioOutput based on mode
    audio_in: AudioInput
    audio_out: AudioOutput

    if canon_mode == "offline":
        in_path = input_file or _find_default_input_wav()
        if not in_path:
            raise FileNotFoundError("No input WAV specified and no default demo WAV found in results/demo_assets/wav/")
        out_path = output_file or str(Path(in_path).with_stem(Path(in_path).stem + "_enhanced"))
        print(f"[Audio] Input WAV:  {in_path}")
        print(f"[Audio] Output WAV: {out_path}")

        audio_in = WAVInput(
            path=in_path,
            target_sample_rate=sample_rate,
            reference_channel=ref_channel,
            error_channel=err_channel,
            realtime=False,
        )
        audio_out = WAVOutput(path=out_path, sample_rate=sample_rate)

    elif canon_mode == "laptop-live":
        print("[Audio] Initializing Laptop Microphone Input...")
        audio_in = LaptopMicInput(sample_rate=sample_rate, frame_ms=20)
        audio_out = LaptopSpeakerOutput(sample_rate=sample_rate)

    elif canon_mode == "pi-prototype":
        print(f"[Audio] Listening for Pi UDP packets on {host}:{port}...")
        if calibrate:
            print("[Calibration] Running 2-second ambient channel calibration...")
            from ai.hardware.udp_receiver import UDPReceiver
            temp_rx = UDPReceiver(host=host, port=port, sample_rate=sample_rate, reference_channel=ref_channel, error_channel=err_channel)
            temp_rx.start()
            cal = temp_rx.calibrate_channels(duration_seconds=2.0)
            temp_rx.stop()
            print(f"[Calibration] RMS Ratio: {cal.get('rms_ratio', 1.0):.2f}, Recommended gain: {cal.get('recommended_gain', 1.0):.2f}")

        audio_in = RaspberryPiInput(
            host=host,
            port=port,
            sample_rate=sample_rate,
            reference_channel=ref_channel,
            error_channel=err_channel,
        )
        audio_out = LaptopSpeakerOutput(sample_rate=sample_rate)

    elif canon_mode == "pi-edge":
        print(f"[Audio] Edge demonstration mode (Pi-3 Simulation)...")
        # Generate or use WAV in real-time simulation pace
        in_path = input_file or _find_default_input_wav()
        if in_path and Path(in_path).exists():
            audio_in = WAVInput(path=in_path, target_sample_rate=sample_rate, realtime=True)
        else:
            # Synthetic 300 Hz drone rotor + speech
            t = np.arange(int(sample_rate * (duration if duration > 0 else 5.0))) / sample_rate
            synth = 0.4 * np.sin(2 * np.pi * 300 * t) + 0.1 * np.random.randn(len(t))
            synth_pcm = (synth * 16384).astype(np.int16)
            tmp_wav = Path("results/scratch_edge.wav")
            tmp_wav.parent.mkdir(parents=True, exist_ok=True)
            from scipy.io import wavfile
            wavfile.write(str(tmp_wav), sample_rate, synth_pcm)
            audio_in = WAVInput(path=tmp_wav, target_sample_rate=sample_rate, realtime=True)

        out_path = output_file or "results/demo_output.wav"
        audio_out = WAVOutput(path=out_path, sample_rate=sample_rate)

    else:
        raise ValueError(f"Unknown demonstration mode: {mode}")

    # Build streaming pipeline
    pipeline = StreamingPipeline(
        input_source=audio_in,
        output_sink=audio_out,
        engine=engine,
        telemetry=telemetry,
        buffer_size=16,
    )

    # Signal handling for clean Ctrl+C shutdown
    running = True

    def _sig_handler(sig, frame):
        nonlocal running
        running = False
        print("\n[Demo] Interrupt received, terminating streaming pipeline...")
        pipeline.stop()

    signal.signal(signal.SIGINT, _sig_handler)

    print("\n[Pipeline] Streaming started. Press Ctrl+C to terminate.")

    # Status printer callback
    last_print = 0.0

    def _on_frame_callback(count: int, snap: PipelineTelemetry) -> None:
        nonlocal last_print
        now = time.time()
        if now - last_print >= 0.5:
            last_print = now
            sys.stdout.write(
                f"\r  Frames: {snap.frames_processed:4d} | "
                f"Uptime: {snap.elapsed_seconds:4.1f}s | "
                f"Latency: {snap.total_latency_ms:4.1f}ms | "
                f"RT Ratio: {snap.realtime_ratio:4.2f}x | "
                f"Attenuation: -{abs(snap.estimated_attenuation_db):4.1f}dB"
            )
            sys.stdout.flush()

    pipeline._on_frame = _on_frame_callback

    # Run
    t_start = time.time()
    if duration > 0 and canon_mode != "offline":
        pipeline.start()
        while running and (time.time() - t_start < duration):
            time.sleep(0.05)
        pipeline.stop()
    else:
        pipeline.run_blocking()

    elapsed = time.time() - t_start
    final_snap = pipeline.get_telemetry()

    print(f"\n\n{'='*75}")
    print(f"DEMONSTRATION RUN SUMMARY")
    print(f"{'='*75}")
    print(f"  Frames Processed: {final_snap.frames_processed}")
    print(f"  Elapsed Time:     {elapsed:.2f} s")
    print(f"  Total Latency:    {final_snap.total_latency_ms:.2f} ms")
    print(f"  Real-Time Ratio:  {final_snap.realtime_ratio:.2f}x (< 1.0x required)")
    print(f"  Est. Attenuation: -{abs(final_snap.estimated_attenuation_db):.1f} dB")
    if canon_mode == "offline" or isinstance(audio_out, WAVOutput):
        print(f"  Enhanced Output:  {getattr(audio_out, '_path', 'N/A')}")
    print(f"{'='*75}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PS26052 ANC Live Demonstration Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["offline", "file", "laptop-live", "loopback", "pi-prototype", "live", "pi-edge", "simulated"],
        default="offline",
        help="Demo mode (default: offline)",
    )
    parser.add_argument("--input", "-i", default=None, help="Input WAV file for offline mode")
    parser.add_argument("--output", "-o", default=None, help="Output WAV file for enhanced audio")
    parser.add_argument("--host", default="0.0.0.0", help="UDP bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5005, help="UDP port (default: 5005)")
    parser.add_argument(
        "--model", default="auto",
        choices=["auto", "dtln", "dtln_quantized", "spectral_gate"],
        help="AI enhancement model (default: auto)",
    )
    parser.add_argument("--no-anc", action="store_true", help="Disable classical ANC stage (AI-only)")
    parser.add_argument("--sample-rate", type=int, default=16_000, help="Sample rate (default: 16000)")
    parser.add_argument("--duration", type=float, default=0.0, help="Duration in seconds (0 = continuous / full file)")
    parser.add_argument("--status-file", default="results/demo_status.json", help="Path to write JSON status file")
    parser.add_argument("--ch0-is-reference", action="store_true", help="Set Channel 0 as Reference and Channel 1 as Error")
    parser.add_argument("--calibrate", action="store_true", help="Run 2s channel gain calibration before streaming")

    args = parser.parse_args()

    run_demo(
        mode=args.mode,
        input_file=args.input,
        output_file=args.output,
        host=args.host,
        port=args.port,
        model_name=args.model,
        no_anc=args.no_anc,
        sample_rate=args.sample_rate,
        duration=args.duration,
        status_file=args.status_file,
        ch0_is_reference=args.ch0_is_reference,
        calibrate=args.calibrate,
    )


if __name__ == "__main__":
    main()
