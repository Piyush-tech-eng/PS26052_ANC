#!/usr/bin/env python3
"""PS26052 ANC — Live Demo Orchestration Entrypoint.

Wires together all streaming/hardware components into a single runnable
demo script:

    UDPReceiver (from Pi)
        → HybridEngine (classical ANC + AI enhancement)
        → AudioPlayback (sounddevice or file fallback)

With LatencyMonitor and LiveMetricsOverlay attached for real-time display.

Modes
-----
- ``live``:     Listen for UDP packets from the Raspberry Pi, process and
                play back in real time.
- ``file``:     Process a local .wav file through the full pipeline and
                write the enhanced output to disk.
- ``loopback``: Capture from the default microphone, enhance, and play
                back — useful for demos without hardware.

Usage::

    # Live mode (Pi must be streaming)
    python run_live_demo.py --mode live --port 5005

    # Process a file
    python run_live_demo.py --mode file \\
        --input results/demo_assets/wav/demo_0_rotor_snr+0_noisy.wav \\
        --output enhanced_output.wav

    # Loopback (mic → enhance → speaker)
    python run_live_demo.py --mode loopback
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ai.models import get_best_available_model
from ai.streaming.hybrid_engine import HybridEngine
from ai.hardware.latency_monitor import LatencyMonitor


def _process_file(
    input_path: str,
    output_path: str,
    model_name: str = "auto",
    no_anc: bool = True,
    sample_rate: int = 16_000,
) -> None:
    """Process a .wav file through the hybrid pipeline."""
    from scipy.io import wavfile

    sr_file, raw = wavfile.read(input_path)
    if raw.dtype == np.int16:
        audio = raw.astype(np.float64) / 32768.0
    elif raw.dtype == np.float32:
        audio = raw.astype(np.float64)
    else:
        audio = raw.astype(np.float64)

    # If stereo, take first channel
    if audio.ndim > 1:
        audio = audio[:, 0]

    # Resample if needed
    if sr_file != sample_rate:
        from anc.speech.resampling import resample_audio
        audio = resample_audio(audio, sr_file, sample_rate)

    print(f"Input:  {input_path}")
    print(f"  Rate: {sr_file} Hz -> {sample_rate} Hz")
    print(f"  Duration: {len(audio)/sample_rate:.2f}s ({len(audio)} samples)")

    # Build engine
    model = get_best_available_model(prefer=model_name, sample_rate=sample_rate)
    engine = HybridEngine(
        model=model,
        sample_rate=sample_rate,
    )

    # Process
    t0 = time.perf_counter()
    enhanced, timing = engine.process_batch(audio)
    t1 = time.perf_counter()

    print(f"\nProcessing complete:")
    print(f"  Model: {model.name}")
    print(f"  Processing time: {(t1-t0)*1000:.0f} ms")
    print(f"  Real-time ratio: {timing.realtime_ratio:.2f}x")

    # Write output
    pcm = (enhanced * 32767).clip(-32768, 32767).astype(np.int16)
    wavfile.write(output_path, sample_rate, pcm)
    print(f"  Output: {output_path}")

    # Quick metrics if we have a clean reference nearby
    _try_compute_metrics(audio, enhanced, sample_rate)


def _try_compute_metrics(noisy, enhanced, sr):
    """If possible, compute and display basic metrics."""
    from anc.evaluation.metrics import compute_si_snr, compute_stoi_standard

    # We don't have a clean reference in file mode, but we can show
    # relative change metrics
    rms_before = float(np.sqrt(np.mean(noisy**2)))
    rms_after = float(np.sqrt(np.mean(enhanced**2)))
    print(f"\n  RMS before: {rms_before:.4f}")
    print(f"  RMS after:  {rms_after:.4f}")
    print(f"  Reduction:  {20*np.log10(rms_after/(rms_before+1e-10)):+.1f} dB")


def _run_live(
    port: int = 5005,
    model_name: str = "auto",
    no_anc: bool = True,
    sample_rate: int = 16_000,
) -> None:
    """Run the live UDP → enhance → playback pipeline."""
    from ai.hardware.udp_receiver import UDPReceiver
    from ai.hardware.playback import AudioPlayback

    model = get_best_available_model(prefer=model_name, sample_rate=sample_rate)
    engine = HybridEngine(model=model, sample_rate=sample_rate)
    monitor = LatencyMonitor()

    receiver = UDPReceiver(port=port, sample_rate=sample_rate)
    playback = AudioPlayback(sample_rate=sample_rate)

    # Graceful shutdown
    running = True

    def _signal_handler(sig, frame):
        nonlocal running
        running = False
        print("\n\nShutting down...")

    signal.signal(signal.SIGINT, _signal_handler)

    print(f"\n{'='*60}")
    print(f"PS26052 ANC — LIVE DEMO")
    print(f"{'='*60}")
    print(f"  Model:       {model.name}")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  UDP port:    {port}")
    print(f"  ANC:         {'OFF (AI-only)' if no_anc else 'ON'}")
    print(f"\nWaiting for audio from Raspberry Pi...")
    print(f"Press Ctrl+C to stop.\n")

    receiver.start()
    playback.start()

    frame_count = 0
    try:
        while running:
            frame = receiver.get_frame(timeout=0.1)
            if frame is None:
                continue

            monitor.begin()

            # Get measured audio (channel 0)
            measured = frame["channels"][0] if "channels" in frame else frame.get("audio", np.zeros(320))
            measured = np.asarray(measured, dtype=np.float64).ravel()

            monitor.mark("receive")

            # Process through hybrid engine
            enhanced, timing = engine.process_frame(measured)
            monitor.mark("enhance")

            # Send to playback
            playback.write(enhanced)
            monitor.mark("playback")

            snapshot = monitor.end()
            frame_count += 1

            # Print status every ~1 second
            if frame_count % (sample_rate // max(len(measured), 1)) == 0:
                avg = monitor.average_ms
                sys.stdout.write(
                    f"\r  Frames: {frame_count} | "
                    f"Latency: {snapshot.total_ms:.1f}ms "
                    f"(recv={avg.get('receive',0):.1f} "
                    f"enh={avg.get('enhance',0):.1f} "
                    f"play={avg.get('playback',0):.1f}) | "
                    f"RT ratio: {timing.realtime_ratio:.2f}x"
                )
                sys.stdout.flush()
    finally:
        receiver.stop()
        playback.stop()
        print(f"\n\nProcessed {frame_count} frames.")
        print(monitor.summary())


def _run_loopback(
    model_name: str = "auto",
    sample_rate: int = 16_000,
    duration: float = 0.0,
) -> None:
    """Capture from mic, enhance, play back — no hardware needed."""
    model = get_best_available_model(prefer=model_name, sample_rate=sample_rate)
    engine = HybridEngine(model=model, sample_rate=sample_rate)
    monitor = LatencyMonitor()

    try:
        import sounddevice as sd
    except ImportError:
        print("ERROR: sounddevice is required for loopback mode.")
        print("  pip install sounddevice")
        sys.exit(1)

    block_size = 512
    running = True

    def _signal_handler(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _signal_handler)

    print(f"\n{'='*60}")
    print(f"PS26052 ANC — LOOPBACK DEMO")
    print(f"{'='*60}")
    print(f"  Model:       {model.name}")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Block size:  {block_size}")
    if duration > 0:
        print(f"  Duration:    {duration}s")
    print(f"\nListening on default mic... Press Ctrl+C to stop.\n")

    frame_count = 0
    start_time = time.time()

    def _callback(indata, outdata, frames, time_info, status):
        nonlocal frame_count
        if status:
            print(f"  [audio status: {status}]")

        mono = indata[:, 0].astype(np.float64)
        enhanced, timing = engine.process_frame(mono)

        # Write to output
        out = enhanced[:frames]
        if len(out) < frames:
            out = np.pad(out, (0, frames - len(out)))
        outdata[:, 0] = out.astype(np.float32)

        frame_count += 1

    with sd.Stream(
        samplerate=sample_rate,
        blocksize=block_size,
        channels=1,
        dtype="float32",
        callback=_callback,
    ):
        while running:
            time.sleep(0.1)
            elapsed = time.time() - start_time

            if frame_count % 10 == 0 and frame_count > 0:
                latest = engine.latest_timing
                if latest:
                    sys.stdout.write(
                        f"\r  Frames: {frame_count} | "
                        f"Elapsed: {elapsed:.0f}s | "
                        f"Latency: {latest.total_ms:.1f}ms | "
                        f"RT ratio: {latest.realtime_ratio:.2f}x"
                    )
                    sys.stdout.flush()

            if 0 < duration <= elapsed:
                break

    print(f"\n\nProcessed {frame_count} frames in {time.time()-start_time:.1f}s.")


def main():
    parser = argparse.ArgumentParser(
        description="PS26052 ANC — Live Demo Orchestration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["live", "file", "loopback"], default="file",
        help="Demo mode (default: file)",
    )
    parser.add_argument(
        "--model", default="auto",
        choices=["auto", "dtln", "rnnoise", "spectral_gate"],
        help="Enhancement model (default: auto = best available)",
    )
    parser.add_argument(
        "--port", type=int, default=5005,
        help="UDP port for live mode (default: 5005)",
    )
    parser.add_argument(
        "--input", "-i", default=None,
        help="Input .wav file for file mode",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output .wav file for file mode (default: <input>_enhanced.wav)",
    )
    parser.add_argument(
        "--no-anc", action="store_true",
        help="Skip classical ANC stage (AI-only mode)",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=16000,
        help="Sample rate (default: 16000)",
    )
    parser.add_argument(
        "--duration", type=float, default=0.0,
        help="Loopback duration in seconds (0 = indefinite)",
    )

    args = parser.parse_args()

    if args.mode == "file":
        if not args.input:
            parser.error("--input is required for file mode")
        output = args.output or str(
            Path(args.input).with_stem(Path(args.input).stem + "_enhanced")
        )
        _process_file(
            args.input, output,
            model_name=args.model,
            no_anc=args.no_anc,
            sample_rate=args.sample_rate,
        )
    elif args.mode == "live":
        _run_live(
            port=args.port,
            model_name=args.model,
            no_anc=args.no_anc,
            sample_rate=args.sample_rate,
        )
    elif args.mode == "loopback":
        _run_loopback(
            model_name=args.model,
            sample_rate=args.sample_rate,
            duration=args.duration,
        )


if __name__ == "__main__":
    main()
