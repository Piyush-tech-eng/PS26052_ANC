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
import json
import signal
import socket
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ai.models import get_best_available_model
from ai.streaming.hybrid_engine import HybridEngine
from ai.hardware.latency_monitor import LatencyMonitor


def _write_status(status_file: str | Path | None, data: dict[str, Any]) -> None:
    """Safely write telemetry dict to status JSON file."""
    if not status_file:
        return
    try:
        p = Path(status_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        tmp.replace(p)
    except Exception:
        pass


def _process_file(
    input_path: str,
    output_path: str,
    model_name: str = "auto",
    no_anc: bool = True,
    sample_rate: int = 16_000,
    status_file: str | None = None,
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

    # Write status file if requested
    _write_status(status_file, {
        "mode": "file",
        "model": model.name,
        "input": input_path,
        "output": output_path,
        "sample_rate": sample_rate,
        "duration_s": len(audio) / sample_rate,
        "process_time_ms": (t1 - t0) * 1000,
        "realtime_ratio": timing.realtime_ratio,
        "input_rms": float(np.sqrt(np.mean(audio**2))),
        "output_rms": float(np.sqrt(np.mean(enhanced**2))),
        "status": "completed",
        "timestamp": time.time(),
    })

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
    print(f"PS26052 ANC -- LIVE DEMO")
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
    print(f"PS26052 ANC -- LOOPBACK DEMO")
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


def _run_simulated(
    model_name: str = "auto",
    sample_rate: int = 16_000,
    duration: float = 5.0,
    port: int = 15005,
    no_anc: bool = False,
    status_file: str | None = None,
) -> None:
    """Run an end-to-end simulated hardware live streaming demo.

    Simulates the Raspberry Pi 3 capture node streaming dual-channel audio
    (speech + nonstationary rotor/engine noise) over UDP to localhost, while
    the laptop processes it through the complete UDPReceiver -> Jitter Buffer ->
    HybridEngine (FxNLMS + DTLN) -> AudioPlayback cascade.
    """
    import threading
    from ai.hardware.playback import AudioPlayback
    from ai.hardware.udp_receiver import UDPReceiver
    from ai.streaming.frame_anc import FrameANCConfig

    print(f"\n{'='*65}")
    print("PS26052 ANC -- SIMULATED HARDWARE LIVE STREAMING DEMO")
    print(f"{'='*65}")
    print(f"  Sample Rate:     {sample_rate} Hz")
    print(f"  Target Duration: {duration:.1f} s")
    print(f"  Transport:       Simulated UDP Loopback (port {port})")

    model = get_best_available_model(prefer=model_name, sample_rate=sample_rate)
    print(f"  Model:           {model.name}")

    anc_cfg = None
    sec_true = None
    sec_model = None
    if not no_anc:
        anc_cfg = FrameANCConfig(filter_length=64, step_size=0.01)
        sec_true = np.zeros(64)
        sec_true[0] = 1.0
        sec_model = sec_true.copy()
        print("  Classical ANC:   Enabled (Streaming FxNLMS, length=64)")
    else:
        print("  Classical ANC:   Disabled (AI-only enhancement)")

    engine = HybridEngine(
        model=model,
        anc_config=anc_cfg,
        secondary_path_true=sec_true,
        secondary_path_model=sec_model,
        sample_rate=sample_rate,
    )
    monitor = LatencyMonitor()
    receiver = UDPReceiver(port=port, sample_rate=sample_rate, jitter_buffer_depth=1)
    playback = AudioPlayback(sample_rate=sample_rate)

    receiver.start()
    playback.start()

    frame_samples = 320  # 20ms at 16kHz
    t_frame = np.arange(frame_samples) / sample_rate
    sender_running = True

    def _simulated_pi_sender():
        seq = 0
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        t_global = 0.0
        dt = frame_samples / sample_rate
        rng = np.random.default_rng(42)
        while sender_running:
            t = t_global + t_frame
            # Speech: harmonics at 150, 300, 450 Hz
            speech = 0.35 * np.sin(2 * np.pi * 150.0 * t) + 0.18 * np.sin(2 * np.pi * 300.0 * t)
            # Noise: 50 Hz rotor harmonics + broadband turbulence
            noise = 0.45 * np.sin(2 * np.pi * 50.0 * t) + 0.25 * np.sin(2 * np.pi * 100.0 * t) + 0.08 * rng.standard_normal(frame_samples)
            # Channel 0: Error/primary mic (speech + noise)
            # Channel 1: Reference mic (noise only)
            audio = np.column_stack([speech + noise, noise])
            pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
            header = struct.pack(">III", seq, 2, frame_samples)
            packet = header + pcm.tobytes()
            try:
                sock.sendto(packet, ("127.0.0.1", port))
            except Exception:
                pass
            seq += 1
            t_global += dt
            time.sleep(dt * 0.95)
        sock.close()

    sender_thread = threading.Thread(target=_simulated_pi_sender, daemon=True)
    sender_thread.start()

    start_time = time.time()
    frame_count = 0
    print("\nStreaming live audio frames through hybrid pipeline...")

    try:
        while time.time() - start_time < duration:
            frame = receiver.get_frame()
            if frame is None:
                time.sleep(0.005)
                continue

            monitor.begin()
            audio_data = frame["audio"]
            if audio_data.ndim == 2 and audio_data.shape[1] >= 2:
                meas = audio_data[:, 0]
                ref = audio_data[:, 1]
            else:
                meas = audio_data.ravel()
                ref = None

            monitor.mark("receive")

            if engine.has_anc and ref is not None:
                enhanced, timing = engine.process_frame(measured=meas, reference=ref)
            else:
                enhanced, timing = engine.process_frame(measured=meas)

            monitor.mark("enhance")
            playback.write(enhanced)
            monitor.mark("playback")

            snapshot = monitor.end()
            frame_count += 1

            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                avg = monitor.average_ms
                sys.stdout.write(
                    f"\r  Frames: {frame_count:3d} | "
                    f"Elapsed: {elapsed:4.1f}s | "
                    f"Latency: {snapshot.total_ms:5.1f}ms "
                    f"(recv={avg.get('receive', 0):.1f}ms, enh={avg.get('enhance', 0):.1f}ms, play={avg.get('playback', 0):.1f}ms) | "
                    f"RT ratio: {timing.realtime_ratio:.2f}x"
                )
                sys.stdout.flush()

                _write_status(status_file, {
                    "mode": "simulated",
                    "model": model.name,
                    "frame_count": frame_count,
                    "elapsed_seconds": elapsed,
                    "total_latency_ms": snapshot.total_ms,
                    "stage_latencies_ms": avg,
                    "realtime_ratio": timing.realtime_ratio,
                    "input_rms": float(np.sqrt(np.mean(meas**2))),
                    "output_rms": float(np.sqrt(np.mean(enhanced**2))),
                    "sample_rate": sample_rate,
                    "status": "running",
                    "timestamp": time.time(),
                })
    finally:
        sender_running = False
        sender_thread.join(timeout=1.0)
        receiver.stop()
        playback.stop()

        _write_status(status_file, {
            "mode": "simulated",
            "model": model.name,
            "frame_count": frame_count,
            "elapsed_seconds": time.time() - start_time,
            "realtime_ratio": timing.realtime_ratio,
            "sample_rate": sample_rate,
            "status": "completed",
            "timestamp": time.time(),
        })

        print(f"\n\nSimulation completed successfully: {frame_count} frames processed in {time.time()-start_time:.1f}s.")
        print(monitor.summary())


def main():
    parser = argparse.ArgumentParser(
        description="PS26052 ANC -- Live Demo Orchestration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["live", "file", "loopback", "simulated"], default="simulated",
        help="Demo mode (default: simulated)",
    )
    parser.add_argument(
        "--model", default="auto",
        choices=["auto", "dtln", "rnnoise", "spectral_gate"],
        help="Enhancement model (default: auto = best available)",
    )
    parser.add_argument(
        "--port", type=int, default=5005,
        help="UDP port for live/simulated mode (default: 5005)",
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
        "--duration", type=float, default=5.0,
        help="Duration in seconds for simulated or loopback mode (default: 5.0)",
    )
    parser.add_argument(
        "--status-file", type=str, default=None,
        help="Path to write JSON status file for dashboard polling",
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
            status_file=args.status_file,
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
    elif args.mode == "simulated":
        _run_simulated(
            model_name=args.model,
            sample_rate=args.sample_rate,
            duration=args.duration if args.duration > 0 else 5.0,
            port=args.port if args.port != 5005 else 15005,
            no_anc=args.no_anc,
            status_file=args.status_file,
        )


if __name__ == "__main__":
    main()
