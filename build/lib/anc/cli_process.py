#!/usr/bin/env python3
"""PS26052 — Production Noise Cancellation Processor.

A self-contained audio noise cancellation tool that uses a pretrained
DTLN (Dual-signal Transformation LSTM Network) to suppress noise from
speech audio in real time.

Modes
-----
file     Process a .wav file and write the enhanced result to disk.
mic      Capture from default microphone, enhance, play back
         through the default speakers -- continuous, real-time.
stream   Read raw PCM from stdin, enhance, write to stdout.
         Suitable for piping with ffmpeg, sox, or any audio tool.

Examples
--------
    # Enhance a WAV file
    python process_audio.py file -i noisy.wav -o clean.wav

    # Real-time mic-to-speaker processing (press Ctrl+C to stop)
    python process_audio.py mic

    # Pipe from ffmpeg (any format to enhanced WAV)
    ffmpeg -i input.mp3 -f s16le -ar 16000 -ac 1 - | \\
        python process_audio.py stream | \\
        ffmpeg -f s16le -ar 16000 -ac 1 -i - enhanced.wav
"""

from __future__ import annotations

import argparse
import signal
import struct
import sys
import time
from pathlib import Path

import numpy as np

# Ensure src/ is on the import path
_src = Path(__file__).resolve().parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from ai.models.pretrained_dtln import DTLNModel, is_available as dtln_available

_SAMPLE_RATE = 16_000


# ======================================================================
# Core engine — thin wrapper, no unnecessary abstractions
# ======================================================================

class NoiseProcessor:
    """Single-responsibility audio noise suppressor.

    Wraps the pretrained DTLN ONNX model for inference. No training,
    no evaluation, no experiment framework — just signal in, clean
    signal out.
    """

    def __init__(self, sample_rate: int = _SAMPLE_RATE) -> None:
        if not dtln_available():
            raise RuntimeError(
                "onnxruntime is required. Install with:  pip install onnxruntime"
            )
        self._model = DTLNModel(target_sample_rate=sample_rate)
        self._sample_rate = sample_rate

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def model_name(self) -> str:
        return self._model.name

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Enhance a mono float64 audio array.

        Parameters
        ----------
        audio : np.ndarray
            1-D float64 mono audio at ``self.sample_rate``.

        Returns
        -------
        np.ndarray
            Enhanced audio, same length as input.
        """
        return self._model.enhance(audio)


# ======================================================================
# Mode: File processing
# ======================================================================

def _cmd_file(args: argparse.Namespace) -> None:
    """Process a .wav file end-to-end."""
    from scipy.io import wavfile

    input_path = args.input
    if not Path(input_path).exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    sr_file, raw = wavfile.read(input_path)

    # Normalize to float64 mono
    if raw.dtype == np.int16:
        audio = raw.astype(np.float64) / 32768.0
    elif raw.dtype == np.float32:
        audio = raw.astype(np.float64)
    else:
        audio = raw.astype(np.float64)

    if audio.ndim > 1:
        audio = audio[:, 0]

    # Resample if needed
    target_sr = args.sample_rate
    if sr_file != target_sr:
        from anc.speech.resampling import resample_audio
        audio = resample_audio(audio, sr_file, target_sr)

    processor = NoiseProcessor(sample_rate=target_sr)

    print(f"Input:       {input_path}")
    print(f"Rate:        {sr_file} Hz -> {target_sr} Hz")
    print(f"Duration:    {len(audio)/target_sr:.2f}s ({len(audio)} samples)")
    print(f"Model:       {processor.model_name}")

    t0 = time.perf_counter()
    enhanced = processor.process(audio)
    elapsed = time.perf_counter() - t0

    audio_dur = len(audio) / target_sr
    ratio = elapsed / audio_dur if audio_dur > 0 else 0

    # Output path
    output_path = args.output
    if output_path is None:
        stem = Path(input_path).stem
        output_path = str(Path(input_path).with_name(f"{stem}_enhanced.wav"))

    # Write 16-bit PCM WAV
    pcm = (enhanced * 32767).clip(-32768, 32767).astype(np.int16)
    wavfile.write(output_path, target_sr, pcm)

    print(f"\nOutput:      {output_path}")
    print(f"Processed:   {elapsed*1000:.0f} ms ({ratio:.2f}x real-time)")


# ======================================================================
# Mode: Real-time mic → speaker
# ======================================================================

def _cmd_mic(args: argparse.Namespace) -> None:
    """Real-time mic capture, enhancement, and speaker playback."""
    try:
        import sounddevice as sd
    except ImportError:
        print(
            "Error: sounddevice is required for mic mode.\n"
            "  pip install sounddevice",
            file=sys.stderr,
        )
        sys.exit(1)

    sr = args.sample_rate
    processor = NoiseProcessor(sample_rate=sr)
    block_size = 512

    running = True

    def _on_sigint(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _on_sigint)

    print(f"PS26052 Noise Cancellation — Real-time Mic Mode")
    print(f"  Model:       {processor.model_name}")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Block size:  {block_size} samples ({block_size/sr*1000:.0f} ms)")
    print(f"\nListening on default microphone... Press Ctrl+C to stop.\n")

    frame_count = 0
    total_samples = 0
    start_time = time.time()

    def _callback(indata, outdata, frames, time_info, status):
        nonlocal frame_count, total_samples
        if status:
            print(f"  [audio: {status}]", file=sys.stderr)

        mono = indata[:, 0].astype(np.float64)
        enhanced = processor.process(mono)

        out = enhanced[:frames]
        if len(out) < frames:
            out = np.pad(out, (0, frames - len(out)))
        outdata[:, 0] = out.astype(np.float32)

        frame_count += 1
        total_samples += frames

    with sd.Stream(
        samplerate=sr,
        blocksize=block_size,
        channels=1,
        dtype="float32",
        callback=_callback,
    ):
        while running:
            time.sleep(0.1)
            elapsed = time.time() - start_time
            if frame_count > 0 and frame_count % 10 == 0:
                sys.stdout.write(
                    f"\r  Processed: {total_samples/sr:.1f}s "
                    f"({elapsed:.0f}s elapsed, "
                    f"{frame_count} frames)"
                )
                sys.stdout.flush()

    elapsed = time.time() - start_time
    print(f"\n\nDone. Processed {total_samples/sr:.1f}s of audio in {elapsed:.1f}s.")


# ======================================================================
# Mode: stdin/stdout streaming (for piping with ffmpeg, sox, etc.)
# ======================================================================

def _cmd_stream(args: argparse.Namespace) -> None:
    """Read raw PCM from stdin, enhance, write to stdout.

    Format: signed 16-bit little-endian, mono, at --sample-rate Hz.
    """
    sr = args.sample_rate
    processor = NoiseProcessor(sample_rate=sr)

    # Process in chunks of 0.5 seconds
    chunk_samples = sr // 2
    chunk_bytes = chunk_samples * 2  # 16-bit = 2 bytes per sample

    if sys.platform == "win32":
        import msvcrt, os
        msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)

    total_samples = 0
    stderr = sys.stderr

    while True:
        raw = sys.stdin.buffer.read(chunk_bytes)
        if not raw:
            break

        # Decode PCM
        n_samples = len(raw) // 2
        pcm = np.frombuffer(raw[:n_samples * 2], dtype=np.int16)
        audio = pcm.astype(np.float64) / 32768.0

        # Enhance
        enhanced = processor.process(audio)

        # Encode back to PCM
        out_pcm = (enhanced * 32767).clip(-32768, 32767).astype(np.int16)
        sys.stdout.buffer.write(out_pcm.tobytes())
        sys.stdout.buffer.flush()

        total_samples += n_samples

    print(
        f"Processed {total_samples/sr:.1f}s of audio",
        file=stderr,
    )


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="process_audio",
        description="PS26052 - AI Noise Cancellation for Defence Communications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python process_audio.py file -i noisy.wav -o clean.wav\n"
            "  python process_audio.py mic\n"
            "  python process_audio.py mic --sample-rate 48000\n"
        ),
    )
    parser.add_argument(
        "--sample-rate", type=int, default=16000,
        help="Audio sample rate in Hz (default: 16000)",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # file mode
    p_file = sub.add_parser("file", help="Process a .wav file")
    p_file.add_argument("-i", "--input", required=True, help="Input .wav file")
    p_file.add_argument("-o", "--output", default=None, help="Output .wav file")
    p_file.set_defaults(func=_cmd_file)

    # mic mode
    p_mic = sub.add_parser("mic", help="Real-time mic-to-speaker")
    p_mic.set_defaults(func=_cmd_mic)

    # stream mode
    p_stream = sub.add_parser(
        "stream",
        help="Pipe raw PCM via stdin/stdout (s16le, mono)",
    )
    p_stream.set_defaults(func=_cmd_stream)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
