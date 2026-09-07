"""Raspberry Pi 3 audio capture and UDP streaming script.

Runs on the Pi with a ReSpeaker 2-Mic HAT.  Reads 2-channel PCM frames
via ALSA and sends each frame as one UDP packet to the laptop.

Usage::

    python capture_stream.py --host 192.168.1.100 --port 5005

Requirements (on Pi)::

    sudo apt-get install python3-pyaudio
    # or: pip install pyaudio

No ANC/ML code runs on the Pi — this keeps it within the Pi 3's
limited RAM/CPU budget entirely.
"""

from __future__ import annotations

import argparse
import socket
import struct
import sys
import time

import numpy as np


def capture_and_stream(
    host: str = "192.168.1.100",
    port: int = 5005,
    sample_rate: int = 16_000,
    channels: int = 2,
    frame_ms: int = 20,
    device_index: int | None = None,
) -> None:
    """Capture audio from the ReSpeaker HAT and stream via UDP.

    Parameters
    ----------
    host : str
        Laptop IP address to send packets to.
    port : int
        UDP port on the laptop.
    sample_rate : int
        Audio sample rate.
    channels : int
        Number of channels on the ReSpeaker HAT (typically 2).
    frame_ms : int
        Frame duration in milliseconds (20-40 ms recommended).
    device_index : int, optional
        ALSA device index for the ReSpeaker. If None, uses default.
    """
    try:
        import pyaudio
    except ImportError:
        print("ERROR: pyaudio not available. Install with:")
        print("  sudo apt-get install python3-pyaudio")
        print("  # or: pip install pyaudio")
        sys.exit(1)

    samples_per_frame = int(sample_rate * frame_ms / 1000)
    print(f"Capture config:")
    print(f"  Target:     {host}:{port}")
    print(f"  Rate:       {sample_rate} Hz")
    print(f"  Channels:   {channels}")
    print(f"  Frame:      {frame_ms} ms ({samples_per_frame} samples)")

    pa = pyaudio.PyAudio()

    # Find the ReSpeaker device if no index specified
    if device_index is None:
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            name = info.get("name", "").lower()
            if "respeaker" in name or "seeed" in name:
                device_index = i
                print(f"  Device:     [{i}] {info['name']}")
                break
        if device_index is None:
            print("  Device:     [default]")

    # Open audio stream
    stream_kwargs = {
        "format": pyaudio.paInt16,
        "channels": channels,
        "rate": sample_rate,
        "input": True,
        "frames_per_buffer": samples_per_frame,
    }
    if device_index is not None:
        stream_kwargs["input_device_index"] = device_index

    stream = pa.open(**stream_kwargs)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    seq = 0
    print(f"\nStreaming to {host}:{port}... (Ctrl+C to stop)")

    try:
        while True:
            # Read one frame of audio
            data = stream.read(samples_per_frame, exception_on_overflow=False)
            pcm = np.frombuffer(data, dtype=np.int16)

            # Build UDP packet: header + PCM payload
            header = struct.pack(">III", seq, channels, samples_per_frame)
            packet = header + pcm.tobytes()

            sock.sendto(packet, (host, port))
            seq += 1

            if seq % (1000 // frame_ms) == 0:  # ~1 per second
                sys.stdout.write(f"\r  Sent {seq} frames ({seq * frame_ms / 1000:.0f}s)")
                sys.stdout.flush()

    except KeyboardInterrupt:
        print(f"\n\nStopped after {seq} frames.")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        sock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ReSpeaker 2-Mic HAT capture → UDP streaming"
    )
    parser.add_argument(
        "--host", default="192.168.1.100",
        help="Laptop IP address (default: 192.168.1.100)",
    )
    parser.add_argument(
        "--port", type=int, default=5005,
        help="UDP port (default: 5005)",
    )
    parser.add_argument(
        "--rate", type=int, default=16000,
        help="Sample rate (default: 16000)",
    )
    parser.add_argument(
        "--channels", type=int, default=2,
        help="Number of channels (default: 2)",
    )
    parser.add_argument(
        "--frame-ms", type=int, default=20,
        help="Frame duration in ms (default: 20)",
    )
    parser.add_argument(
        "--device", type=int, default=None,
        help="ALSA device index (default: auto-detect ReSpeaker)",
    )
    args = parser.parse_args()

    capture_and_stream(
        host=args.host,
        port=args.port,
        sample_rate=args.rate,
        channels=args.channels,
        frame_ms=args.frame_ms,
        device_index=args.device,
    )
