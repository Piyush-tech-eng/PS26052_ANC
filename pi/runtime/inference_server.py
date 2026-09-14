#!/usr/bin/env python3
"""PS26052 ANC — Raspberry Pi 5 File-Based Inference Server.

Minimal HTTP server for offline/batch DTLN speech enhancement on Pi 5.
Receives a WAV file from the laptop over HTTP, runs the existing
NoiseProcessor pipeline, and returns the enhanced audio with telemetry.

Architecture
------------
This is intentionally NOT the real-time UDP streaming path. It processes
complete WAV files (batch/offline) and returns the result + metadata.

Endpoints
---------
POST /process    Accept WAV audio (multipart or base64 JSON), enhance,
                 return enhanced audio + processing telemetry.
GET  /health     Liveness check and model-loaded status.

Usage
-----
    python inference_server.py --port 8090
    python inference_server.py --port 8090 --model-dir /path/to/models/dtln_quantized
"""

from __future__ import annotations

import argparse
import base64
import http.server
import io
import json
import os
import platform
import socketserver
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — ensure repo src/ is importable (matches process_audio.py pattern)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from process_audio import NoiseProcessor  # noqa: E402

# ---------------------------------------------------------------------------
# Telemetry helpers (stdlib only — no psutil dependency)
# ---------------------------------------------------------------------------

def _get_cpu_load_percent() -> float:
    """Estimate CPU load percentage from os.getloadavg().

    Uses 1-minute load average scaled to core count.
    Falls back to 0.0 on platforms without getloadavg (Windows).
    """
    try:
        load1, _, _ = os.getloadavg()
        cores = os.cpu_count() or 4
        return min(round(load1 / cores * 100.0, 1), 100.0)
    except (OSError, AttributeError):
        return 0.0


def _get_ram_info() -> dict[str, float]:
    """Parse /proc/meminfo for RAM usage (Linux only).

    Returns dict with ram_total_mb, ram_used_mb, ram_percent.
    Falls back to zeros on non-Linux platforms.
    """
    try:
        with open("/proc/meminfo", "r") as f:
            lines = f.readlines()
        info: dict[str, int] = {}
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0].rstrip(":")
                info[key] = int(parts[1])  # kB

        total_kb = info.get("MemTotal", 0)
        available_kb = info.get("MemAvailable", info.get("MemFree", 0))
        used_kb = total_kb - available_kb
        total_mb = total_kb / 1024.0
        used_mb = used_kb / 1024.0
        percent = (used_kb / max(total_kb, 1)) * 100.0
        return {
            "ram_total_mb": round(total_mb, 1),
            "ram_used_mb": round(used_mb, 1),
            "ram_percent": round(percent, 1),
        }
    except (FileNotFoundError, PermissionError, ValueError):
        return {"ram_total_mb": 0.0, "ram_used_mb": 0.0, "ram_percent": 0.0}


def _get_cpu_temperature() -> float | None:
    """Read CPU temperature via vcgencmd (Raspberry Pi OS).

    Falls back to /sys/class/thermal on generic Linux, or None on
    non-Pi / non-Linux platforms (e.g. dev testing on a laptop).
    """
    # Method 1: vcgencmd (Pi-specific, most reliable)
    try:
        result = subprocess.run(
            ["vcgencmd", "measure_temp"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0:
            # Output format: "temp=42.8'C"
            temp_str = result.stdout.strip().split("=")[1].replace("'C", "")
            return round(float(temp_str), 1)
    except (FileNotFoundError, IndexError, ValueError, subprocess.TimeoutExpired, OSError):
        pass

    # Method 2: sysfs thermal zone (generic Linux)
    try:
        thermal_path = Path("/sys/class/thermal/thermal_zone0/temp")
        if thermal_path.exists():
            raw = thermal_path.read_text().strip()
            return round(int(raw) / 1000.0, 1)
    except (ValueError, PermissionError, OSError):
        pass

    return None


def _collect_telemetry() -> dict[str, Any]:
    """Collect system telemetry at request time."""
    ram = _get_ram_info()
    temp = _get_cpu_temperature()
    return {
        "cpu_load_percent": _get_cpu_load_percent(),
        "ram_total_mb": ram["ram_total_mb"],
        "ram_used_mb": ram["ram_used_mb"],
        "ram_percent": ram["ram_percent"],
        "temperature_c": temp,  # None if unavailable
    }


# ---------------------------------------------------------------------------
# HTTP Request Handler
# ---------------------------------------------------------------------------

class InferenceRequestHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler for Pi 5 file-based inference."""

    processor: NoiseProcessor | None = None

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.end_headers()

    def do_GET(self) -> None:
        if self.path.rstrip("/") in ("/health", "/api/health"):
            self._handle_health()
        else:
            self.send_error(404, f"Path not found: {self.path}")

    def do_POST(self) -> None:
        if self.path.rstrip("/") in ("/process", "/api/process"):
            self._handle_process()
        else:
            self.send_error(404, f"Endpoint not found: {self.path}")

    def _json_response(self, data: dict[str, Any], status: int = 200) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _handle_health(self) -> None:
        model_loaded = self.processor is not None
        self._json_response({
            "status": "ok",
            "model_loaded": model_loaded,
            "model_name": self.processor.model_name if model_loaded else None,
            "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
            "sample_rate": self.processor.sample_rate if model_loaded else 16000,
        })

    def _handle_process(self) -> None:
        """Process uploaded audio through DTLN pipeline."""
        if self.processor is None:
            self._json_response(
                {"success": False, "error": "Model not loaded", "enhanced_audio_base64": None,
                 "sample_rate": 16000, "processing_time_ms": 0, "realtime_ratio": 0,
                 "telemetry": {}, },
                status=503,
            )
            return

        try:
            audio_bytes = self._extract_audio_bytes()
        except ValueError as e:
            self._json_response(
                {"success": False, "error": str(e), "enhanced_audio_base64": None,
                 "sample_rate": 16000, "processing_time_ms": 0, "realtime_ratio": 0,
                 "telemetry": {}, },
                status=400,
            )
            return

        try:
            # Decode WAV to float64 mono array
            import soundfile as sf

            buf = io.BytesIO(audio_bytes)
            try:
                audio, sr = sf.read(buf)
            except Exception:
                # Fallback to scipy
                import scipy.io.wavfile
                buf.seek(0)
                sr, raw = scipy.io.wavfile.read(buf)
                if raw.dtype == np.int16:
                    audio = raw.astype(np.float64) / 32768.0
                elif raw.dtype == np.int32:
                    audio = raw.astype(np.float64) / 2147483648.0
                else:
                    audio = raw.astype(np.float64)

            # Ensure mono
            if audio.ndim > 1:
                audio = audio[:, 0]
            audio = audio.astype(np.float64)

            # Resample to model rate if needed
            target_sr = self.processor.sample_rate
            if sr != target_sr:
                import scipy.signal
                num_samples = int(len(audio) * float(target_sr) / sr)
                audio = scipy.signal.resample(audio, num_samples)
                sr = target_sr

            # Collect pre-processing telemetry
            telemetry = _collect_telemetry()

            # Run DTLN inference
            t_start = time.perf_counter()
            enhanced = self.processor.process(audio)
            t_end = time.perf_counter()

            processing_time_s = t_end - t_start
            audio_duration_s = len(audio) / float(sr)
            realtime_ratio = processing_time_s / max(audio_duration_s, 1e-6)

            # TODO: Per-stage inference timing (STFT/mask stage vs time-domain stage)
            # would require refactoring LocalPiAI.process_frame() inner loop.
            # Reporting single total inference time for now.

            # Encode enhanced audio as base64 WAV
            pcm = (np.clip(enhanced, -1.0, 1.0) * 32767.0).astype(np.int16)
            wav_buf = io.BytesIO()
            import scipy.io.wavfile
            scipy.io.wavfile.write(wav_buf, sr, pcm)
            enhanced_b64 = base64.b64encode(wav_buf.getvalue()).decode("utf-8")

            self._json_response({
                "success": True,
                "enhanced_audio_base64": enhanced_b64,
                "sample_rate": sr,
                "processing_time_ms": round(processing_time_s * 1000.0, 1),
                "realtime_ratio": round(realtime_ratio, 3),
                "telemetry": telemetry,
                "error": None,
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._json_response(
                {"success": False, "error": f"Processing error: {str(e)}",
                 "enhanced_audio_base64": None, "sample_rate": 16000,
                 "processing_time_ms": 0, "realtime_ratio": 0,
                 "telemetry": {}, },
                status=500,
            )

    def _extract_audio_bytes(self) -> bytes:
        """Extract raw WAV bytes from either multipart/form-data or base64 JSON body.

        Raises ValueError with a descriptive message on bad input.
        """
        content_type = self.headers.get("Content-Type", "")
        content_length = int(self.headers.get("Content-Length", 0))

        if "multipart/form-data" in content_type:
            # Manual multipart parsing (stdlib-only, no cgi dependency)
            try:
                raw_body = self.rfile.read(content_length)
                # Extract boundary from Content-Type header
                boundary = None
                for part in content_type.split(";"):
                    part = part.strip()
                    if part.startswith("boundary="):
                        boundary = part.split("=", 1)[1].strip('"')
                        break
                if not boundary:
                    raise ValueError("No boundary found in multipart Content-Type")

                boundary_bytes = boundary.encode("utf-8")
                # Split body on boundary markers
                parts = raw_body.split(b"--" + boundary_bytes)
                for part in parts:
                    # Skip preamble and epilogue
                    if not part or part.strip() in (b"", b"--", b"--\r\n"):
                        continue
                    # Find the blank line separating headers from body
                    header_end = part.find(b"\r\n\r\n")
                    if header_end == -1:
                        continue
                    headers_section = part[:header_end].decode("utf-8", errors="replace")
                    body_data = part[header_end + 4:]
                    # Remove trailing \r\n before next boundary
                    if body_data.endswith(b"\r\n"):
                        body_data = body_data[:-2]

                    # Check if this part contains a file field named 'file'
                    if 'name="file"' in headers_section or 'name="audio"' in headers_section:
                        return body_data

                raise ValueError("No 'file' field found in multipart upload")
            except ValueError:
                raise
            except Exception as e:
                raise ValueError(f"Failed to parse multipart upload: {e}")

        else:
            # Assume JSON body with audio_base64 field
            try:
                raw = self.rfile.read(content_length).decode("utf-8")
                body = json.loads(raw)
            except Exception as e:
                raise ValueError(f"Invalid JSON payload: {e}")

            audio_b64 = body.get("audio_base64")
            if not audio_b64:
                raise ValueError(
                    "JSON body must contain 'audio_base64' field with base64-encoded WAV data."
                )

            # Strip data URI prefix if present (e.g. "data:audio/wav;base64,...")
            if "," in audio_b64:
                audio_b64 = audio_b64.split(",", 1)[1]

            try:
                return base64.b64decode(audio_b64)
            except Exception as e:
                raise ValueError(f"Invalid base64 audio data: {e}")

    def log_message(self, format: str, *args: Any) -> None:
        """Log all requests to stderr (useful for debugging on Pi)."""
        sys.stderr.write(
            f"[Pi5-Inference] {self.client_address[0]} - {format % args}\n"
        )


# ---------------------------------------------------------------------------
# Threaded HTTP Server (matches src/anc/interface/server.py pattern)
# ---------------------------------------------------------------------------

class ThreadedInferenceServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """Multi-threaded HTTP server for concurrent inference requests."""
    daemon_threads = True
    allow_reuse_address = True


# ---------------------------------------------------------------------------
# Server launcher
# ---------------------------------------------------------------------------

def start_inference_server(
    port: int = 8090,
    model_dir: str | Path | None = None,
    sample_rate: int = 16_000,
) -> None:
    """Launch the Pi 5 inference server."""
    print(f"\n{'='*70}")
    print("PS26052 ANC — Raspberry Pi 5 Inference Server")
    print(f"{'='*70}")

    # Initialize NoiseProcessor (reuses existing process_audio.py engine)
    try:
        processor = NoiseProcessor(sample_rate=sample_rate)
        InferenceRequestHandler.processor = processor
        print(f"  Model:       {processor.model_name}")
        print(f"  Sample Rate: {sample_rate} Hz")
    except Exception as e:
        print(f"  [ERROR] Failed to load model: {e}")
        print(f"  Server will start but /process will return 503.")
        InferenceRequestHandler.processor = None

    # Bind to all interfaces (0.0.0.0) — required for direct Ethernet access
    server_address = ("", port)
    server = ThreadedInferenceServer(server_address, InferenceRequestHandler)

    print(f"  Listening:   0.0.0.0:{port}")
    print(f"  Health:      http://localhost:{port}/health")
    print(f"  Process:     POST http://localhost:{port}/process")
    print(f"  Press Ctrl+C to stop.")
    print(f"{'='*70}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Pi 5 Inference Server...")
        server.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PS26052 ANC — Raspberry Pi 5 Inference Server",
    )
    parser.add_argument(
        "--port", type=int, default=8090,
        help="HTTP port to listen on (default: 8090)",
    )
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help="Path to DTLN model directory (default: auto-detect from repo)",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=16000,
        help="Audio sample rate in Hz (default: 16000)",
    )
    args = parser.parse_args()

    start_inference_server(
        port=args.port,
        model_dir=args.model_dir,
        sample_rate=args.sample_rate,
    )


if __name__ == "__main__":
    main()
