"""Multi-threaded HTTP Server for PS26052 ANC Unified Interface.

Provides REST APIs for audio processing, spectrogram generation, preset loading,
and hardware inspection, alongside serving the modern web application frontend.
"""

from __future__ import annotations

import argparse
import base64
import http.server
import json
import os
import platform
import socket
import socketserver
import sys
import threading
import time
import urllib.parse
import webbrowser
from dataclasses import asdict
from pathlib import Path
from typing import Any

from anc.interface.processor import AudioProcessingPipeline
from anc.interface.pi_client import PiClient


class InterfaceRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP Request handler for the unified interface."""

    pipeline: AudioProcessingPipeline
    base_dir: Path = Path(__file__).resolve().parent

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.end_headers()

    def do_GET(self) -> None:
        url = urllib.parse.urlparse(self.path)
        path = url.path.rstrip("/")

        if path in ("", "/index.html"):
            self._serve_file(self.base_dir / "templates" / "index.html", "text/html; charset=utf-8")
        elif path == "/static/style.css":
            self._serve_file(self.base_dir / "static" / "style.css", "text/css; charset=utf-8")
        elif path == "/static/app.js":
            self._serve_file(self.base_dir / "static" / "app.js", "application/javascript; charset=utf-8")
        elif path == "/api/presets":
            self._handle_get_presets()
        elif path == "/api/hardware":
            self._handle_get_hardware()
        elif path == "/api/pi-health":
            self._handle_get_pi_health(url)
        elif path == "/api/health":
            self._json_response({"status": "ok", "app": "PS26052 ANC Unified Interface"})
        else:
            self.send_error(404, f"Path not found: {url.path}")

    def do_POST(self) -> None:
        url = urllib.parse.urlparse(self.path)

        if url.path == "/api/process":
            self._handle_post_process()
        elif url.path == "/api/process-remote":
            self._handle_post_process_remote()
        else:
            self.send_error(404, f"API endpoint not found: {url.path}")

    def _serve_file(self, file_path: Path, content_type: str) -> None:
        if not file_path.exists():
            self.send_error(404, f"File not found: {file_path.name}")
            return
        content = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content)

    def _json_response(self, data: dict[str, Any], status: int = 200) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _handle_get_presets(self) -> None:
        try:
            presets = self.pipeline.get_available_presets()
            self._json_response({"success": True, "presets": presets})
        except Exception as e:
            self._json_response({"success": False, "error": str(e)}, status=500)

    def _handle_get_hardware(self) -> None:
        """Report host hardware and model status."""
        models_status = {
            "dtln_quantized": (self.pipeline.models_dir / "dtln_quantized").exists(),
            "dtln_finetuned": (self.pipeline.models_dir / "dtln_finetuned" / "best_model.pt").exists(),
            "dtln_stock": (self.pipeline.models_dir / "dtln").exists(),
            "spectral_gate": True,
        }

        data = {
            "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
            "python": platform.python_version(),
            "cpu_count": os.cpu_count() or 4,
            "models": models_status,
            "sample_rate": 16000,
            "frame_ms": 20,
            "recommended_model": "dtln_quantized" if models_status["dtln_quantized"] else "dtln_stock",
        }
        self._json_response(data)

    def _handle_get_pi_health(self, url: urllib.parse.ParseResult) -> None:
        """Proxy health check to the Pi 5 to circumvent browser-side CORS/mDNS restrictions."""
        query = urllib.parse.parse_qs(url.query)
        host = query.get("host", ["raspberrypi.local"])[0]
        port_str = query.get("port", ["8090"])[0]
        try:
            port = int(port_str)
        except ValueError:
            port = 8090

        client = PiClient(host=host, port=port, timeout_seconds=3)
        health = client.check_health()
        if health is not None:
            self._json_response({"connected": True, "data": health})
        else:
            self._json_response({"connected": False, "error": f"Unable to reach Pi at {host}:{port}"})

    def _handle_post_process(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(length).decode("utf-8")
            body = json.loads(raw_body)
        except Exception as e:
            self._json_response({"success": False, "error": f"Invalid JSON payload: {e}"}, status=400)
            return

        preset_id = body.get("preset_id")
        audio_b64 = body.get("audio_base64")
        mode = body.get("mode", "hybrid")
        model_name = body.get("model_name", "dtln_quantized")
        filter_length = int(body.get("filter_length", 64))
        step_size = float(body.get("step_size", 0.01))

        audio_in = None
        clean_ref = None
        noise_ref = None
        sr = 16000

        try:
            if preset_id:
                # Load preset (returns noisy, clean_ref, noise_ref, sr)
                audio_in, clean_ref, noise_ref, sr = self.pipeline.load_preset_audio(preset_id)
            elif audio_b64:
                # Decode base64 audio (handles mono and stereo)
                if "," in audio_b64:
                    audio_b64 = audio_b64.split(",", 1)[1]
                audio_bytes = base64.b64decode(audio_b64)
                audio_in, noise_ref, sr = self.pipeline.decode_audio_bytes(audio_bytes)
                clean_ref = None
            else:
                self._json_response({"success": False, "error": "No audio input provided. Supply preset_id or audio_base64."}, status=400)
                return

            # Execute pipeline
            result = self.pipeline.process(
                audio_in=audio_in,
                sample_rate=sr,
                mode=mode,
                model_name=model_name,
                filter_length=filter_length,
                step_size=step_size,
                noise_reference=noise_ref,
                clean_reference=clean_ref,
            )

            response_data = {
                "success": result.success,
                "message": result.message,
                "metrics": asdict(result.metrics),
                "enhanced_audio_base64": result.enhanced_audio_base64,
                "input_audio_base64": result.input_audio_base64,
                "reference_audio_base64": result.reference_audio_base64,
                "input_spectrogram_base64": result.input_spectrogram_base64,
                "output_spectrogram_base64": result.output_spectrogram_base64,
                "difference_spectrogram_base64": result.difference_spectrogram_base64,
                "model_name": result.model_name,
                "mode": result.mode,
                "filter_taps": result.filter_taps,
                "sample_rate": result.sample_rate,
            }
            self._json_response(response_data)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._json_response({"success": False, "error": f"Processing error: {str(e)}"}, status=500)

    def _handle_post_process_remote(self) -> None:
        """Process audio on a remote Raspberry Pi 5 via inference_server.py."""
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(length).decode("utf-8")
            body = json.loads(raw_body)
        except Exception as e:
            self._json_response({"success": False, "error": f"Invalid JSON payload: {e}"}, status=400)
            return

        pi_host = body.get("pi_host", "raspberrypi.local")
        pi_port = int(body.get("pi_port", 8090))
        preset_id = body.get("preset_id")
        audio_b64 = body.get("audio_base64")

        audio_in = None
        clean_ref = None
        sr = 16000

        try:
            # Load audio source (reuse existing logic — same as /api/process)
            if preset_id:
                audio_in, clean_ref, _, sr = self.pipeline.load_preset_audio(preset_id)
            elif audio_b64:
                if "," in audio_b64:
                    audio_b64 = audio_b64.split(",", 1)[1]
                audio_bytes = base64.b64decode(audio_b64)
                audio_in, _, sr = self.pipeline.decode_audio_bytes(audio_bytes)
                clean_ref = None
            else:
                self._json_response(
                    {"success": False, "error": "No audio input provided. Supply preset_id or audio_base64."},
                    status=400,
                )
                return

            # Encode input audio as WAV bytes for the Pi
            import io
            import scipy.io.wavfile
            import numpy as np

            clipped = np.clip(audio_in, -1.0, 1.0)
            pcm = (clipped * 32767.0).astype(np.int16)
            wav_buf = io.BytesIO()
            scipy.io.wavfile.write(wav_buf, sr, pcm)
            wav_bytes = wav_buf.getvalue()

            # Send to Pi 5 for inference
            client = PiClient(host=pi_host, port=pi_port)
            pi_result = client.send_for_processing(wav_bytes)

            if not pi_result.get("success", False):
                self._json_response(
                    {"success": False, "error": pi_result.get("error", "Unknown Pi error")},
                    status=502,
                )
                return

            # Decode enhanced audio from Pi response
            enhanced_b64 = pi_result.get("enhanced_audio_base64", "")
            enhanced_bytes = base64.b64decode(enhanced_b64)
            enhanced_audio, _, enh_sr = self.pipeline.decode_audio_bytes(enhanced_bytes)

            # Build ProcessResult with metrics/spectrograms (no local inference)
            result = self.pipeline.process_remote(
                audio_in=audio_in,
                enhanced_audio=enhanced_audio,
                sample_rate=sr,
                clean_reference=clean_ref,
                model_name=f"Pi 5 DTLN (Remote @ {pi_host}:{pi_port})",
                pi_processing_time_ms=pi_result.get("processing_time_ms", 0.0),
                pi_realtime_ratio=pi_result.get("realtime_ratio", 0.0),
            )

            response_data = {
                "success": result.success,
                "message": result.message,
                "metrics": asdict(result.metrics),
                "enhanced_audio_base64": result.enhanced_audio_base64,
                "input_audio_base64": result.input_audio_base64,
                "reference_audio_base64": result.reference_audio_base64,
                "input_spectrogram_base64": result.input_spectrogram_base64,
                "output_spectrogram_base64": result.output_spectrogram_base64,
                "difference_spectrogram_base64": result.difference_spectrogram_base64,
                "model_name": result.model_name,
                "mode": result.mode,
                "filter_taps": result.filter_taps,
                "sample_rate": result.sample_rate,
                "has_clean_reference": clean_ref is not None,
                "telemetry": pi_result.get("telemetry", {}),
            }
            self._json_response(response_data)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._json_response({"success": False, "error": f"Remote processing error: {str(e)}"}, status=500)

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress routine static asset polling logs
        pass


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """Multi-threaded HTTP server allowing non-blocking concurrent requests.

    Supports dual-stack (IPv6 + IPv4) so that modern web browsers resolving
    'localhost' to ::1 or 127.0.0.1 can connect without 'connection refused' errors.
    """
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], RequestHandlerClass: type) -> None:
        try:
            self.address_family = socket.AF_INET6
            super().__init__(("::", server_address[1]), RequestHandlerClass)
            return
        except Exception:
            pass
        self.address_family = socket.AF_INET
        super().__init__(("", server_address[1]), RequestHandlerClass)

    def server_bind(self) -> None:
        if self.address_family == socket.AF_INET6:
            try:
                self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            except (AttributeError, OSError):
                pass
        super().server_bind()


def start_server(
    port: int = 8080,
    open_browser: bool = True,
    repo_root: Path | None = None,
) -> None:
    """Launch the unified ANC command interface server."""
    pipeline = AudioProcessingPipeline(repo_root=repo_root)
    InterfaceRequestHandler.pipeline = pipeline
    InterfaceRequestHandler.base_dir = Path(__file__).resolve().parent

    server_address = ("", port)
    server = ThreadedHTTPServer(server_address, InterfaceRequestHandler)

    url = f"http://localhost:{port}"
    print(f"\n{'='*75}")
    print(f"PS26052 ANC — UNIFIED COMMAND CENTER & DELIVERABLE")
    print(f"{'='*75}")
    print(f"  Interface URL:    {url}")
    print(f"  Audio Sample Rate: 16,000 Hz")
    print(f"  Available Presets: {len(pipeline.get_available_presets())}")
    print(f"  Ready for Audio Upload, Live Mic, & Spectrogram Analytics")
    print(f"  Press Ctrl+C to terminate.")
    print(f"{'='*75}\n")

    if open_browser:
        def _open():
            time.sleep(1.0)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down PS26052 ANC Unified Command Center...")
        server.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="PS26052 ANC Unified Command Center")
    parser.add_argument("--port", type=int, default=8080, help="Web server port (default: 8080)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args()

    start_server(port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
