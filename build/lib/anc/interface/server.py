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
        elif path == "/api/health":
            self._json_response({"status": "ok", "app": "PS26052 ANC Unified Interface"})
        else:
            self.send_error(404, f"Path not found: {url.path}")

    def do_POST(self) -> None:
        url = urllib.parse.urlparse(self.path)

        if url.path == "/api/process":
            self._handle_post_process()
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

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress routine static asset polling logs
        pass


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """Multi-threaded HTTP server allowing non-blocking concurrent requests."""
    daemon_threads = True
    allow_reuse_address = True


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
