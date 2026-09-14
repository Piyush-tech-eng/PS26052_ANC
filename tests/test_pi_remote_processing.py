"""Tests for Pi 5 remote processing: client error handling, processor.process_remote(),
and inference server endpoints.

Follows the existing test conventions from test_interface.py and test_speech_metrics.py.
"""

from __future__ import annotations

import base64
import io
import json
import socket
import threading
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import scipy.io.wavfile

from anc.interface.pi_client import PiClient, PiConnectionConfig, PI5_REQUEST_TIMEOUT_SECONDS
from anc.interface.processor import AudioProcessingPipeline, ProcessResult


# ---------------------------------------------------------------------------
# Helper: generate synthetic WAV bytes
# ---------------------------------------------------------------------------

def _make_wav_bytes(duration_s: float = 0.3, sr: int = 16000) -> bytes:
    """Generate a synthetic noisy speech-like WAV as raw bytes."""
    t = np.linspace(0, duration_s, int(sr * duration_s))
    audio = (0.5 * np.sin(2 * np.pi * 400 * t) + 0.1 * np.random.randn(len(t))).astype(np.float64)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    scipy.io.wavfile.write(buf, sr, pcm)
    return buf.getvalue()


def _get_free_port() -> int:
    """Acquire an ephemeral free port from the OS."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# ===========================================================================
# Test PiClient error handling
# ===========================================================================

class TestPiClient:
    """Test Pi client graceful error handling when Pi is unreachable."""

    def test_unreachable_pi_returns_error(self) -> None:
        """Connecting to an invalid host should return an error dict, not crash."""
        client = PiClient(host="192.0.2.1", port=1, timeout_seconds=2)
        result = client.send_for_processing(b"fake wav data")
        assert isinstance(result, dict)
        assert result["success"] is False
        assert "error" in result
        assert len(result["error"]) > 0

    def test_connection_timeout(self) -> None:
        """Short timeout should fire correctly against a non-routable address."""
        client = PiClient(host="192.0.2.1", port=1, timeout_seconds=1)
        result = client.send_for_processing(b"data")
        assert result["success"] is False

    def test_health_check_unreachable(self) -> None:
        """Health check against offline Pi should return None."""
        client = PiClient(host="192.0.2.1", port=1, timeout_seconds=1)
        result = client.check_health()
        assert result is None

    def test_config_defaults(self) -> None:
        """Verify default config uses mDNS hostname and named timeout constant."""
        config = PiConnectionConfig()
        assert config.host == "raspberrypi.local"
        assert config.port == 8090
        assert config.timeout_seconds == PI5_REQUEST_TIMEOUT_SECONDS


# ===========================================================================
# Test processor.process_remote()
# ===========================================================================

class TestProcessorRemote:
    """Test the process_remote() method in AudioProcessingPipeline."""

    def test_process_remote_with_clean_reference(self) -> None:
        """With a clean reference, SI-SNR and STOI metrics should be populated."""
        pipeline = AudioProcessingPipeline()
        sr = 16000
        t = np.linspace(0, 0.5, int(sr * 0.5))
        clean_ref = 0.5 * np.sin(2 * np.pi * 400 * t)
        noise = 0.3 * np.random.randn(len(t))
        noisy = clean_ref + noise
        # Simulate enhanced audio (slightly denoised)
        enhanced = clean_ref + 0.05 * np.random.randn(len(t))

        result = pipeline.process_remote(
            audio_in=noisy,
            enhanced_audio=enhanced,
            sample_rate=sr,
            clean_reference=clean_ref,
            model_name="Test Pi 5 DTLN",
            pi_processing_time_ms=42.0,
            pi_realtime_ratio=0.085,
        )

        assert isinstance(result, ProcessResult)
        assert result.success is True
        assert result.mode == "pi5_remote"
        assert result.model_name == "Test Pi 5 DTLN"
        # SI-SNR should be populated
        assert result.metrics.si_snr_output_db is not None
        assert result.metrics.si_snr_improvement_db is not None
        # Pi timing should be propagated
        assert result.metrics.ai_ms == 42.0
        assert result.metrics.realtime_ratio == 0.085

    def test_process_remote_without_reference(self) -> None:
        """Without a clean reference (arbitrary upload), STOI/SI-SNR should be None."""
        pipeline = AudioProcessingPipeline()
        sr = 16000
        t = np.linspace(0, 0.3, int(sr * 0.3))
        noisy = 0.4 * np.sin(2 * np.pi * 300 * t) + 0.2 * np.random.randn(len(t))
        enhanced = 0.35 * np.sin(2 * np.pi * 300 * t)

        result = pipeline.process_remote(
            audio_in=noisy,
            enhanced_audio=enhanced,
            sample_rate=sr,
            clean_reference=None,
        )

        assert result.success is True
        # No reference → metrics should be None (not fabricated)
        assert result.metrics.si_snr_input_db is None
        assert result.metrics.si_snr_output_db is None
        assert result.metrics.stoi_input is None
        assert result.metrics.stoi_output is None
        # Attenuation should still be computed (no-reference metric)
        assert result.metrics.estimated_attenuation_db is not None

    def test_spectrograms_generated(self) -> None:
        """Spectrograms should be generated as base64 PNG data URIs."""
        pipeline = AudioProcessingPipeline()
        sr = 16000
        t = np.linspace(0, 0.3, int(sr * 0.3))
        noisy = 0.4 * np.sin(2 * np.pi * 400 * t)
        enhanced = 0.3 * np.sin(2 * np.pi * 400 * t)

        result = pipeline.process_remote(
            audio_in=noisy,
            enhanced_audio=enhanced,
            sample_rate=sr,
        )

        assert result.input_spectrogram_base64.startswith("data:image/png;base64,")
        assert result.output_spectrogram_base64.startswith("data:image/png;base64,")
        assert result.difference_spectrogram_base64.startswith("data:image/png;base64,")
        assert result.enhanced_audio_base64.startswith("data:audio/wav;base64,")
        assert result.input_audio_base64.startswith("data:audio/wav;base64,")


# ===========================================================================
# Test inference server endpoints (run locally over loopback)
# ===========================================================================

@pytest.fixture(scope="module")
def inference_server_url():
    """Start inference server on ephemeral port for the test module."""
    import sys
    repo_root = Path(__file__).resolve().parent.parent
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from pi.runtime.inference_server import (
        InferenceRequestHandler,
        ThreadedInferenceServer,
    )

    port = _get_free_port()

    # Try to load processor; if model isn't available, skip tests
    try:
        from process_audio import NoiseProcessor
        processor = NoiseProcessor(sample_rate=16000)
        InferenceRequestHandler.processor = processor
    except Exception:
        pytest.skip("NoiseProcessor / ONNX model not available for inference server tests")

    server = ThreadedInferenceServer(("", port), InferenceRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    yield base_url

    server.shutdown()
    server.server_close()


class TestInferenceServerEndpoints:
    """Test Pi 5 inference server without real Pi hardware.

    Spins up inference_server.py on an ephemeral port over loopback.
    Tests require onnxruntime and the DTLN model files to be available.
    """

    def test_health_endpoint(self, inference_server_url: str) -> None:
        """GET /health should return model status."""
        with urllib.request.urlopen(f"{inference_server_url}/health") as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] == "ok"
            assert data["model_loaded"] is True
            assert isinstance(data["model_name"], str)
            assert len(data["model_name"]) > 0

    def test_process_endpoint_base64(self, inference_server_url: str) -> None:
        """POST /process with base64 JSON should return enhanced audio."""
        wav_bytes = _make_wav_bytes(duration_s=0.3)
        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
        payload = json.dumps({"audio_base64": audio_b64}).encode("utf-8")

        req = urllib.request.Request(
            f"{inference_server_url}/process",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["success"] is True
            assert isinstance(data["enhanced_audio_base64"], str)
            assert len(data["enhanced_audio_base64"]) > 0
            assert data["sample_rate"] == 16000
            assert data["processing_time_ms"] > 0
            assert data["realtime_ratio"] > 0
            assert isinstance(data["telemetry"], dict)
            assert "cpu_load_percent" in data["telemetry"]

    def test_process_endpoint_invalid_audio(self, inference_server_url: str) -> None:
        """POST /process with garbage data should return a clear error."""
        payload = json.dumps({"audio_base64": "dGhpcyBpcyBub3QgYXVkaW8="}).encode("utf-8")

        req = urllib.request.Request(
            f"{inference_server_url}/process",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                # Should get an error response (either 400 or 500)
                assert data["success"] is False
        except urllib.error.HTTPError as e:
            data = json.loads(e.read().decode("utf-8"))
            assert data["success"] is False
            assert "error" in data

    def test_process_endpoint_missing_audio(self, inference_server_url: str) -> None:
        """POST /process with no audio field should return 400."""
        payload = json.dumps({"mode": "test"}).encode("utf-8")

        req = urllib.request.Request(
            f"{inference_server_url}/process",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                assert data["success"] is False
        except urllib.error.HTTPError as e:
            assert e.code == 400
            data = json.loads(e.read().decode("utf-8"))
            assert data["success"] is False


@pytest.fixture(scope="module")
def dashboard_server_url():
    from anc.interface.server import InterfaceRequestHandler, ThreadedHTTPServer
    port = _get_free_port()
    pipeline = AudioProcessingPipeline()
    InterfaceRequestHandler.pipeline = pipeline
    InterfaceRequestHandler.base_dir = (
        Path(__file__).resolve().parent.parent / "src" / "anc" / "interface"
    )
    server = ThreadedHTTPServer(("", port), InterfaceRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    yield base_url

    server.shutdown()
    server.server_close()


class TestDashboardRemoteEndpoints:
    """Test dashboard server's /api/pi-health and /api/process-remote endpoints."""

    def test_pi_health_proxy_unreachable(self, dashboard_server_url: str) -> None:
        """Dashboard /api/pi-health proxy to invalid port returns connected=False."""
        url = f"{dashboard_server_url}/api/pi-health?host=127.0.0.1&port=59999"
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert data["connected"] is False

    def test_pi_health_proxy_live(self, dashboard_server_url: str, inference_server_url: str) -> None:
        """Dashboard /api/pi-health proxy to live inference server returns connected=True."""
        port = inference_server_url.split(":")[-1]
        url = f"{dashboard_server_url}/api/pi-health?host=127.0.0.1&port={port}"
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert data["connected"] is True
            assert data["data"]["status"] == "ok"

    def test_process_remote_unreachable_pi(self, dashboard_server_url: str) -> None:
        """POST /api/process-remote with unreachable Pi returns 502."""
        wav_bytes = _make_wav_bytes(duration_s=0.3)
        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
        payload = json.dumps({
            "audio_base64": f"data:audio/wav;base64,{audio_b64}",
            "pi_host": "127.0.0.1",
            "pi_port": 59999,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{dashboard_server_url}/api/process-remote",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                assert data["success"] is False
        except urllib.error.HTTPError as e:
            assert e.code == 502
            data = json.loads(e.read().decode("utf-8"))
            assert data["success"] is False
            assert "unreachable" in data["error"].lower()

    def test_process_remote_end_to_end(self, dashboard_server_url: str, inference_server_url: str) -> None:
        """POST /api/process-remote end-to-end through dashboard to inference server."""
        pi_port = int(inference_server_url.split(":")[-1])
        wav_bytes = _make_wav_bytes(duration_s=0.3)
        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
        payload = json.dumps({
            "audio_base64": f"data:audio/wav;base64,{audio_b64}",
            "pi_host": "127.0.0.1",
            "pi_port": pi_port,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{dashboard_server_url}/api/process-remote",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert data["success"] is True
            assert data["mode"] == "pi5_remote"
            assert "enhanced_audio_base64" in data
            assert "telemetry" in data
            assert "output_spectrogram_base64" in data
            assert "input_spectrogram_base64" in data

