"""Integration and acceptance tests for the Unified Common Root Interface."""

from __future__ import annotations

import base64
import json
import socket
import threading
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from anc.interface.processor import AudioProcessingPipeline, ProcessResult
from anc.interface.server import InterfaceRequestHandler, ThreadedHTTPServer


def get_free_port() -> int:
    """Acquire an ephemeral free port from the OS."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def interface_server():
    """Spin up interface server on an ephemeral port for testing."""
    port = get_free_port()
    pipeline = AudioProcessingPipeline()
    InterfaceRequestHandler.pipeline = pipeline
    InterfaceRequestHandler.base_dir = Path(__file__).resolve().parent.parent / "src" / "anc" / "interface"

    server = ThreadedHTTPServer(("", port), InterfaceRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    yield base_url

    server.shutdown()
    server.server_close()


class TestAudioProcessingPipeline:
    """Test core algorithmic pipeline and spectrogram generator."""

    def test_presets_discovery(self) -> None:
        pipeline = AudioProcessingPipeline()
        presets = pipeline.get_available_presets()
        assert len(presets) > 0
        first = presets[0]
        assert "id" in first
        assert "name" in first
        assert "noisy_filename" in first

    def test_hybrid_processing_and_spectrograms(self) -> None:
        pipeline = AudioProcessingPipeline()
        # Synthetic noisy tone
        sr = 16000
        t = np.linspace(0, 0.5, int(sr * 0.5))
        clean_target = 0.5 * np.sin(2 * np.pi * 400 * t)
        noise = 0.3 * np.sin(2 * np.pi * 120 * t) + 0.1 * np.random.randn(len(t))
        noisy_signal = clean_target + noise

        result = pipeline.process(
            audio_in=noisy_signal,
            sample_rate=sr,
            mode="hybrid",
            model_name="dtln_quantized",
            filter_length=64,
            step_size=0.01,
            reference_audio=clean_target,
        )

        assert isinstance(result, ProcessResult)
        assert result.success is True
        assert result.metrics.duration_seconds > 0.4
        assert result.metrics.realtime_ratio > 0.0
        assert result.enhanced_audio_base64.startswith("data:audio/wav;base64,")
        assert result.input_spectrogram_base64.startswith("data:image/png;base64,")
        assert result.output_spectrogram_base64.startswith("data:image/png;base64,")
        assert result.difference_spectrogram_base64.startswith("data:image/png;base64,")

    def test_anc_only_processing(self) -> None:
        pipeline = AudioProcessingPipeline()
        sr = 16000
        t = np.linspace(0, 0.2, int(sr * 0.2))
        audio = 0.4 * np.sin(2 * np.pi * 250 * t)
        noise = 0.2 * np.sin(2 * np.pi * 100 * t)

        # Single-channel ANC test
        res_single = pipeline.process(audio, sample_rate=sr, mode="anc_only", filter_length=32)
        assert res_single.success is True
        assert res_single.mode == "anc_only"

        # Dual-channel FxNLMS test with noise reference
        res_dual = pipeline.process(audio, sample_rate=sr, mode="anc_only", noise_reference=noise, filter_length=32)
        assert res_dual.success is True
        assert res_dual.mode == "anc_only"
        assert "FxNLMS" in res_dual.model_name


class TestInterfaceServerEndpoints:
    """Test HTTP API routes and static asset serving."""

    def test_serve_html(self, interface_server: str) -> None:
        with urllib.request.urlopen(f"{interface_server}/") as resp:
            assert resp.status == 200
            content = resp.read().decode("utf-8")
            assert "PS26052" in content
            assert "ANC COMMAND CENTER" in content
            assert "spectrogram" in content.lower()

    def test_serve_css(self, interface_server: str) -> None:
        with urllib.request.urlopen(f"{interface_server}/static/style.css") as resp:
            assert resp.status == 200
            assert resp.headers.get_content_type() == "text/css"

    def test_serve_js(self, interface_server: str) -> None:
        with urllib.request.urlopen(f"{interface_server}/static/app.js") as resp:
            assert resp.status == 200
            assert resp.headers.get_content_type() == "application/javascript"

    def test_api_presets(self, interface_server: str) -> None:
        with urllib.request.urlopen(f"{interface_server}/api/presets") as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["success"] is True
            assert isinstance(data["presets"], list)
            assert len(data["presets"]) > 0

    def test_api_hardware(self, interface_server: str) -> None:
        with urllib.request.urlopen(f"{interface_server}/api/hardware") as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert "os" in data
            assert "models" in data

    def test_api_process_preset(self, interface_server: str) -> None:
        payload = json.dumps({
            "preset_id": "demo_0_rotor_snr+0",
            "mode": "hybrid",
            "model_name": "dtln_quantized",
            "filter_length": 64,
            "step_size": 0.01,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{interface_server}/api/process",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["success"] is True
            assert "metrics" in data
            assert "input_spectrogram_base64" in data
            assert "output_spectrogram_base64" in data
            assert "difference_spectrogram_base64" in data
            assert data["metrics"]["estimated_attenuation_db"] is not None
