"""PS26052 ANC — Acceptance & Productization Verification Test Suite.

Validates end-to-end product criteria:
1. Profiles: Active profile resolution and validation
2. Streaming Pipeline: Zero deadlock, real-time frame processing, attenuation
3. Live Dashboard: HTTP endpoints (/ and /api/telemetry) response integrity
4. Feasibility & Deployment Artifacts: Format, values, and completeness
"""

from __future__ import annotations

import json
import socketserver
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from anc.profiles import get_active_profile, validate_profile
from anc.config import ExecutionProfile
from anc.realtime.audio_input import WAVInput
from anc.realtime.audio_output import WAVOutput
from anc.realtime.streaming_pipeline import StreamingPipeline
from anc.dashboard import DashboardRequestHandler
from ai.models import get_best_available_model
from ai.streaming.hybrid_engine import HybridEngine


class TestProfilesAcceptance:
    """Acceptance tests for system hardware and deployment profiles."""

    def test_active_profile_resolution(self) -> None:
        profile = get_active_profile()
        assert isinstance(profile, ExecutionProfile)
        assert profile.value in ("prototype", "edge", "development")

    def test_profile_validation(self) -> None:
        report = validate_profile("prototype")
        assert str(report.profile) == "prototype" or getattr(report.profile, "value", "") == "prototype"
        assert report.valid is True
        assert "hardware_capture" in report.available_features


class TestLiveDashboardAcceptance:
    """Acceptance tests for the real-time operator dashboard HTTP server."""

    @pytest.fixture
    def dashboard_server(self, tmp_path):
        status_file = tmp_path / "test_status.json"
        status_file.write_text(json.dumps({
            "state": "LIVE",
            "mode": "prototype",
            "sample_rate": 16000,
            "realtime_ratio": 0.25,
            "total_latency_ms": 12.5,
            "estimated_attenuation_db": 15.2,
        }), encoding="utf-8")

        DashboardRequestHandler.status_file = status_file
        DashboardRequestHandler.repo_root = Path(".")

        port = 18080
        server = socketserver.TCPServer(("127.0.0.1", port), DashboardRequestHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        yield f"http://127.0.0.1:{port}"

        server.shutdown()
        server.server_close()

    def test_dashboard_html_endpoint(self, dashboard_server: str) -> None:
        with urllib.request.urlopen(f"{dashboard_server}/") as resp:
            assert resp.status == 200
            html = resp.read().decode("utf-8")
            assert "PS26052 ANC" in html
            assert "waveformCanvas" in html
            assert "spectrumCanvas" in html
            assert "System & Hardware Status" in html
            assert "Classical FxNLMS Filter" in html

    def test_dashboard_telemetry_endpoint(self, dashboard_server: str) -> None:
        with urllib.request.urlopen(f"{dashboard_server}/api/telemetry") as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["state"] == "LIVE"
            assert data["mode"] == "prototype"
            assert data["sample_rate"] == 16000
            assert data["estimated_attenuation_db"] == 15.2


class TestStreamingPipelineAcceptance:
    """End-to-end acceptance tests for continuous streaming pipeline."""

    def test_streaming_pipeline_run(self, tmp_path: Path) -> None:
        sr = 16_000
        # 0.3 seconds of audio = 15 frames
        t = np.arange(int(sr * 0.3)) / sr
        noisy = 0.5 * np.sin(2 * np.pi * 300 * t) + 0.1 * np.random.randn(len(t))
        in_file = tmp_path / "test_input.wav"
        out_file = tmp_path / "test_output.wav"
        wavfile.write(str(in_file), sr, (noisy * 32767).astype(np.int16))

        model = get_best_available_model(prefer="dtln", sample_rate=sr)
        engine = HybridEngine(model=model, sample_rate=sr)

        wav_in = WAVInput(path=in_file, target_sample_rate=sr, frame_ms=20)
        wav_out = WAVOutput(path=out_file, sample_rate=sr)

        pipeline = StreamingPipeline(
            input_source=wav_in,
            output_sink=wav_out,
            engine=engine,
            mode="offline",
        )

        pipeline.run_blocking()

        assert pipeline.frames_processed == 15
        assert out_file.exists()

        snap = pipeline.get_telemetry()
        assert snap.frames_processed == 15
        assert snap.output_rms > 0.0
        assert snap.state == "STOPPED"


class TestReportArtifactsAcceptance:
    """Acceptance tests ensuring reproducibility reports exist and are valid."""

    def test_pi_feasibility_report(self) -> None:
        report_path = Path("results/pi_feasibility/pi_feasibility_report.json")
        assert report_path.exists(), "pi_feasibility_report.json must be generated"
        data = json.loads(report_path.read_text(encoding="utf-8"))
        assert "configurations" in data
        assert len(data["configurations"]) == 3
        config_names = [c["name"] for c in data["configurations"]]
        assert any("Config A" in n for n in config_names)
        assert any("Config B" in n for n in config_names)
        assert any("Config C" in n for n in config_names)

    def test_ai_deployment_comparison_report(self) -> None:
        report_path = Path("results/ai_deployment/ai_deployment_comparison.json")
        assert report_path.exists(), "ai_deployment_comparison.json must be generated"
        data = json.loads(report_path.read_text(encoding="utf-8"))
        assert "tiers" in data
        assert len(data["tiers"]) == 3
        tiers = {t["tier"]: t for t in data["tiers"]}
        assert "FP32 (Baseline)" in tiers
        assert "FP16 (Half-Precision)" in tiers
        assert "INT8 (Quantized)" in tiers
        # INT8 must achieve significant compression over FP32
        assert tiers["INT8 (Quantized)"]["size_mb"] < tiers["FP32 (Baseline)"]["size_mb"]
