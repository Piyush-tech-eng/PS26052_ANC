from __future__ import annotations

import numpy as np
import pytest

from anc.evaluation import compute_band_attenuation, compute_noise_reduction_db, compute_segment_mse, evaluate_scenario


def test_standard_metrics_report_known_power_reduction() -> None:
    baseline = np.ones(1_024)
    residual = 0.5 * baseline
    assert compute_noise_reduction_db(baseline, residual) == pytest.approx(6.020599913279624)
    assert compute_segment_mse(residual, np.zeros_like(residual))["rmse"] == pytest.approx(0.5)
    metrics = evaluate_scenario(residual, sampling_rate_hz=8_000, baseline_residual=baseline)
    assert metrics["attenuation_db_vs_no_control"] == pytest.approx(6.020599913279624)
    assert "0-500Hz" in compute_band_attenuation(baseline, residual, 8_000)
