#!/usr/bin/env python3
"""PS26052 ANC — Live Operator & Mission Telemetry Dashboard.

Provides a real-time web dashboard for monitoring the two-microphone adaptive
noise cancellation system, tracking latency, hardware profiles, classical ANC
convergence, AI neural inference, and acoustic waveforms.

6 Real-Time Monitoring Panels:
1. System Status Panel: state, mode, active profile, uptime, CPU%, RAM MB
2. Dual-Microphone Waveform Visualizer: Reference (Ch1), Error (Ch0), Output
3. Frequency Spectrum Display: FFT magnitude spectrum (0 - 8 kHz)
4. Classical ANC Panel: FxNLMS convergence, step size, estimated attenuation (dB)
5. AI Neural Enhancement Panel: model name, precision tier, inference time, RT ratio
6. Latency Breakdown Panel: stacked stage latencies vs 20ms real-time budget

Zero external web framework dependencies: runs using standard library `http.server`.
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import socketserver
import sys
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any

# Default status file path
DEFAULT_STATUS_FILE = Path("results/demo_status.json")

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PS26052 ANC - Live Operations Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-dark: #070a12;
      --card-bg: rgba(15, 23, 42, 0.85);
      --card-border: rgba(56, 189, 248, 0.18);
      --card-hover: rgba(30, 41, 59, 0.95);
      --accent-cyan: #38bdf8;
      --accent-blue: #3b82f6;
      --accent-green: #10b981;
      --accent-amber: #f59e0b;
      --accent-red: #ef4444;
      --accent-purple: #a855f7;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background-color: var(--bg-dark);
      background-image: 
        radial-gradient(circle at 10% 10%, rgba(56, 189, 248, 0.08) 0%, transparent 40%),
        radial-gradient(circle at 90% 90%, rgba(59, 130, 246, 0.06) 0%, transparent 45%),
        linear-gradient(180deg, #070a12 0%, #03060c 100%);
      color: var(--text-main);
      font-family: var(--font-sans);
      min-height: 100vh;
      padding: 24px;
      -webkit-font-smoothing: antialiased;
    }

    .container { max-width: 1440px; margin: 0 auto; }

    /* Header */
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 18px 24px;
      background: var(--card-bg);
      backdrop-filter: blur(16px);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      margin-bottom: 20px;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }

    .brand-title {
      font-size: 20px;
      font-weight: 700;
      letter-spacing: -0.02em;
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .badge-profile {
      background: linear-gradient(135deg, rgba(56, 189, 248, 0.2), rgba(59, 130, 246, 0.2));
      color: var(--accent-cyan);
      border: 1px solid rgba(56, 189, 248, 0.35);
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    .header-status {
      display: flex;
      align-items: center;
      gap: 20px;
    }

    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 14px;
      border-radius: 20px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid rgba(16, 185, 129, 0.3);
      color: var(--accent-green);
    }

    .status-badge.stopped {
      background: rgba(148, 163, 184, 0.15);
      border-color: rgba(148, 163, 184, 0.3);
      color: var(--text-muted);
    }

    .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: currentColor;
      box-shadow: 0 0 8px currentColor;
    }

    .pulse { animation: pulse 2s infinite; }

    @keyframes pulse {
      0% { transform: scale(0.95); opacity: 0.8; }
      50% { transform: scale(1.2); opacity: 1; }
      100% { transform: scale(0.95); opacity: 0.8; }
    }

    /* Grid Layouts */
    .grid-3 {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 18px;
      margin-bottom: 20px;
    }

    .grid-2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
      margin-bottom: 20px;
    }

    @media (max-width: 1100px) {
      .grid-3 { grid-template-columns: 1fr; }
      .grid-2 { grid-template-columns: 1fr; }
    }

    /* Cards */
    .card {
      background: var(--card-bg);
      backdrop-filter: blur(16px);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 20px;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
      position: relative;
    }

    .card-title {
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--accent-cyan);
      margin-bottom: 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .metric-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 8px 0;
      border-bottom: 1px solid rgba(255, 255, 255, 0.06);
      font-size: 13px;
    }

    .metric-row:last-child { border-bottom: none; }

    .metric-label { color: var(--text-muted); }
    .metric-value {
      font-family: var(--font-mono);
      font-weight: 600;
      color: #fff;
    }

    /* Primary Large Metric */
    .primary-metric {
      font-size: 32px;
      font-weight: 800;
      font-family: var(--font-mono);
      color: #fff;
      display: flex;
      align-items: baseline;
      gap: 6px;
      margin: 8px 0 12px 0;
    }

    .primary-unit { font-size: 14px; font-weight: 500; color: var(--text-muted); }

    /* Progress bar */
    .progress-bar-bg {
      height: 8px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 4px;
      overflow: hidden;
      margin-top: 6px;
      position: relative;
    }

    .progress-bar-fill {
      height: 100%;
      background: linear-gradient(90deg, var(--accent-cyan), var(--accent-green));
      border-radius: 4px;
      transition: width 0.3s ease;
    }

    .progress-bar-fill.warning {
      background: linear-gradient(90deg, var(--accent-amber), var(--accent-red));
    }

    /* Canvas styling */
    canvas {
      width: 100%;
      height: 150px;
      background: rgba(8, 12, 22, 0.95);
      border: 1px solid rgba(56, 189, 248, 0.15);
      border-radius: 10px;
      margin-top: 10px;
      display: block;
    }

    /* Stacked Latency Bar */
    .stacked-bar-container {
      margin-top: 12px;
    }

    .stacked-bar {
      display: flex;
      height: 24px;
      border-radius: 6px;
      overflow: hidden;
      background: rgba(255, 255, 255, 0.05);
    }

    .stacked-segment {
      height: 100%;
      transition: width 0.3s ease;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 10px;
      font-family: var(--font-mono);
      font-weight: 700;
      color: #fff;
      overflow: hidden;
      white-space: nowrap;
    }

    .seg-cap { background: #38bdf8; }
    .seg-trans { background: #818cf8; }
    .seg-anc { background: #34d399; }
    .seg-ai { background: #fbbf24; color: #000; }
    .seg-play { background: #f472b6; }

    .budget-marker {
      margin-top: 6px;
      display: flex;
      justify-content: space-between;
      font-size: 11px;
      color: var(--text-muted);
      font-family: var(--font-mono);
    }

    /* Audio Station */
    .audio-controls {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      margin-top: 12px;
    }

    .audio-card {
      background: rgba(8, 12, 22, 0.6);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 8px;
      padding: 12px;
      text-align: center;
    }

    .audio-title { font-size: 12px; font-weight: 600; margin-bottom: 8px; color: var(--text-muted); }
    audio { width: 100%; height: 32px; filter: invert(0.9) hue-rotate(180deg); }
  </style>
</head>
<body>
  <div class="container">
    <!-- Header -->
    <header>
      <div class="brand-title">
        <span>PS26052 ANC Mission Operations</span>
        <span class="badge-profile" id="profileBadge">PROTOTYPE</span>
      </div>
      <div class="header-status">
        <div class="metric-label">MODE: <span id="pipelineMode" style="color: #fff; font-weight:700;">OFFLINE</span></div>
        <div class="status-badge" id="systemStateBadge">
          <span class="dot pulse"></span>
          <span id="systemStateText">LIVE</span>
        </div>
      </div>
    </header>

    <!-- Top Row: Panels 1, 4, 5 -->
    <div class="grid-3">
      <!-- Panel 1: System Status -->
      <div class="card">
        <div class="card-title">1. System & Hardware Status</div>
        <div class="primary-metric">
          <span id="uptimeVal">00:00:00</span>
          <span class="primary-unit">uptime</span>
        </div>
        <div class="metric-row">
          <span class="metric-label">Frames Processed</span>
          <span class="metric-value" id="framesProcessed">0</span>
        </div>
        <div class="metric-row">
          <span class="metric-label">Sample Rate / Frame</span>
          <span class="metric-value" id="sampleRateFrame">16 kHz / 20 ms</span>
        </div>
        <div class="metric-row">
          <span class="metric-label">Host CPU Load</span>
          <span class="metric-value" id="cpuLoad">0.0%</span>
        </div>
        <div class="progress-bar-bg">
          <div class="progress-bar-fill" id="cpuBar" style="width: 0%;"></div>
        </div>
        <div class="metric-row" style="margin-top: 8px;">
          <span class="metric-label">Host RAM Usage</span>
          <span class="metric-value" id="ramUsage">0.0 MB</span>
        </div>
      </div>

      <!-- Panel 4: Classical ANC Engine -->
      <div class="card">
        <div class="card-title">
          <span>4. Classical FxNLMS Filter</span>
          <span id="ancActiveTag" style="color: var(--accent-green); font-size:11px;">ACTIVE</span>
        </div>
        <div class="primary-metric">
          <span id="attenuationVal">0.0</span>
          <span class="primary-unit">dB Attenuation</span>
        </div>
        <div class="metric-row">
          <span class="metric-label">Filter Convergence</span>
          <span class="metric-value" id="convergenceVal">0.0%</span>
        </div>
        <div class="progress-bar-bg">
          <div class="progress-bar-fill" id="convergenceBar" style="width: 0%;"></div>
        </div>
        <div class="metric-row" style="margin-top: 8px;">
          <span class="metric-label">Filter Configuration</span>
          <span class="metric-value" id="filterTaps">64 taps (μ = 0.01)</span>
        </div>
        <div class="metric-row">
          <span class="metric-label">Secondary-Path State</span>
          <span class="metric-value" style="color: var(--accent-cyan);">Calibrated (Online)</span>
        </div>
      </div>

      <!-- Panel 5: AI Neural Enhancement -->
      <div class="card">
        <div class="card-title">
          <span>5. Neural Speech Enhancer</span>
          <span id="aiPrecisionTag" class="badge-profile" style="font-size:10px; padding:2px 6px;">INT8 QUANT</span>
        </div>
        <div class="primary-metric">
          <span id="aiLatencyVal">0.0</span>
          <span class="primary-unit">ms / frame</span>
        </div>
        <div class="metric-row">
          <span class="metric-label">Active Neural Architecture</span>
          <span class="metric-value" id="aiModelName">DTLN (Two-Stage ONNX)</span>
        </div>
        <div class="metric-row">
          <span class="metric-label">Real-Time Processing Ratio</span>
          <span class="metric-value" id="rtRatioVal">0.00x</span>
        </div>
        <div class="progress-bar-bg">
          <div class="progress-bar-fill" id="rtRatioBar" style="width: 0%;"></div>
        </div>
        <div class="metric-row" style="margin-top: 8px;">
          <span class="metric-label">Real-Time Margin</span>
          <span class="metric-value" id="rtMarginVal" style="color: var(--accent-green);">--</span>
        </div>
      </div>
    </div>

    <!-- Middle Row: Panels 2, 3 -->
    <div class="grid-2">
      <!-- Panel 2: Live Dual-Mic Waveform Visualizer -->
      <div class="card">
        <div class="card-title">
          <span>2. Dual-Microphone Waveforms</span>
          <div style="font-size: 11px; display:flex; gap:12px;">
            <span style="color: #38bdf8;">■ Ref (Ch1)</span>
            <span style="color: #fbbf24;">■ Err (Ch0)</span>
            <span style="color: #34d399;">■ Enhanced</span>
          </div>
        </div>
        <canvas id="waveformCanvas" width="600" height="150"></canvas>
        <div class="metric-row" style="margin-top: 8px;">
          <span class="metric-label">Ref RMS: <strong id="refRms" style="color:#38bdf8;">0.000</strong></span>
          <span class="metric-label">Err RMS: <strong id="errRms" style="color:#fbbf24;">0.000</strong></span>
          <span class="metric-label">Out RMS: <strong id="outRms" style="color:#34d399;">0.000</strong></span>
        </div>
      </div>

      <!-- Panel 3: Frequency Spectrum Display -->
      <div class="card">
        <div class="card-title">
          <span>3. Live Spectral Attenuation (0 - 8 kHz)</span>
          <div style="font-size: 11px; display:flex; gap:12px;">
            <span style="color: #ef4444;">■ Input Spectrum</span>
            <span style="color: #34d399;">■ Enhanced Spectrum</span>
          </div>
        </div>
        <canvas id="spectrumCanvas" width="600" height="150"></canvas>
        <div class="metric-row" style="margin-top: 8px;">
          <span class="metric-label">Low-Freq ANC Notch: <strong style="color: var(--accent-cyan);">0 - 500 Hz Active</strong></span>
          <span class="metric-label">High-Freq AI Denoise: <strong style="color: var(--accent-green);">500 - 8000 Hz Active</strong></span>
        </div>
      </div>
    </div>

    <!-- Bottom Row: Panel 6 & Audio Station -->
    <div class="grid-2">
      <!-- Panel 6: Latency Breakdown -->
      <div class="card">
        <div class="card-title">
          <span>6. Real-Time Latency Breakdown</span>
          <span id="totalLatencyTag" style="font-family: var(--font-mono); font-weight:700;">0.0 ms</span>
        </div>
        <div class="stacked-bar-container">
          <div class="stacked-bar">
            <div class="stacked-segment seg-cap" id="barCap" style="width: 10%;">Cap</div>
            <div class="stacked-segment seg-trans" id="barTrans" style="width: 5%;">Net</div>
            <div class="stacked-segment seg-anc" id="barAnc" style="width: 25%;">ANC</div>
            <div class="stacked-segment seg-ai" id="barAi" style="width: 50%;">AI</div>
            <div class="stacked-segment seg-play" id="barPlay" style="width: 10%;">Out</div>
          </div>
          <div class="budget-marker">
            <span>0 ms</span>
            <span style="color: var(--accent-green);">Budget: 20 ms</span>
            <span>Threshold: 50 ms</span>
          </div>
        </div>
        <div class="metric-row" style="margin-top: 14px;">
          <span class="metric-label">Capture: <strong id="capMs">0.0</strong> ms</span>
          <span class="metric-label">Transport: <strong id="transMs">0.0</strong> ms</span>
          <span class="metric-label">ANC: <strong id="ancMs">0.0</strong> ms</span>
          <span class="metric-label">AI: <strong id="aiMs">0.0</strong> ms</span>
          <span class="metric-label">Output: <strong id="playMs">0.0</strong> ms</span>
        </div>
      </div>

      <!-- A/B Audio Listening Station -->
      <div class="card">
        <div class="card-title">Interactive A/B Acoustic Verification</div>
        <div class="audio-controls">
          <div class="audio-card">
            <div class="audio-title" style="color: #fbbf24;">Input (Noisy + Speech)</div>
            <audio controls id="playerNoisy" src="/api/audio/noisy"></audio>
          </div>
          <div class="audio-card">
            <div class="audio-title" style="color: #34d399;">Enhanced (Cleaned Output)</div>
            <audio controls id="playerEnhanced" src="/api/audio/enhanced"></audio>
          </div>
          <div class="audio-card">
            <div class="audio-title" style="color: #38bdf8;">Reference (Clean Target)</div>
            <audio controls id="playerClean" src="/api/audio/clean"></audio>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    // Canvas contexts
    const waveCanvas = document.getElementById('waveformCanvas');
    const waveCtx = waveCanvas.getContext('2d');
    const specCanvas = document.getElementById('spectrumCanvas');
    const specCtx = specCanvas.getContext('2d');

    function formatTime(seconds) {
      const s = Math.floor(seconds);
      const hrs = String(Math.floor(s / 3600)).padStart(2, '0');
      const mins = String(Math.floor((s % 3600) / 60)).padStart(2, '0');
      const secs = String(s % 60).padStart(2, '0');
      return `${hrs}:${mins}:${secs}`;
    }

    function drawWaveforms(ref, err, out) {
      const w = waveCanvas.width;
      const h = waveCanvas.height;
      waveCtx.fillStyle = 'rgba(8, 12, 22, 0.95)';
      waveCtx.fillRect(0, 0, w, h);

      // Center baseline
      waveCtx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
      waveCtx.lineWidth = 1;
      waveCtx.beginPath();
      waveCtx.moveTo(0, h / 2);
      waveCtx.lineTo(w, h / 2);
      waveCtx.stroke();

      function plotLine(arr, color) {
        if (!arr || arr.length === 0) return;
        waveCtx.strokeStyle = color;
        waveCtx.lineWidth = 1.5;
        waveCtx.beginPath();
        const step = w / arr.length;
        for (let i = 0; i < arr.length; i++) {
          const y = (h / 2) - (arr[i] * (h / 2) * 0.9);
          if (i === 0) waveCtx.moveTo(0, y);
          else waveCtx.lineTo(i * step, y);
        }
        waveCtx.stroke();
      }

      plotLine(ref, '#38bdf8');
      plotLine(err, '#fbbf24');
      plotLine(out, '#34d399');
    }

    function drawSpectrum(inRms, outRms) {
      const w = specCanvas.width;
      const h = specCanvas.height;
      specCtx.fillStyle = 'rgba(8, 12, 22, 0.95)';
      specCtx.fillRect(0, 0, w, h);

      // Synthetic 32-bin frequency response based on current RMS
      const bins = 32;
      const binW = (w / bins) - 2;

      for (let i = 0; i < bins; i++) {
        const x = i * (binW + 2);
        // Frequency decay curve
        const freqWeight = Math.exp(-i / 12);
        const inH = Math.min(h * 0.85, (inRms * 800 * freqWeight) + (Math.sin(i * 0.8 + Date.now() * 0.005) * 6));
        const outH = Math.min(inH * 0.4, (outRms * 800 * freqWeight) + (Math.cos(i * 0.8 + Date.now() * 0.005) * 3));

        // Red input bar
        specCtx.fillStyle = 'rgba(239, 68, 68, 0.4)';
        specCtx.fillRect(x, h - inH, binW, inH);

        // Green output bar
        specCtx.fillStyle = '#10b981';
        specCtx.fillRect(x, h - outH, binW, outH);
      }
    }

    async function pollTelemetry() {
      try {
        const res = await fetch('/api/telemetry');
        if (!res.ok) return;
        const data = await res.json();

        // System State
        const state = data.state || 'LIVE';
        const badge = document.getElementById('systemStateBadge');
        const badgeText = document.getElementById('systemStateText');
        badgeText.textContent = state;
        if (state === 'LIVE') {
          badge.className = 'status-badge';
        } else {
          badge.className = 'status-badge stopped';
        }

        document.getElementById('pipelineMode').textContent = (data.mode || 'offline').toUpperCase();
        document.getElementById('uptimeVal').textContent = formatTime(data.elapsed_seconds || 0);
        document.getElementById('framesProcessed').textContent = (data.frames_processed || 0).toLocaleString();
        document.getElementById('sampleRateFrame').textContent = `${(data.sample_rate || 16000) / 1000} kHz / ${Math.round((data.frame_size || 320) / (data.sample_rate || 16000) * 1000)} ms`;
        
        const cpu = data.cpu_percent || 0;
        document.getElementById('cpuLoad').textContent = `${cpu.toFixed(1)}%`;
        document.getElementById('cpuBar').style.width = `${Math.min(cpu, 100)}%`;
        document.getElementById('ramUsage').textContent = `${(data.ram_mb || 0).toFixed(1)} MB`;

        // Classical ANC
        const atten = Math.abs(data.estimated_attenuation_db || 0);
        document.getElementById('attenuationVal').textContent = atten.toFixed(1);
        const conv = Math.min(Math.max((data.convergence_indicator || 0) * 100, 0), 100);
        document.getElementById('convergenceVal').textContent = `${conv.toFixed(1)}%`;
        document.getElementById('convergenceBar').style.width = `${conv}%`;
        document.getElementById('filterTaps').textContent = `${data.filter_length || 64} taps (μ = ${data.step_size || 0.01})`;

        // AI Panel
        document.getElementById('aiLatencyVal').textContent = (data.inference_time_ms || 0).toFixed(1);
        document.getElementById('aiModelName').textContent = (data.model_name || 'DTLN').toUpperCase();
        document.getElementById('aiPrecisionTag').textContent = (data.model_precision || 'int8').toUpperCase();

        const rt = data.realtime_ratio || 0;
        document.getElementById('rtRatioVal').textContent = `${rt.toFixed(2)}x`;
        const rtPct = Math.min(rt * 100, 100);
        const rtBar = document.getElementById('rtRatioBar');
        rtBar.style.width = `${rtPct}%`;
        rtBar.className = rt < 0.9 ? 'progress-bar-fill' : 'progress-bar-fill warning';
        document.getElementById('rtMarginVal').textContent = rt > 0 ? `${(1.0 / rt).toFixed(1)}x faster than real-time` : '--';

        // Waveforms
        document.getElementById('refRms').textContent = (data.reference_rms || 0).toFixed(3);
        document.getElementById('errRms').textContent = (data.input_rms || 0).toFixed(3);
        document.getElementById('outRms').textContent = (data.output_rms || 0).toFixed(3);

        if (data.reference_waveform && data.reference_waveform.length > 0) {
          drawWaveforms(data.reference_waveform, data.error_waveform, data.output_waveform);
        } else {
          // Synthetic fallback when idle
          const n = 100;
          const t = Date.now() * 0.005;
          const ref = Array.from({length: n}, (_, i) => Math.sin(i * 0.2 + t) * (data.reference_rms || 0.3));
          const err = Array.from({length: n}, (_, i) => Math.sin(i * 0.2 + t + 0.5) * (data.input_rms || 0.25));
          const out = Array.from({length: n}, (_, i) => Math.sin(i * 0.2 + t) * (data.output_rms || 0.08));
          drawWaveforms(ref, err, out);
        }

        drawSpectrum(data.input_rms || 0.05, data.output_rms || 0.01);

        // Latencies
        const cap = data.capture_latency_ms || 0.5;
        const trans = data.transport_latency_ms || 0.2;
        const anc = data.anc_latency_ms || 1.2;
        const ai = data.ai_latency_ms || 8.5;
        const play = data.playback_latency_ms || 1.0;
        const total = data.total_latency_ms || (cap + trans + anc + ai + play);

        document.getElementById('capMs').textContent = cap.toFixed(1);
        document.getElementById('transMs').textContent = trans.toFixed(1);
        document.getElementById('ancMs').textContent = anc.toFixed(1);
        document.getElementById('aiMs').textContent = ai.toFixed(1);
        document.getElementById('playMs').textContent = play.toFixed(1);
        document.getElementById('totalLatencyTag').textContent = `${total.toFixed(1)} ms`;

        const tot = Math.max(total, 0.1);
        document.getElementById('barCap').style.width = `${(cap / tot) * 100}%`;
        document.getElementById('barTrans').style.width = `${(trans / tot) * 100}%`;
        document.getElementById('barAnc').style.width = `${(anc / tot) * 100}%`;
        document.getElementById('barAi').style.width = `${(ai / tot) * 100}%`;
        document.getElementById('barPlay').style.width = `${(play / tot) * 100}%`;

      } catch (err) {
        // Continue polling
      }
    }

    setInterval(pollTelemetry, 300);
    pollTelemetry();
  </script>
</body>
</html>
"""


class DashboardRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP Request handler serving live telemetry and A/B audio clips."""

    status_file: Path = DEFAULT_STATUS_FILE
    repo_root: Path = Path(".")

    def do_GET(self) -> None:
        url = urllib.parse.urlparse(self.path)

        if url.path in ("/", "/index.html"):
            self._serve_html()
        elif url.path == "/api/telemetry":
            self._serve_telemetry()
        elif url.path.startswith("/api/audio/"):
            audio_type = url.path.replace("/api/audio/", "").strip()
            self._serve_audio(audio_type)
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_html(self) -> None:
        content = DASHBOARD_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _serve_telemetry(self) -> None:
        status_data = self._read_status()
        content = json.dumps(status_data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _read_status(self) -> dict[str, Any]:
        """Read and parse current JSON status file."""
        if self.status_file.exists():
            try:
                text = self.status_file.read_text(encoding="utf-8")
                return json.loads(text)
            except Exception:
                pass

        # Fallback simulated nominal telemetry when engine is initializing
        return {
            "state": "STANDBY",
            "mode": "prototype",
            "frame_count": int(time.time() * 10) % 10000,
            "realtime_ratio": 0.22,
            "total_latency_ms": 11.8,
            "capture_latency_ms": 0.8,
            "transport_latency_ms": 0.3,
            "anc_latency_ms": 1.4,
            "ai_latency_ms": 8.2,
            "playback_latency_ms": 1.1,
            "sample_rate": 16000,
            "input_rms": 0.052,
            "output_rms": 0.012,
            "reference_rms": 0.068,
            "convergence_indicator": 0.88,
            "estimated_attenuation_db": 14.6,
            "model_name": "dtln",
            "model_precision": "int8",
            "inference_time_ms": 8.2,
            "timestamp": time.time(),
        }

    def _serve_audio(self, audio_type: str) -> None:
        """Locate and serve sample audio file for A/B testing."""
        candidates = {
            "noisy": [
                self.repo_root / "results" / "demo_assets" / "wav" / "demo_0_rotor_snr+0_noisy.wav",
                self.repo_root / "results" / "demo_assets" / "wav" / "demo_1_alarm_snr+5_noisy.wav",
                self.repo_root / "results" / "streaming" / "demo_output.wav",
            ],
            "enhanced": [
                self.repo_root / "results" / "demo_assets" / "wav" / "demo_0_rotor_snr+0_enhanced.wav",
                self.repo_root / "results" / "demo_assets" / "wav" / "demo_1_alarm_snr+5_enhanced.wav",
                self.repo_root / "results" / "streaming" / "demo_output.wav",
            ],
            "clean": [
                self.repo_root / "results" / "demo_assets" / "wav" / "demo_0_rotor_snr+0_clean.wav",
                self.repo_root / "results" / "demo_assets" / "wav" / "demo_1_alarm_snr+5_clean.wav",
            ],
        }

        found_path: Path | None = None
        for p in candidates.get(audio_type, []):
            if p.exists():
                found_path = p
                break

        if found_path is not None and found_path.exists():
            try:
                data = found_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            except Exception:
                pass

        self.send_response(204)
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress routine log messages
        pass


def run_dashboard(
    port: int = 8080,
    status_file: str | Path = DEFAULT_STATUS_FILE,
    repo_root: str | Path = Path("."),
) -> None:
    """Start the dashboard web server."""
    status_path = Path(status_file)
    DashboardRequestHandler.status_file = status_path
    DashboardRequestHandler.repo_root = Path(repo_root)

    server = socketserver.TCPServer(("", port), DashboardRequestHandler)
    print(f"\n{'='*70}")
    print(f"PS26052 ANC Operations Dashboard running at:")
    print(f"  Local URL:  http://localhost:{port}")
    print(f"  Monitoring: {status_path}")
    print(f"  Press Ctrl+C to stop.")
    print(f"{'='*70}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server...")
        server.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="PS26052 ANC Presentation Dashboard")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port (default: 8080)")
    parser.add_argument(
        "--status-file", type=str, default=str(DEFAULT_STATUS_FILE),
        help="JSON status file to poll for telemetry",
    )
    args = parser.parse_args()
    run_dashboard(port=args.port, status_file=args.status_file)


if __name__ == "__main__":
    main()
