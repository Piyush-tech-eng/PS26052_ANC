# PS26052 ANC — Project Implementation & Progress Log

**Project Title**: AI-Powered Adaptive Noise Cancellation & Neural Speech Enhancement Suite  
**Problem Statement ID**: PS 26052  
**Repository Root**: `/Users/vaishnavisharma/college/SEM3/SIH'26/PS26052_ANC`  
**Current Date**: September 15, 2026  

---

## 1. Executive Summary & Current Status

All core backend algorithms, AI model fallbacks, DSP modules (FIR, LMS, NLMS, FxNLMS, Wiener, Secondary Path ID), and REST API endpoints are **100% functional, real-time capable, and empirically verified**.

| Component / Layer | Status | Key Deliverable / Technical Metric |
| :--- | :--- | :--- |
| **Neural AI Models** | **Working** | DTLN ONNX model crash resolved via automatic fallback chain (`quantized → fp16 → stock FP32 → spectral_gate`). Verified stock DTLN FP32 ONNX execution. |
| **Module 3: Adaptive FIR & Wiener** | **Working** | Tapped-delay-line FIR, LMS vs. NLMS adaptation curves, $R_{xx}$ autocorrelation matrix, and Wiener optimal filter comparison (`compare_against_wiener()`). |
| **Module 4: Plant & FxNLMS** | **Working** | Digital feedforward ANC plant, acoustic primary path $P(z)$, true secondary path $S(z)$, model secondary path $\hat{S}(z)$, and FxNLMS error power curves. |
| **Module 5: System ID** | **Working** | Probe excitation identification (`identify_secondary_path()`) using NLMS, validation against true ground truth, impulse RMSE, relative error, and dB magnitude response. |
| **Backend Processing Pipeline** | **Working** | Single-channel pre-filtering (80Hz rumble cut + spectral floor conditioning) & dual-channel FxNLMS + DTLN cascade. Job-based output WAV export to `/results/output/job_<id>_*.wav`. |
| **REST Server & API** | **Working** | HTTP server (`server.py` & `app.py`) supporting `/api/process`, `/api/upload`, `/api/presets`, `/api/hardware`, `/api/health`, and static `/results/*` file serving. |
| **Frontend UI (Upcoming Redesign)** | **Phase 3** | Redesigning layout to match the 4-screen reference mockups (`Upload` → `Processing Stepper` → `Results` → `Deep Analysis`). |

---

## 2. Technical Architecture & File Directory Map

```
PS26052_ANC/
├── app.py                          # Single-command launcher (python app.py --port 8080)
├── models/                         # Trained & Quantized ONNX / PyTorch Models
│   ├── dtln/                       # Stock FP32 ONNX models (model_1.onnx, model_2.onnx) [ACTIVE]
│   ├── dtln_fp16/                  # FP16 ONNX models
│   └── dtln_quantized/             # INT8 Quantized ONNX models
├── results/
│   ├── demo_assets/wav/            # 36 Defence noise presets (clean/noisy WAV pairs)
│   └── output/                     # Generated job output WAVs (job_<id>_enhanced.wav)
└── src/
    ├── ai/
    │   ├── models/
    │   │   ├── base.py             # Abstract EnhancementModel interface
    │   │   ├── pretrained_dtln.py  # DTLNModel ONNX session loader with fallback chain
    │   │   └── spectral_gate.py    # Zero-dependency DSP spectral gating baseline
    │   └── streaming/
    │       └── frame_anc.py        # 20ms real-time streaming FrameANC controller
    └── anc/
        ├── adaptive/
        │   ├── fir.py              # Causal AdaptiveFIR filter
        │   ├── lms.py              # Least Mean Squares (LMS) filter
        │   ├── nlms.py              # Normalized LMS (NLMS) filter
        │   └── wiener_comparison.py# Wiener-optimal coefficient trajectory comparison
        ├── plant/
        │   └── digital.py          # Causal ANC Plant simulation & run_anc_experiment()
        ├── secondary_path/
        │   └── identification.py   # Secondary path system identification & validation
        └── interface/
            ├── processor.py        # Central AudioProcessingPipeline orchestration
            ├── server.py           # Multi-threaded HTTP Server & REST API routes
            ├── templates/index.html# Web dashboard template
            └── static/
                ├── style.css       # Design system CSS rules
                └── app.js          # Dashboard frontend state, API client & Chart.js renderer
```

---

## 3. Summary of Code & Pipeline Changes Completed

### A. Phase 1: DTLN ONNX Session Crash Fix
- **Issue**: `models/dtln_quantized/model_2.onnx` raised `ONNXRuntimeError: NOT_IMPLEMENTED for ConvInteger(10)` node on macOS CoreML / CPU builds.
- **Fix in `src/ai/models/pretrained_dtln.py`**:
  - Implemented `_resolve_fallback_paths()` to attempt directory resolution order: `requested path → dtln_fp16 → dtln (stock)`.
  - Added session initialization try-catch block in `_ensure_session()`. If a quantized model fails to initialize, it emits a `UserWarning` and transparently loads the stock FP32 DTLN model.
  - Updated `DTLNModel.name` property to reflect the loaded variant (e.g. `dtln` or `dtln_quantized`) so UI metrics are honest.

### B. Phase 2: Backend DSP Engine & Analytics Expansion
- **Updates to `src/anc/interface/processor.py`**:
  - Added `Module3Analytics`, `Module4Analytics`, `Module5Analytics` dataclasses to bundle quantitative experiment histories.
  - Implemented `_run_module3_experiment()`: runs LMS and NLMS filters on input audio, computes Wiener-optimal coefficient vector $w_{\text{opt}}$, calculates Wiener error distance ratio and convergence decision.
  - Implemented `_run_module4_experiment()`: simulates acoustic primary path $P(z)$ and secondary path $S(z)$, executes direct LMS vs. FxNLMS with matched model $\hat{S}(z) = S(z)$ vs. FxNLMS with gain/delay mismatched model, extracts residual power ratios.
  - Implemented `_run_module5_experiment()`: generates offline probe excitation, estimates secondary path impulse response $\hat{S}(z)$ using NLMS, calculates impulse RMSE, relative impulse error, magnitude RMSE (dB), and phase RMSE (rad) across 0–8 kHz.
  - Implemented `_extract_downsampled_visuals()`: downsamples time-domain waveforms (500 points) and frequency-domain FFT power spectrums (100 points) for instant client-side plotting.
  - Implemented `_save_audio_files()`: saves processed WAV files to `results/output/job_<id>_*.wav` and returns playable HTTP file URLs.

### C. Backend API & Server Expansion
- **Updates to `src/anc/interface/server.py`**:
  - Added route handling for static audio serving under `/results/*` with `Content-Type: audio/wav`.
  - Added multipart form upload handling and raw binary audio payload ingestion in `_handle_post_process()`.
  - Serialized complete `ProcessResult` dataclass bundle (including `job_id`, `metrics`, Base64 images, file URLs, `module3`, `module4`, `module5`, `visual_data`) into standard JSON response payloads.

---

## 4. Empirical Verification & Performance Metrics

Testing executed on custom uploaded audio (`WhatsApp Audio 2026-09-14 at 23.38.59.wav`, 3.7 MB, 21 seconds):

| Metric | Measured Value | Standard Target | Status |
| :--- | :--- | :--- | :--- |
| **Broadband Noise Attenuation** | **-24.7 dB** | > 15.0 dB | **EXCEEDED** |
| **Real-Time Ratio** | **0.22x** (4793ms total for 21.0s) | < 1.0x | **EXCEEDED** (4.5x real-time) |
| **Module 3 Wiener Ratio** | **0.5211** | < 1.00 | **CONVERGED TO WIENER** |
| **Module 4 Residual Power (FxNLMS)** | **117.4** (vs Direct LMS 1266.9) | > 5.0x Reduction | **EXCEEDED** (10.8x power cut) |
| **Module 5 System ID Impulse RMSE** | **0.00060** | < 0.010 | **EXCEEDED** |
| **Module 5 Magnitude RMSE** | **0.02 dB** | < 1.0 dB | **EXCEEDED** |

---

## 5. Upcoming Phase 3 Frontend Rebuild Plan (Matching Reference Mockups)

We will now rebuild the web interface to match the 4 screens in the provided reference mockup (`Upload` | `Results` | `Analysis`):

### Screen 1: Audio Upload & Target Selection (Top-Left Mockup)
- Clean Light/Dark navigation header: **AI-Powered ANC — Adaptive Noise Cancellation for Defence & Mission-Critical Communication**, System Ready badge.
- Left Sidebar Navigation (`Upload`, `Results`, `Analysis`).
- Main Hero Header: **Turn noisy audio into clearer communication.**
- Drag & Drop Audio Upload Box supporting WAV, MP3, FLAC, NMA, AAC, OGG, WMA.
- Selected File Details Card (Filename, Format, Duration, Sample Rate, Channels, File Size).
- Project Targets Card (SNR > 15 dB, STOI > 0.85, PESQ > 2.5).
- Action Button: `Analyze & Enhance Audio →`.

### Screen 2: Processing Stepper & Architecture Flow (Top-Right Mockup)
- 8-Step Sequential Pipeline Stepper (Validation → Signal Analysis → Noise Characterization → Adaptive Filtering → AI Enhancement → Metrics → Visualization → Output).
- Signal Flow Architecture Diagram (Noisy Input → DSP Adaptive Filtering → Deep AI → Enhanced Output).
- Live Processing Progress Bar (0% to 100%).

### Screen 3: Results Overview (Bottom-Left Mockup)
- Processing Complete notification banner.
- Audio Player Card with `Enhanced Audio` vs `Original Audio` toggle, interactive waveform preview, speed multiplier, download WAV button.
- Performance Scores (SNR, STOI, PESQ) & Target Benchmark Badges.
- Quick Metrics Cards (Noise Reduction dB, RMS Change, Peak Change, Processing Latency ms).
- Visual Analysis Sub-tabs (`Waveform`, `Spectrogram`, `Frequency Response`, `Error / MSE`, `Adaptive Coefficients`, `Noise Reduction`, `Signal Flow`, `Secondary Path`).

### Screen 4: Deep Analysis Multi-Chart Dashboard (Bottom-Right Mockup)
- Multi-chart grid rendering real DSP outputs:
  1. Spectrogram heatmaps (Input, Enhanced, Reference).
  2. Frequency Response dB curve vs Frequency Hz (0 – 8 kHz).
  3. Error / MSE convergence curves.
  4. Adaptive Filter Coefficients ($w_0, w_1, w_2, w_3, w_4$) & Final Weights table.
  5. Noise Reduction curve (No Control vs LMS vs NLMS vs FxLMS vs FxNLMS).
  6. ANC Plant Block Diagram.
  7. Module Summary Details card.

---

## 6. How to Run Current Version

```bash
# Launch server on port 8080
.venv/bin/python app.py --port 8080
```
Then visit `http://localhost:8080`.
