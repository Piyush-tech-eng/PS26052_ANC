# PS26052 — AI-Driven Adaptive Noise Cancellation (ANC) for Defence Communications

An end-to-end, real-time hybrid noise cancellation platform combining classical adaptive filtering (FxNLMS) and deep neural network speech enhancement (dual-stage DTLN & Conv-TasNet), targeting defence communications under nonstationary acoustic disturbance.

---

## System Architecture

```
┌──────────────────────────────────────┐          ┌────────────────────────────────────────────────────────┐
│  Edge Capture Node (Raspberry Pi 3)  │   UDP    │  Compute & Processing Node (Laptop)                    │
│  + ReSpeaker 2-Mic Array HAT         │  (PCM)   │                                                        │
│                                      │─────────▶│  UDPReceiver (Jitter Buffer, seq reordering)           │
│  pi/capture_stream.py:               │  Eth/WiFi│    │                                                   │
│   - Dual-channel ALSA capture        │          │    ▼                                                   │
│   - Reference (mic 0) + Error (mic 1)│          │  HybridEngine                                          │
│   - 16 kHz, 16-bit PCM UDP stream    │          │   ├── Stage 1: FrameANC (Streaming FxNLMS)             │
│   - Zero on-Pi ML (latency budget)   │          │   └── Stage 2: OverlapAddProcessor                     │
│                                      │          │                └── Model (DTLN INT8 / Conv-TasNet)     │
│                                      │          │    │                                                   │
│                                      │          │    ▼                                                   │
│                                      │          │  Audio Playback (SoundDevice / Local Speaker)          │
│                                      │          │  LatencyMonitor (Percentiles & Real-Time Ratio)        │
└──────────────────────────────────────┘          └────────────────────────────────────────────────────────┘
```

### Architecture Note: Computational Noise Suppression vs. Physical Acoustic ANC
> **Scope & Engineering Clarification**:
> - **Operational Goal**: This system provides **computational noise suppression for a tactical voice communications link**, rather than closed-loop physical active noise cancellation (acoustic wave interference in open air).
> - **Input Channels**: "Reference mic" and "error mic" denote two synchronous input channels consumed by an adaptive FxNLMS algorithm, rather than an analog loudspeaker feedback loop emitting anti-noise into an acoustic space.
> - **Acoustic Latency Realities**: Physical acoustic cancellation demands sub-millisecond phase alignment ($< 0.5\text{ ms}$). Neural speech enhancement networks (DTLN/Conv-TasNet), STFT analysis windows, and jitter buffers naturally operate at $20\text{–}40\text{ ms}$ latency—completely incompatible with open-air destructive interference, but optimal for high-intelligibility voice reception across a communication link.

### Key Technical Differentiators
1. **True Hybrid Cascade**: Classical FxNLMS handles correlated, stationary noise and secondary path acoustics with sub-millisecond algorithmic delay; the deep speech enhancement model removes nonstationary, impulsive, and diffuse defence noise (artillery, rotor, engine, sirens).
2. **Proven Generalization**: Rigid speaker-isolated splits and held-out noise-family reservation guarantee models generalize to unseen speakers and acoustic environments without memorization.
3. **Rigorous Hardware-Agnostic Engine**: Interface-driven architecture (`EnhancementModel`) allows seamless switching between fine-tuned DTLN, INT8 quantized ONNX, Conv-TasNet, and classical baselines without changing streaming code.
4. **Network Jitter Resilience**: Dynamic UDP jitter buffer with out-of-order sequence reassembly and frame interpolation protects against wireless network packet loss without stream stalls.

---

## Project Structure

```
PS26052_ANC/
├── configs/                     # System configuration YAMLs
├── data/
│   ├── corpora/                 # Downloaded training corpora (LibriSpeech, VCTK, MUSAN, ESC-50)
│   └── download_corpora.py      # Automated, idempotent dataset downloader
├── docs/
│   └── final_report.md          # Comprehensive technical report and empirical validation
├── experiments/                 # Reproducible experiment runners
│   ├── m7_02_speech_dataset_generation.py    # Speech+noise mixing with RIR augmentation
│   ├── m7_04_full_benchmark_matrix.py        # Benchmark across all methods & noise families
│   ├── m8_01_train_enhancement_model.py      # PyTorch training entry point
│   ├── m8_02_ablation_study.py               # Loss term ablation study (SI-SNR, L1, STFT)
│   ├── m9_01_heldout_generalization_report.py# Held-out speaker/noise evaluation
│   └── m10_01_robustness_sweep.py            # Dynamic conditions sweep (onset, SNR jumps)
├── models/                      # Pretrained and fine-tuned ONNX/PyTorch models
├── pi/
│   └── capture_stream.py        # Lightweight ALSA capture & UDP streaming daemon for Pi 3
├── src/
│   ├── ai/                      # Machine learning and streaming engine
│   │   ├── export/              # ONNX export and INT8/FP16 quantization tooling
│   │   ├── hardware/            # UDP receiver, jitter buffer, latency monitor
│   │   ├── models/              # DTLN (trainable & ONNX), Conv-TasNet, RNNoise, SpectralGate
│   │   ├── streaming/           # HybridEngine, FrameANC, OverlapAddProcessor
│   │   └── training/            # PyTorch Trainer, DataLoader, Hyperparameter Search
│   └── anc/                     # Classical ANC and DSP foundations
│       ├── adaptive/            # LMS, NLMS, Adaptive FIR filters
│       ├── evaluation/          # Metrics (SI-SNR, STOI, PESQ), Scenario Matrix, Dataset splits
│       ├── plant/               # Acoustic plant simulation and secondary path modeling
│       └── speech/              # Audio loaders, mixing, RIR augmentation, manifest generator
├── tests/                       # Complete pytest suite (342 passing tests)
├── pyproject.toml               # Package configuration with optional dependency groups
└── run_live_demo.py             # Live demonstration CLI (Simulated, Loopback, & Pi Modes)
```

---

## Installation & Setup

### Prerequisites
- Python 3.10 to 3.13
- Virtual environment recommended:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate
```

### Install Dependencies
Dependencies are split into targeted optional groups:

```bash
# Core classical DSP and evaluation:
pip install -e .

# With PyTorch model training support:
pip install -e ".[training]"

# With ONNX export & quantization tools:
pip install -e ".[export]"

# With live hardware audio I/O (PortAudio):
pip install -e ".[hardware]"

# Complete development environment:
pip install -e ".[full,test]"
```

> **Note for Raspberry Pi / Linux users**: Install system PortAudio headers before installing PyAudio:
> ```bash
> sudo apt-get update && sudo apt-get install -y portaudio19-dev
> ```

---

## Unified Command Center & Deliverable (All-in-One Interface)

The project includes an all-in-one interactive web interface packing the complete noise cancellation pipeline into a single deliverable. It provides defence noise preset selection, custom WAV/MP3 upload, live in-browser microphone recording, dual time-frequency spectrogram visualization (input, enhanced, and attenuation heatmap), and an instant A/B audio player.

### Quick Start (Single Command)

```bash
# Windows / Linux / macOS:
python app.py
```

*Or on Windows, double-click* `start_app.bat`.

- Automatically opens **`http://localhost:8080`** in your browser.
- Select from 36 curated defence noise scenarios (Rotor, Engine, Impulsive, Wind, Alarm).
- Or upload your own `.wav` file, or click **Start Recording** to test your live microphone.
- View high-resolution spectrograms (0–8 kHz) and click **Instant A/B** to hear immediate noise suppression.

---

## Running the Live CLI Demonstrations

The repository includes a self-contained live runner `run_live_demo.py` supporting three operating modes:

### 1. Simulated Audio Mode (No Hardware Required)
Runs simulated multi-tone and helicopter noise through the hybrid pipeline with real-time audio playback:
```bash
python run_live_demo.py --mode simulated --duration 10.0
```

### 2. Local Loopback Mode (Audio Hardware / Laptop Mic)
Captures live audio from your default laptop microphone and runs real-time ANC + AI enhancement:
```bash
python run_live_demo.py --mode loopback --duration 15.0
```

### 3. Live Hardware Mode (Raspberry Pi 3 + Laptop)
**Step A (On Raspberry Pi 3)**:
```bash
# Install ALSA utilities & run capture stream
python pi/capture_stream.py --host <LAPTOP_IP> --port 5005 --channels 2 --sample-rate 16000
```

**Step B (On Laptop)**:
```bash
python run_live_demo.py --mode live --port 5005
```

---

## Training & Evaluation Workflows

### 1. Downloading Training Corpora
Idempotently download LibriSpeech, VCTK, MUSAN, and ESC-50 datasets:
```bash
python data/download_corpora.py --all
```

### 2. Generating the Multi-Speaker Augmented Dataset
Creates dataset splits with RIR reverberation augmentation and provenance tracking:
```bash
python experiments/m7_02_speech_dataset_generation.py
```

### 3. Model Training & Fine-Tuning
Train the dual-stage DTLN or from-scratch Conv-TasNet:
```bash
# Train DTLN:
python experiments/m8_01_train_enhancement_model.py --model dtln --epochs 50 --batch-size 8

# Train Conv-TasNet:
python experiments/m8_01_train_enhancement_model.py --model conv_tasnet --epochs 50 --batch-size 8
```

### 4. Running the Ablation Study
Evaluate combinations of loss terms (SI-SNR, L1, multi-resolution STFT):
```bash
python experiments/m8_02_ablation_study.py --epochs 20
```

### 5. Running the Full Benchmark Matrix
Evaluates all algorithms (No-ANC, Wiener, NLMS, FxNLMS, AI-Only, Hybrid) across all noise families:
```bash
python experiments/m7_04_full_benchmark_matrix.py
```

### 6. Generalization & Robustness Verification
```bash
# Held-out speaker and noise family evaluation:
python experiments/m9_01_heldout_generalization_report.py --dataset results/m7_speech_ai_handoff

# Dynamic condition sweep (abrupt onset, mid-clip SNR jump):
python experiments/m10_01_robustness_sweep.py
```

---

## Verification & Testing

Execute the comprehensive test suite (342 unit and integration tests):
```bash
pytest tests/ -v
```

---

## Phase 3 — Dashboard Rebuild & UI Enhancements

The web interface (`src/anc/interface/`) received a comprehensive visual and logic overhaul. All backend APIs remain unchanged.

### Running the Dashboard (Updated)

The preferred entry point is the dedicated server module:

```bash
# Recommended — custom port, no auto-browser:
python src/anc/interface/server.py --port 8080 --no-browser

# Open browser automatically on launch:
python src/anc/interface/server.py --port 8080

# Legacy shorthand (wraps the above):
python app.py
```

Then open **`http://localhost:8080`** in your browser.

> **Tip**: On first visit the interface auto-detects your OS dark/light preference.  
> Click the ☀️/🌙 button in the top-right header to override it at any time — the choice is remembered across sessions in `localStorage`.

---

### What changed

#### Design System
- **Token-driven CSS** — all colors, spacing, shadows, and typography use CSS custom properties in `:root`; zero hardcoded hex values in component rules.
- **Font scale bumped** — `--text-xs` through `--text-3xl` increased by ~0.07–0.1 rem for comfortable readability across all cards and panels.
- **Semantic signal palette** — `--color-signal-noisy` (orange-600) for raw/input signal, `--color-signal-enhanced` (cyan-600) for processed output; applied consistently to waveform datasets, spectrogram labels, and legend dots.
- **WCAG AA contrast** — `--color-text-tertiary` raised from `#94a3b8` → `#64748b` (achieves 4.5:1 on white backgrounds).

#### Dark Mode
- Full dark theme via `[data-theme="dark"]` on `<html>` — covers all surfaces, text, borders, shadows, and status badges.
- Persisted in `localStorage`; respects the OS `prefers-color-scheme` on first visit.
- ☀️ / 🌙 toggle button in the top-right of the header.

#### Header / Navbar
- Replaced the plain long-text header with a proper branded header:
  - Mini **radar SVG logo mark** (concentric rings + sweep line).
  - Bold **"ANC Defence Suite"** app name + muted subtitle line.
  - Dynamic **system status badge** cycling through: `System Ready → Uploading → Processing Signal... → Enhancement Complete` with distinct colors and pulse animations per state.
- **Sidebar step-state indicators** — `●` (current) / `✓` (completed) / `○` (pending) badges on each nav item, updated automatically as the user progresses through Upload → Results → Analysis.

#### Upload Screen
- **Format hint** moved below the dropzone as plain muted text instead of pills inside the upload area.
- Hero section rebuilt as an animated **radar SVG**: three staggered rings pulse/radiate outward from the centre, a rotating sweep line with echo arc trails, and blinking blip targets at random positions.

#### Results Screen
- **Unified score cards** — "Performance Scores" and "Target Benchmarks" merged into one full-width card per metric (SNR / STOI / PESQ). Each card shows the actual value, inline target, and a ✓/✕ pass-fail icon for colorblind-accessible feedback.
- **`Est. SNR` label** — when no clean reference is available, the SNR label reads "Est. SNR" with a tooltip clarifying it is reference-free.
- **Real-time ratio fix** — correctly computed as `processing_time_ms / (audio_duration_s × 1000)`; only labelled "(real-time capable)" when ratio < 1.0, otherwise "(offline only)".
- **Numeric precision standardized** — dB/ratio values always 1 decimal place; ms values rounded to integer (< 10 ms shows 1 decimal).
- **`Enhancement Results` `<h1>`** heading added at the top of the Results view.

#### Analysis Screen
- **Renamed tabs** — removed opaque module numbering: now "Adaptive Filtering", "ANC Plant Dynamics", "System Identification".
- **`Deep Signal Analysis` `<h1>`** heading added at the top of the Analysis view.
- **Chart readability** — axis title font 11→13 px, tick font 10→12 px, legend font 11→12 px across all Chart.js instances.
- **Audio player** `min-height: 44px` for WCAG 2.1 touch-target compliance.

---

### Interface capabilities at a glance

| Feature | Details |
|---|---|
| **Defence presets** | 36 curated scenarios — Helicopter, Armoured Vehicle, Artillery, Wind, Alarm, etc. |
| **Custom upload** | WAV · MP3 · FLAC · AAC · OGG · WMA up to 50 MB |
| **Live microphone** | In-browser `MediaRecorder` → base64 pipeline |
| **Pipeline modes** | Hybrid (FxNLMS + DTLN), AI-Only (Neural DTLN), Classical ANC-Only (FxNLMS) |
| **Neural backends** | DTLN FP32 ONNX, DTLN INT8 Quantized, Spectral Gating baseline |
| **Spectrograms** | 0–8 kHz input + enhanced comparison images |
| **Waveform / Spectrum** | Chart.js overlay comparisons (noisy input vs enhanced output) |
| **Module analytics** | Adaptive Filtering convergence, ANC Plant dynamics, Secondary Path System ID |
| **A/B audio player** | Toggle between enhanced and original audio in-browser |
| **Dark mode** | Full token-driven dark theme, persisted across sessions |
