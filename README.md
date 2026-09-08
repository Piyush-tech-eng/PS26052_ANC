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

## Running the Live Demonstration

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
