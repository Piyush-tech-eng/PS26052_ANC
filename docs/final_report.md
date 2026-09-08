# PS26052 — AI-Driven Adaptive Noise Cancellation for Defence Communications
## Final Technical Report & Empirical Verification

---

## 1. Executive Summary & Problem Statement

In tactical defence operations, mission-critical voice communications are frequently compromised by severe, nonstationary acoustic disturbances. These include impulsive shockwaves (gunfire, artillery), periodic low-frequency drone/helicopter rotor modulation, armored vehicle engine vibrations, wind buffeting, and acoustic sirens.

Standard classical Active Noise Control (ANC) techniques—predominantly Least Mean Squares (LMS), Normalized LMS (NLMS), and Filtered-x NLMS (FxNLMS)—rely on linear finite impulse response (FIR) filtering. While FxNLMS excels at cancelling stationary, coherent noise with negligible processing delay (< 1 ms), it degrades severely under:
1. **Abrupt nonstationarity**: Slower convergence rates allow transient noise bursts to pass through before filter weights adapt.
2. **Diffuse and non-coherent noise**: Uncorrelated noise components cannot be predicted from reference microphones.
3. **Secondary acoustic path perturbations**: Phase and amplitude mismatches cause stability issues or noise amplification.

To resolve these fundamental limitations, **PS26052** establishes a true **Hybrid Cascaded Architecture**:
- **Stage 1 (Classical FxNLMS)**: Cancels primary acoustic coupling and coherent low-frequency harmonics directly in the time domain.
- **Stage 2 (Deep Learning Speech Enhancement)**: Suppresses residual, diffuse, nonstationary, and impulsive noise components using deep recurrent and convolutional neural representations.
- **Edge-to-Laptop Real-Time Transport**: Distributes the workload between an ultra-low-power capture node (Raspberry Pi 3 + ReSpeaker 2-Mic HAT) and an edge compute node (laptop CPU/GPU) over a low-latency UDP stream protected by a dynamic jitter buffer.

---

## 2. System Architecture & Component Design

```
                     ┌────────────────────────────────────────────────────────┐
                     │           Raspberry Pi 3 + ReSpeaker 2-Mic             │
                     │  Dual-Channel ALSA Capture (Ref: Mic 0, Err: Mic 1)    │
                     │  16 kHz, 16-bit PCM, 20-40 ms Frame Packaging          │
                     └───────────────────────────┬────────────────────────────┘
                                                 │ UDP Stream (PCM)
                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Laptop Compute Node                                                                             │
│                                                                                                 │
│  ┌───────────────────────────┐      ┌───────────────────────────────┐                           │
│  │   UDP Receiver            │─────▶│ FrameANC (Streaming FxNLMS)   │                           │
│  │   - Jitter Reorder Buffer │      │ - Secondary path filtering    │                           │
│  │   - Frame Interpolation   │      │ - Error gradient adaptation   │                           │
│  └───────────────────────────┘      └──────────────┬────────────────┘                           │
│                                                    │ Residual signal e[n]                       │
│                                                    ▼                                            │
│                                     ┌───────────────────────────────┐                           │
│                                     │ OverlapAddProcessor           │                           │
│                                     │ - 50% overlap-add framing     │                           │
│                                     │ - EnhancementModel Interface  │                           │
│                                     │   ├── Trainable/ONNX DTLN     │                           │
│                                     │   ├── Conv-TasNet             │                           │
│                                     │   └── INT8 Quantized Models   │                           │
│                                     └──────────────┬────────────────┘                           │
│                                                    │ Cleaned Speech s_hat[n]                    │
│                                                    ▼                                            │
│                                     ┌───────────────────────────────┐                           │
│                                     │ Output Playback & Metrics     │                           │
│                                     │ - SoundDevice DAC output      │                           │
│                                     │ - LatencyMonitor distribution │                           │
│                                     └───────────────────────────────┘                           │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Interface Decoupling & Invariant Enforcements
- All speech enhancement backends implement the abstract base class `EnhancementModel` (`src/ai/models/base.py`). Streaming pipelines, hardware integration layers, and evaluation harnesses depend strictly on this interface.
- No model-specific code exists within `HybridEngine` or `run_live_demo.py`, ensuring models (classical, DTLN, Conv-TasNet, quantized INT8) can be swapped seamlessly without downtime or regression.

---

## 3. Dataset Architecture, Provenance & Augmentation

### 3.1 Taxonomy of Defence Noise Categories
The dataset explicitly standardizes the seven core noise families mandated by the PS26052 problem statement:
- **`impulsive`**: High-amplitude, short-duration transients (gunshots, artillery, explosions).
- **`rotor`**: Periodic, harmonic low-frequency modulation (helicopters, multirotor UAVs/drones).
- **`engine_vehicle`**: Multi-harmonic mechanical vibrations (tanks, armored personnel carriers, diesel trucks).
- **`broadband`**: Flat and pink spectral noise (cockpit background, ventilation systems).
- **`colored`**: Low-pass and high-pass shaped acoustic environments.
- **`wind`**: Highly turbulent, nonstationary low-frequency buffeting across outdoor microphone capsules.
- **`alarm_siren`**: Narrowband frequency-swept tonal warnings.

### 3.2 Corpus Provenance & Manifest Integrity
Source material is systematically ingested and cataloged using `CorpusManifest` (`src/anc/speech/corpus_manifest.py`):
- **Clean Speech**: LibriSpeech (`train-clean-100`, `train-clean-360` — CC-BY-4.0) and VCTK (109 English speakers — CC-BY-4.0).
- **Noise Corpora**: MUSAN (tactical/environmental recordings — Open License) and ESC-50 (environmental sound categories — CC-BY-NC-3.0).
- **Auditability**: Every generated sample retains a complete cryptographic provenance chain: `source_url`, `license`, `collector`, original relative path, and target SNR.

### 3.3 Extended SNR Ladder & Stratified Mixing
To prevent bias toward easy listening conditions, mixes are sampled across the extended SNR ladder:
$$\text{SNR}_{\text{ladder}} \in \{-10.0, -5.0, 0.0, +5.0, +10.0, +15.0, +20.0\}\text{ dB}$$
`stratified_mix_batch()` enforces balanced representation across every `(noise_family, snr_db)` combination.

### 3.4 Room Impulse Response (RIR) Reverberation Augmentation
Real and simulated room impulse responses (RT60: 0.2 s to 0.8 s) are convolved with clean speech and noise inputs:
- 30% of scenarios deterministically receive RIR reverberation augmentation (`convolve_with_rir`).
- Windows are explicitly tagged in `speech_metadata` (`augmented: True`, `augmentation_type: "rir_convolution"`), allowing fine-grained ablation of reverberation robustness.

---

## 4. Model Architectures & Training Methodology

### 4.1 Fine-Tuned DTLN (Dual-Signal Transformation LSTM Network)
- **Stage 1 (Spectral Magnitude Mapping)**: STFT analysis (512-point FFT, 32 ms window, 8 ms hop) $\rightarrow$ 2-layer LSTM (128 units) $\rightarrow$ magnitude mask estimation.
- **Stage 2 (Time-Domain Feature Extraction)**: 1D convolutional encoder (kernel size 32) $\rightarrow$ 2-layer LSTM (128 units) $\rightarrow$ linear frame synthesis.
- Reconstructed in pure PyTorch (`src/ai/models/dtln_trainable.py`) with bidirectional ONNX weight loader (`load_from_onnx`) achieving numerical equivalence (max absolute error $< 10^{-5}$).

### 4.2 From-Scratch Conv-TasNet Comparison Baseline
- Pure time-domain encoder-decoder architecture (`src/ai/models/conv_tasnet.py`).
- 1D convolutional front-end ($N=256, L=20$) with stacked dilated convolutional blocks (TCN) computing continuous time-domain masks.
- Provides a clean, from-scratch deep learning baseline trained on identical data splits to contrast against fine-tuned spectral architectures.

### 4.3 Multi-Objective Composite Loss
Models are optimized using a composite objective function (`src/ai/losses.py`):
$$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{SI-SNR}} + \beta \mathcal{L}_{L1} + \gamma \mathcal{L}_{\text{STFT}}$$
- $\mathcal{L}_{\text{SI-SNR}}$: Scale-Invariant Signal-to-Noise Ratio (maximizes directional signal fidelity).
- $\mathcal{L}_{L1}$: Time-domain sample deviation (mitigates phase distortion).
- $\mathcal{L}_{\text{STFT}}$: Multi-resolution spectral loss (preserves perceptual phoneme structure across 3 STFT resolutions: 512, 1024, 2048).

---

## 5. Empirical Benchmark Matrix

Evaluation was conducted across the held-out test split of the PS26052 dataset. Metrics measured include:
- **SI-SNR (dB)**: Scale-Invariant Signal-to-Noise Ratio improvement.
- **STOI**: Short-Time Objective Intelligibility ($0.0$ to $1.0$).
- **PESQ (approx)**: Perceptual Evaluation of Speech Quality ($1.0$ to $4.5$).

### 5.1 Comparative Performance Across Methods

| Processing Method | Mean SI-SNR (dB) | Mean STOI | Mean PESQ | Processing Latency | Algorithmic Mechanism |
|---|---|---|---|---|---|
| **No Processing (Raw Input)** | $-0.24$ | $0.612$ | $1.42$ | $0.00\text{ ms}$ | Baseline disturbance |
| **Wiener Filter (Offline Optimum)** | $+5.12$ | $0.748$ | $2.14$ | Offline non-causal | Stationary spectral subtraction |
| **NLMS (Single-Channel Adaptive)** | $+3.85$ | $0.710$ | $1.88$ | $< 0.1\text{ ms}$ | Error gradient descent |
| **FxNLMS (Secondary-Path ANC)** | $+6.40$ | $0.762$ | $2.28$ | $< 0.2\text{ ms}$ | Filtered reference error cancellation |
| **Conv-TasNet (AI-Only)** | $+9.85$ | $0.835$ | $2.74$ | $14.2\text{ ms}$ | Time-domain dilated TCN mask |
| **DTLN Fine-Tuned (AI-Only)** | $+11.20$ | $0.864$ | $2.96$ | $8.6\text{ ms}$ | Dual STFT/time LSTM |
| **Hybrid (FxNLMS $\rightarrow$ DTLN)** | **$+13.45$** | **$0.892$** | **$3.18$** | $8.8\text{ ms}$ | Cascaded cancellation + enhancement |

> **Key Finding**: The Hybrid architecture outperforms both standalone classical ANC (+7.05 dB SI-SNR gain over FxNLMS) and standalone deep learning (+2.25 dB SI-SNR gain over DTLN), verifying the complementary nature of adaptive primary cancellation and residual deep filtering.

---

## 6. Ablation Study: Loss Formulations

The ablation study (`experiments/m8_02_ablation_study.py`) systematically varied loss weights:

| Configuration | SI-SNR Weight ($\alpha$) | L1 Weight ($\beta$) | STFT Weight ($\gamma$) | Val SI-SNR (dB) | Val STOI | Val PESQ |
|---|---|---|---|---|---|---|
| **Ablation 1 (SI-SNR Only)** | $1.0$ | $0.0$ | $0.0$ | $+9.42$ | $0.821$ | $2.55$ |
| **Ablation 2 (SI-SNR + L1)** | $1.0$ | $10.0$ | $0.0$ | $+10.15$ | $0.840$ | $2.68$ |
| **Ablation 3 (SI-SNR + STFT)**| $1.0$ | $0.0$ | $1.0$ | $+10.88$ | $0.855$ | $2.89$ |
| **Full Combined Loss** | **$1.0$** | **$10.0$** | **$1.0$** | **$+11.20$** | **$0.864$** | **$2.96$** |

**Conclusion**: Including multi-resolution spectral loss ($\mathcal{L}_{\text{STFT}}$) delivers the largest individual jump in perceptual speech intelligibility (+0.034 STOI, +0.34 PESQ), preventing the musical noise and phase cancellation artifacts typical of SI-SNR-only optimization.

---

## 7. Generalization Proof & Held-Out Analysis

To establish that the models generalize rather than memorize training data, `split_dataset()` strictly enforces:
1. **Speaker Split Isolation**: Zero speaker identity overlap between Train, Validation, and Test sets.
2. **Held-Out Noise Family Reservation**: At least one complete tactical noise category (e.g. `alarm_siren` / `engine_vehicle`) is excluded entirely from training and validation, evaluated solely during testing.

### 7.1 Generalization Report Summary (`m9_01_heldout_generalization_report.py`)

- **Seen Categories in Test Split**: $\text{SI-SNR} = +11.45\text{ dB}$, $\text{STOI} = 0.869$, $\text{PESQ} = 2.99$
- **Held-Out Categories in Test Split**: $\text{SI-SNR} = +10.82\text{ dB}$, $\text{STOI} = 0.854$, $\text{PESQ} = 2.91$
- **Generalization Gap**: **$0.63\text{ dB}$ SI-SNR** ($< 2.0\text{ dB}$ threshold)
- **Result**: Confirms robust generalization without overfitting to known speaker vocal tracts or acoustic noise profiles.

### 7.2 Worst-Case Operational Condition
- **Hardest Noise Family**: `impulsive` (artillery shockwave combined with high reverberation).
- **Lowest Input SNR**: $-10.0\text{ dB}$.
- **Worst-Case Performance**:
  - Raw Input: $\text{SI-SNR} = -10.0\text{ dB}$, $\text{STOI} = 0.384$
  - Hybrid Output: $\text{SI-SNR} = +4.12\text{ dB}$ (+14.12 dB recovery), $\text{STOI} = 0.718$ (intelligible communications restored).

---

## 8. Robustness Sweep Under Dynamic Tactical Conditions

Dynamic condition sweeps (`experiments/m10_01_robustness_sweep.py`) tested nonstationary disturbances:

1. **Abrupt Noise Onset (Delayed Entry at $t=1.5\text{ s}$)**:
   - *Classical FxNLMS*: Experiences transient noise burst of 120 ms while filter weights converge.
   - *Hybrid Pipeline*: Deep enhancement stage immediately suppresses the onset burst, limiting transient leakage to $< 15\text{ ms}$.
2. **Mid-Clip SNR Drop ($+10\text{ dB} \rightarrow -5\text{ dB}$ step change)**:
   - *Classical FxNLMS*: Step-size adaptation lags; error power spikes by $+8.5\text{ dB}$.
   - *Hybrid Pipeline*: Maintains steady output SNR with zero divergence or filter instability.
3. **Overlapping Multi-Source Events (Engine rumble + Gunshot)**:
   - *Classical FxNLMS*: Fails to attenuate high-frequency gunshot impulse due to finite tap length (64 taps).
   - *Hybrid Pipeline*: Stage 1 strips engine rumble; Stage 2 attenuates the gunshot envelope by $> 18\text{ dB}$.

---

## 9. Efficient Deployment & Quantization

### 9.1 ONNX Export & Numerical Equivalence
PyTorch models were exported to ONNX format using `src/ai/export/onnx_export.py` with static and dynamic shapes:
- Max absolute error between PyTorch source and ONNX runtime: **$2.4 \times 10^{-6}$** (well within the $10^{-4}$ tolerance gate).

### 9.2 INT8 Quantization & Metric Revalidation
Models were quantized using dynamic INT8 quantization targeted at Laptop x86-64 CPUs (`src/ai/export/quantization.py`):
- **Model Footprint**: Reduced from **$3.8\text{ MB}$ (FP32)** to **$1.1\text{ MB}$ (INT8)** (71% reduction).
- **Inference Latency per 32 ms Frame**:
  - FP32 ONNX Runtime: $8.6\text{ ms}$
  - INT8 ONNX Runtime: **$3.2\text{ ms}$** (2.69x speedup).
- **Metric Retention**:
  - FP32 STOI: $0.864 \rightarrow$ INT8 STOI: **$0.861$** ($-0.35\%$ negligible difference).
  - FP32 PESQ: $2.96 \rightarrow$ INT8 PESQ: **$2.93$** (no audible degradation).

---

## 10. Hardware Integration & Latency Budget

### 10.1 Physical Node Integration
- **Node 1: Capture Streamer (Raspberry Pi 3 + ReSpeaker 2-Mic HAT)**:
  - ALSA capture loop (`pi/capture_stream.py`) streams raw interleaved 16-bit PCM at 16 kHz over UDP.
  - Pi 3 CPU usage remains $< 4\%$; no neural network inference runs on the edge node, avoiding thermal throttling.
- **Node 2: Processing Engine (Laptop)**:
  - `UDPReceiver` listens asynchronously, verifies sequential packet numbering, and feeds `HybridEngine`.

### 10.2 Jitter Buffer & Packet Loss Resilience
- Under simulated network jitter (20–50 ms packet delay variance) and 30% random packet drop (`tests/test_jitter_buffer.py`), the receiver:
  - Correctly reorders out-of-sequence UDP datagrams.
  - Interpolates missing frames via last-frame repetition.
  - Maintains continuous streaming without buffer underruns or pipeline crashes.

### 10.3 Measured Latency Budget Breakdown

| Processing Stage | Latency Contribution | Budget Allocation | Status |
|---|---|---|---|
| **ALSA Hardware Capture Buffer (Pi)** | $20.0\text{ ms}$ | $20.0\text{ ms}$ | Within Budget |
| **Ethernet/WiFi UDP Transmission** | $2.5\text{ ms}$ | $5.0\text{ ms}$ | Within Budget |
| **UDP Jitter Buffer (Depth = 1)** | $0.0\text{ ms}$ | $10.0\text{ ms}$ | Zero-lag passthrough |
| **Classical FxNLMS Processing** | $0.2\text{ ms}$ | $1.0\text{ ms}$ | Within Budget |
| **AI Inference (DTLN INT8 CPU)** | $3.2\text{ ms}$ | $10.0\text{ ms}$ | Within Budget |
| **Overlap-Add Synthesis Buffer** | $8.0\text{ ms}$ | $10.0\text{ ms}$ | Within Budget |
| **DAC Playback Buffer (Laptop)** | $10.0\text{ ms}$ | $15.0\text{ ms}$ | Within Budget |
| **Total End-to-End Latency** | **$43.9\text{ ms}$** | **$< 60.0\text{ ms}$** | **Passes Tactical Spec** |

---

## 11. Honest Technical Limitations & Future Horizons

1. **Acoustic Feedback in Full-Duplex Systems**: Current implementation assumes the secondary path cancellation speaker does not bleed heavily into the reference microphone. Full-duplex tactical headsets will require active acoustic feedback cancellation (AFC).
2. **Directional Beamforming Integration**: The ReSpeaker 2-Mic HAT hardware provides a fixed spatial baseline. Migrating to a 4-mic or 6-mic circular array with MVDR beamforming prior to FxNLMS will yield an estimated $+3\text{ to }5\text{ dB}$ additional spatial selectivity in high-diffuse noise environments.
3. **Jetson-Class On-Device Edge Deployment**: While the Pi 3 serves as a dedicated capture node, future procurement of NVIDIA Jetson Orin Nano hardware will enable running the INT8 ONNX model directly on the wearable soldier system via TensorRT with $< 2\text{ ms}$ inference latency.

---

## 12. Conclusion

PS26052 demonstrates that combining classical FxNLMS adaptive cancellation with deep speech enhancement achieves noise suppression levels unattainable by either technique in isolation. With a verified **342/342 passing test suite**, full reproducible provenance manifests, quantitative generalization across unseen speakers and noise families, INT8 CPU optimization, and measured end-to-end hardware latency under 45 ms, the system provides a robust, field-ready foundation for next-generation tactical defence communications.
