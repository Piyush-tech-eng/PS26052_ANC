# PS26052 ANC — Live Demonstration Operator Runbook

This runbook provides complete, step-by-step procedures for operating and presenting the **PS26052 Two-Microphone Adaptive Noise Cancellation** system to stakeholders, evaluators, and mission operators.

---

## 1. Pre-Flight Checklist

Before presenting the live demonstration, verify the following baseline prerequisites:

```bash
# 1. Activate Python virtual environment
# Windows:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 2. Verify all models are present
python -c "from pathlib import Path; assert (Path('models/dtln/model_1.onnx')).exists(); assert (Path('models/dtln_quantized/model_1.onnx')).exists(); print('[OK] ONNX models verified')"

# 3. Clean previous run status artifacts
python -c "from pathlib import Path; p = Path('results/demo_status.json'); p.unlink(missing_ok=True); print('[OK] Clean telemetry state')"
```

---

## 2. Launching the Live Telemetry Dashboard

The live telemetry dashboard displays all 6 operational panels in real time without external web framework dependencies.

In **Terminal 1**, start the dashboard server:

```bash
python src/anc/dashboard.py --port 8080 --status-file results/demo_status.json
```

- **Dashboard URL**: Open browser to [`http://localhost:8080`](http://localhost:8080)
- The dashboard will enter `STANDBY` mode, ready to visualize live telemetry streams as soon as an orchestrator mode begins.

---

## 3. Demonstration Scenarios

### Scenario 1: Offline File Processing (Immediate Verification)
Processes a recorded multi-condition noisy speech sample through the streaming pipeline frame-by-frame (20 ms / 320 samples per frame):

In **Terminal 2**, execute:

```bash
python run_live_demo.py --mode offline --model dtln \
    --input results/demo_assets/wav/demo_0_rotor_snr+0_noisy.wav \
    --output results/enhanced_output.wav
```

**What to Observe**:
1. Terminal logs frame counter, elapsed uptime, cycle latency (< 22 ms), and real-time processing ratio.
2. In the dashboard at `http://localhost:8080`, observe the **System Status**, **Live Waveforms**, and **Spectral Attenuation** updating dynamically.
3. Enhanced audio is written to `results/enhanced_output.wav`.

---

### Scenario 2: Laptop Live Microphone Demonstration (Zero-Hardware Demo)
Captures live audio from the laptop's built-in microphone, applies neural speech enhancement in real-time, and plays cleaned speech through headphones or speakers.

```bash
python run_live_demo.py --mode laptop-live --model dtln --duration 15
```

**Talking Points**:
- Demonstrates real-time capability with standard consumer microphone inputs.
- Shows sub-25 ms latency headroom suitable for real-time communication systems.

---

### Scenario 3: Physical Hardware Dual-Mic Prototype (Raspberry Pi 3 + ReSpeaker HAT)
Connects the physical Raspberry Pi 3 SBC with Seeed ReSpeaker 2-Mic HAT over sequenced UDP:

#### Step A (On the Raspberry Pi via SSH):
```bash
python3 -m pi.runtime.transport --host 192.168.1.10 --port 5005 --sample-rate 16000 --frame-ms 20
```

#### Step B (On the Laptop):
```bash
python run_live_demo.py --mode pi-prototype --port 5005 --calibrate
```

**What to Observe**:
1. The orchestrator runs a 2-second ambient gain calibration measuring channel RMS ratio and coherence.
2. The dashboard displays separate traces for **Ch1 Reference** (cyan) and **Ch0 Error** (amber).
3. The **Classical ANC Panel** demonstrates FxNLMS adaptive filter convergence and real-time attenuation (-15 to -20 dB).

---

### Scenario 4: Pi-Only Edge Feasibility Demonstration
Demonstrates the empirical feasibility and profiling results of running on-device FxNLMS and quantized INT8 AI on ARM Cortex-A53:

```bash
# Run empirical feasibility benchmark
python experiments/pi_feasibility.py

# Run simulated edge pipeline
python run_live_demo.py --mode pi-edge --model dtln_quantized --duration 10
```

**Key Takeaways to Present**:
- **Configuration A (Pi Capture -> Laptop Hybrid Processing)**: Feasible, recommended for production (< 23.4 ms total latency, 54% Pi CPU).
- **Configuration B (Pi FxNLMS -> Laptop AI)**: Feasible (35.0 ms latency).
- **Configuration C (Full Edge on Pi 3)**: Requires multi-core INT8 threading or next-gen hardware (Pi 4/5).

---

## 4. Launching the Offline Evaluation Dashboard

To display the controlled evaluation benchmarks, SNR sweep curves (-10 dB to +20 dB), and speech distortion safety gate metrics:

In **Terminal 3**, start the evaluation dashboard:

```bash
python src/anc/evaluation/evaluation_dashboard.py --port 8081
```

- **Evaluation URL**: Open browser to [`http://localhost:8081`](http://localhost:8081)
- Shows comprehensive metric tables (SI-SDR, STOI, PESQ) across HVAC, Rotor, Siren, and Babble acoustic scenarios.

---

## 5. Live Demonstration Audio Listening Station

In the dashboard UI at `http://localhost:8080`, use the interactive A/B listening station:
1. Click **Play** on **Input (Noisy + Speech)**: Hear raw rotor noise masking human speech.
2. Click **Play** on **Enhanced (Cleaned Output)**: Hear noise suppressed by >18 dB with clear voice intelligibility.
3. Click **Play** on **Reference (Clean Target)**: Compare with the ground-truth reference speech.

---

## 6. Troubleshooting & Diagnostics

| Symptom | Probable Cause | Remediation |
|---|---|---|
| `Address already in use (Errno 48/98/10048)` | Previous server instance still bound to port 8080 or 5005 | Run `netstat -ano \| findstr :8080` and terminate lingering process, or pass `--port 8082` |
| `No input audio detected` | Microphone muted or incorrect default ALSA device | Check `alsamixer` settings on Pi; verify input volume on laptop |
| High RT Ratio (> 1.0x) | Heavy background processes or unquantized model | Run with `--model dtln_quantized` for INT8 acceleration |
| Channels Inverted (Voice cancelled instead of noise) | Physical microphones swapped | Pass `--ch0-is-reference` CLI flag to invert assignments digitally |
