# PS26052_ANC — Hackathon-Ready Plan (Software Rescope + Hardware + Integration)

This document does three things, in order: (1) stress-tests the AI-transformation plan for genericness/completeness and rescopes it to fit **3–4 days**, (2) gives a concrete hardware bill-of-materials and architecture using only a Pi 3 + laptop + newly-bought mic modules, (3) gives a detailed software↔hardware integration plan you can walk a judge through end to end.

---

## PART 1 — Is the plan complete and general? What changes.

### 1.1 Where the plan was already generic (keep as-is)

- Multi-speaker speech corpus (LibriSpeech/VCTK) + multiple noise families + an SNR ladder + splitting by speaker/source identity — this is what makes a model generalize instead of memorizing one clip. Keep this design.
- The dataset contract (`reference_x`, `noisy_speech`, `baseline_residual`, `clean_speech_target`) is format-general: any audio, once resampled to the canonical rate and windowed, fits the same pipeline. That's correct and doesn't need to change.

### 1.2 Real gaps that would make it "a single-audio experiment" if not fixed

These are worth fixing regardless of time pressure, because they're cheap and they're exactly the kind of thing a judge will probe:

1. **Fixed-window training vs. arbitrary-length inference.** Training happens on fixed 512-sample windows; a live demo needs to process a *continuous, arbitrary-length* stream. You need an explicit **overlap-add (or overlap-save) streaming wrapper** around the model: buffer incoming audio into overlapping frames, run inference per frame, cross-fade/sum the overlaps back into a continuous output. Without this, the model will only ever "work" on clips shaped exactly like training data — which is precisely the toy-demo failure mode you're worried about. **This is now a required Module 9 component, not optional.**
2. **Sample-rate/format agnosticism at the input boundary.** Add one small resampling/format-normalization function at the very entry point of inference (convert whatever comes in — 44.1kHz stereo WAV, 16kHz mono mic stream, etc. — to the model's canonical rate/mono/float32). This is a 20-line utility, but without it the system only "works" on inputs pre-shaped to match training, which again reads as a toy.
3. **Held-out generalization proof, not just a training-loss number.** Keep at least one noise family and a couple of speakers **entirely out of training**, and show the metric table on that held-out set during judging. This is the single most convincing way to show "this isn't overfit to a demo clip."
4. **Metrics must be reported over a batch, not eyeballed on one file.** Whatever time you have for evaluation, run it over a test set (even a modest one — 20–30 clips across noise families/SNRs) and report SNR/STOI/PESQ as a table. One anecdote is not evidence of generality.

### 1.3 The honest rescope for 3–4 days

Training a neural speech-enhancement model from scratch — data pipeline debugging, training runs, hyperparameter iteration — reliably takes longer than 3–4 days even with GPU access, before you've touched hardware at all. Trying to do that *and* build a live hardware demo in the same window is the actual risk to your hackathon outcome, not the ANC math.

**Rescoped Module 8: use a small pretrained, open, lightweight speech-enhancement model as the deployed AI component, and demonstrate your training pipeline as a capability rather than the source of the final weights.**

Concretely:
- **RNNoise** (tiny RNN-based denoiser, extremely fast, runs comfortably on a laptop CPU in real time, has pretrained weights freely available) or a small pretrained **DTLN** checkpoint are both legitimate, well-known, lightweight speech-enhancement models that match the PS statement's "real-time, edge-deployable" requirement far better than anything you could train well from scratch in days.
- Use it as the AI stage **downstream of your existing FxNLMS classical residual** — this is still exactly the hybrid architecture the PS statement and your own Master History describe. You are not skipping the "AI/ML model" requirement; you're being pragmatic about where the weights come from under a hard deadline.
- **If time genuinely permits after the hardware demo is solid**, fine-tune the pretrained model briefly on your own generated dataset (even a few epochs on a subset) and show a before/after metrics delta — this demonstrates the training framework works without betting the whole demo on convergence happening in time.
- Present this honestly to judges: "We built the full dataset-generation and training pipeline [show it, show it running]; for the deployed prototype under the hackathon's time constraint we integrated it with a proven lightweight pretrained model, and our pipeline is designed to swap in a fully custom-trained model as training time increases." That is a *stronger*, more credible story than a half-trained custom model with bad metrics.

### 1.4 What now genuinely satisfies every PS objective within this rescope

| PS objective | How it's satisfied in the rescoped plan |
|---|---|
| Scalable dataset pipeline, noisy-clean pairs, varying SNR | Module 7 extension (already designed) — runs regardless of which model consumes it |
| AI/ML model for noise suppression | Pretrained lightweight model (RNNoise/DTLN) wired into the same `[x, e_ANC]`-style hybrid input |
| Training framework with losses/metrics | Built and demonstrably run (even briefly) — SI-SNR/L1/perceptual loss, STOI/PESQ/SNR eval, all from the existing `metrics.py` extension plan |
| Real-time inference engine | Overlap-add streaming wrapper (Section 1.2.1) — this is the piece that makes it real, not a script |
| Prototype with mic/headset integration | Part 2/3 below |
| SNR>15dB / STOI>0.85 / PESQ>2.5 | Report honestly on your held-out test set; if the pretrained-only path doesn't hit these on your specific noise types, say so and show the delta after brief fine-tuning — judges respect honest measured numbers far more than an unverifiable claim |

---

## PART 2 — Hardware Plan (Pi 3 + laptop, few days, limited budget)

### 2.1 The core architecture decision

**The Pi 3 does capture only. The laptop does all compute (classical ANC + AI). No model runs on the Pi.**

This is not a limitation you need to apologize for — it's literally the "microphone acquisition → edge processor → ... → communication output" chain the PS statement describes, where your laptop is standing in for the Jetson-class hardware the PS names as the eventual target. Say this explicitly to judges: *"Pi 3 + laptop is our rapid prototype of the acquisition/compute split; the production target is a Jetson-class SoC replacing the laptop role, per the problem statement."* That framing turns a constraint into a deliberate engineering decision.

### 2.2 Bill of materials (buy these)

| Item | Why | Notes |
|---|---|---|
| **ReSpeaker 2-Mic Pi HAT** (or similar 2-mic array HAT for Raspberry Pi) | This is the single best buy for your situation: gives you **two synchronized mic channels** (reference + error mic, exactly what feedforward ANC needs) over a well-documented driver stack, mounts directly on the Pi 3's GPIO header, no soldering. Far less setup risk than wiring raw I2S MEMS mics yourself in a few days. | Comes with example capture scripts; budget ~1 day for setup/driver install + test |
| A small powered speaker (USB or 3.5mm, battery-powered ones work well for demo mobility) | Plays the "noise source" during the demo, and optionally the enhanced output if you do the full round-trip | Cheap, widely available |
| Ethernet cable (or reliable WiFi hotspot) between Pi and laptop | Lower, more stable latency than shared WiFi for the live streaming link | Ethernet strongly preferred if a judge's venue WiFi is congested |
| (Optional, stretch) Small amp+speaker module (e.g. PAM8403 + speaker) wired to Pi's audio out or a USB DAC | Only needed if you want the enhanced audio to physically play back *from the Pi side* for a more "closed-loop" visual demo | Skip if time is tight — playing the output from the laptop speaker is a fully legitimate demo and much lower risk |

**Skip:** raw I2S MEMS mic breakout boards (INMP441 etc.) unless someone on the team has done Pi I2S driver config before — it's a real time sink for the payoff you get from the HAT solution above. **Skip:** buying a Jetson or any AI accelerator board — not needed for the prototype story above, and there's no time to bring one up properly anyway.

### 2.3 Physical demo setup

```
[Speaker: plays defence-like noise clip]         [Person speaking, or phone playing clean speech]
              \                                          /
               \                                        /
          [ReSpeaker 2-Mic HAT on Pi 3]  ← captures reference mic (near noise) + error/mixture mic (near speech)
                              │
                    Pi 3 streams both channels
                    over Ethernet/WiFi (UDP)
                              │
                              ▼
                     [Laptop receives stream]
                              │
                    Classical FxNLMS (streaming) → residual
                              │
                    AI enhancement (RNNoise/DTLN, streaming, overlap-add)
                              │
                              ▼
                 [Laptop speaker / headphone jack plays enhanced output]
                              │
                 (stretch) streamed back to Pi-side speaker
```

---

## PART 3 — Integration Plan (software ↔ hardware, end to end)

### 3.1 Pi 3 side (capture + transport only)

1. Install ReSpeaker drivers, verify 2-channel capture with `arecord -l` / a short test recording.
2. Write a small capture-and-stream script: read fixed-size PCM16 frames (e.g., 20–40 ms per frame — small enough for low latency, large enough not to drown the network in packet overhead) from ALSA, and send each frame as one UDP packet to the laptop's IP:port. Keep it dead simple — raw PCM over UDP, no compression, no fancy protocol. A dropped packet should just mean a tiny audio glitch, not a crash.
3. That's the Pi's entire job. No ANC code, no model, no Python ML dependencies on the Pi at all — keeps the Pi's limited RAM/CPU completely out of the critical path.

### 3.2 Laptop side (all the intelligence)

1. **Socket receiver**: a small server that listens on the UDP port, reassembles incoming frames per channel into a continuous ring buffer per channel (reference, error/mixture).
2. **Streaming classical ANC**: take the existing offline `run_anc_replay()` FxNLMS logic and refactor it into a *stateful, frame-by-frame* call — same filter-update math, but called once per incoming frame with the adaptive filter's coefficients persisted between calls instead of processed over a whole pre-loaded array. This is a genuinely important piece of new code (the current repo is 100% offline/batch), budget real time for it, but it's the same algorithm, not new math.
3. **Streaming AI stage**: feed the ANC residual frame (plus reference, if using the 2-channel model input) into the **overlap-add wrapper** from Part 1.2.1 around the pretrained RNNoise/DTLN model, producing an enhanced-audio frame.
4. **Playback**: push the enhanced frame to the laptop's audio output device in real time (`sounddevice`/`pyaudio`, blocking or callback-based streaming — callback-based is smoother and worth the setup time).
5. **Live metrics overlay (nice-to-have, high judge impact)**: while the demo runs, compute a rolling SNR estimate on-screen so judges see a live number moving, not just hear an audio difference.

### 3.3 Latency budget (track this explicitly, don't guess)

| Stage | Rough budget |
|---|---|
| Pi capture buffer (frame size) | 20–40 ms |
| Network transfer (Ethernet) | 1–5 ms |
| Streaming FxNLMS processing | a few ms (it's a small FIR-order filter) |
| AI model inference (RNNoise-class model, laptop CPU) | 5–20 ms per frame |
| Output buffer/playback | 20–40 ms |
| **Total round-trip target** | aim for **under ~150 ms** — noticeable but acceptable for a live demo; RNNoise-class models comfortably fit this on a laptop CPU |

Measure this live and put the number on a slide — "our end-to-end latency is X ms" is a concrete, credible engineering statement judges will remember.

### 3.4 Always have a backup demo path

Live audio demos fail in front of judges for boring reasons (venue WiFi, mic feedback, a loose cable). Prepare, in parallel:
- A pre-recorded before/after audio pair (noisy vs. enhanced) you can play instantly if the live rig hiccups.
- A one-slide metrics table (SNR/STOI/PESQ, held-out test set, per noise family) you present regardless of whether the live demo works — this is actually stronger evidence than the live demo anyway, and protects you if the room's acoustics make a live mic demo unconvincing.

### 3.5 Day-by-day schedule (2+ people, parallel tracks)

| Day | Software track | Hardware track |
|---|---|---|
| 1 | Finish Module 7 dataset extension + get RNNoise/DTLN pretrained model running offline on test files; extend `metrics.py` for STOI/PESQ/SNR | ReSpeaker HAT setup on Pi 3, verify 2-channel capture, write + test UDP streaming script |
| 2 | Build streaming FxNLMS (frame-by-frame) + overlap-add AI wrapper; test on pre-recorded files simulating a stream | Laptop-side UDP receiver + ring buffer; get raw passthrough (mic → laptop → speaker, no processing) working end to end |
| 3 | Wire streaming ANC + AI into the receiver pipeline; measure latency; tune buffer sizes | Full physical setup rehearsal: noise speaker + speech + Pi + laptop, in the room type you'll present in |
| 4 | Polish: live metrics overlay, held-out test-set metrics table, (stretch) brief fine-tune if time allows | Backup recordings, slide deck, full dry-run presentation with both tracks combined |

This gets you to a judge-facing demo that is genuinely real-time, genuinely general (held-out speakers/noise, batch-evaluated metrics), and honestly framed about where the AI weights came from under the time constraint — which is a materially stronger position than an unfinished custom-trained model or a single-clip toy demo.
