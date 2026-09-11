# PS26052 ANC — Physical Hardware Setup & Configuration Guide

This document specifies the complete physical setup, electrical wiring, audio codec configuration, network transport, and calibration procedure for deploying the **PS26052 Active Noise Cancellation** system to physical hardware consisting of a **Raspberry Pi 3 Model B/B+** paired with a **Seeed Studio ReSpeaker 2-Mic Pi HAT**.

---

## 1. Hardware Bill of Materials (BOM)

| Component | Specification | Function in System |
|---|---|---|
| **Host SBC** | Raspberry Pi 3 Model B / B+ (1.2 GHz Quad-Core Cortex-A53, 1GB LPDDR2) | Edge capture, UDP transport, optional edge FxNLMS filtering |
| **Audio HAT** | Seeed Studio ReSpeaker 2-Mic Pi HAT (WM8960 Audio Codec) | Dual-microphone capture, hardware pre-amps, 3.5mm headphone playback |
| **Microphone Array** | 2x Analog MEMS Microphones (Separation distance: 58 mm) | Ch0: Error / cancellation point; Ch1: Reference / noise-facing |
| **Storage** | 16 GB+ MicroSD Card (UHS-1 / Class 10), Raspberry Pi OS Lite (64-bit or 32-bit) | OS and Pi runtime environment |
| **Network** | Cat5e/Cat6 Ethernet Cable or 2.4 GHz 802.11n Wi-Fi | Low-latency UDP transport (< 1.0 ms LAN latency) |
| **Power Supply** | 5V / 2.5A Micro-USB Dedicated Power Supply | Clean power to prevent ground loop noise on audio ADCs |

---

## 2. Physical Assembly & Microphone Placement

```
                      ┌────────────────────────────────────────┐
                      │    ReSpeaker 2-Mic HAT (Top View)     │
                      │                                        │
     Noise Source     │  [MIC 1 / Ch1]          [MIC 0 / Ch0]  │    Speech / Speaker
   (Engine, Rotor, ──>│  Reference Mic           Error Mic     │<── Cancellation Zone
      HVAC, Fan)      │  (Facing Noise)        (Facing Voice)  │
                      │                                        │
                      │   [Headphone Jack 3.5mm]     [Button]  │
                      └──────────────────┬─────────────────────┘
                                         │ 40-Pin GPIO Header
                      ┌──────────────────┴─────────────────────┐
                      │          Raspberry Pi 3 SBC            │
                      │                                        │
                      │   [Ethernet]     [USB x4]   [5V Power] │
                      └────────────────────────────────────────┘
```

### Channel Orientation:
- **Channel 1 (Reference Microphone)**: Pointed towards the dominant primary ambient noise source (e.g., motor, fan, drone rotor, external siren).
- **Channel 0 (Error Microphone)**: Positioned near the operator's ear / mouth / desired cancellation zone. This captures residual acoustic noise plus desired speech.
- **Orientation Inversion**: If physical mounting places Channel 0 towards the noise, pass `--ch0-is-reference` to the runtime software to digitally invert assignments without rewiring.

---

## 3. Operating System & Audio Codec Driver Setup

### 3.1 Kernel Configuration
On the Raspberry Pi, edit `/boot/config.txt` (or `/boot/firmware/config.txt` on newer OS releases):

```bash
sudo nano /boot/config.txt
```

Ensure the following device tree overlays are enabled:
```ini
# Disable onboard Broadcom audio to avoid sound card index jitter
dtparam=audio=off

# Enable I2S and I2C buses
dtparam=i2s=on
dtparam=i2c_arm=on

# Enable Seeed Voicecard WM8960 overlay
dtoverlay=seeed-2mic-voicecard
```

### 3.2 Install WM8960 Kernel Driver
Clone and install the official Seeed-voicecard driver:

```bash
git clone https://github.com/respeaker/seeed-voicecard.git
cd seeed-voicecard
sudo ./install.sh
sudo reboot
```

### 3.3 Verify Sound Card Enumeration
After reboot, verify the audio devices:

```bash
# Verify capture devices
arecord -l
# Expected output:
# card 0: seeed2micvoicec [seeed-2mic-voicec], device 0: bcm2835-i2s-wm8960-hifi wm8960-hifi-0

# Verify playback devices
aplay -l
# Expected output:
# card 0: seeed2micvoicec [seeed-2mic-voicec], device 0: bcm2835-i2s-wm8960-hifi wm8960-hifi-0
```

---

## 4. ALSA Configuration & Gain Staging

Create or update `/etc/asound.conf`:

```ini
pcm.!default {
    type asym
    playback.pcm "hw:CARD=seeed2micvoicec,DEV=0"
    capture.pcm  "hw:CARD=seeed2micvoicec,DEV=0"
}

ctl.!default {
    type hw
    card seeed2micvoicec
}
```

### Gain Staging with ALSA Mixer:
Open `alsamixer`:
```bash
alsamixer -c seeed2micvoicec
```
Recommended gain staging to prevent ADC clipping while maintaining high dynamic range:
- **Capture**: 80% (approx +18 dB)
- **ADC PCM**: 85%
- **Left Input Mixer Boost**: Enable (+20 dB)
- **Right Input Mixer Boost**: Enable (+20 dB)
- **Playback**: 75%

Save ALSA mixer settings permanently:
```bash
sudo alsactl store
```

---

## 5. Network Configuration for Low-Jitter UDP

To ensure packet jitter < 1.0 ms across the UDP link:

### 5.1 Static Ethernet Configuration (Recommended)
Connect Pi directly to laptop Ethernet port or LAN switch:
- **Pi IP**: `192.168.1.50` (Subnet mask: `255.255.255.0`)
- **Laptop IP**: `192.168.1.10`

Configure static IP in `/etc/dhcpcd.conf` on the Pi:
```ini
interface eth0
static ip_address=192.168.1.50/24
```

### 5.2 Network Jitter Verification
From the laptop terminal, test ping latency:
```bash
ping 192.168.1.50
# Expected: time < 0.8 ms, 0% packet loss
```

---

## 6. Software Deployment to Raspberry Pi

The automated deployment script `pi/runtime/deploy.sh` bundles the standalone Pi runtime:

```bash
# On Laptop: Copy pi runtime to Raspberry Pi
scp -r pi/ pi@192.168.1.50:/home/pi/ps26052_pi/

# SSH into Raspberry Pi
ssh pi@192.168.1.50

# Execute automated deployment
cd /home/pi/ps26052_pi/runtime
chmod +x deploy.sh
./deploy.sh
```

---

## 7. Starting the Hardware Audio Streaming Link

### On the Raspberry Pi:
Start the low-latency audio capture and transport stream:

```bash
# Stream 16 kHz stereo audio to Laptop (192.168.1.10) on UDP port 5005
python3 -m pi.runtime.transport --host 192.168.1.10 --port 5005 --sample-rate 16000 --frame-ms 20
```

### On the Laptop:
Run the live ANC demo in `pi-prototype` mode:

```bash
python run_live_demo.py --mode pi-prototype --port 5005 --calibrate
```

The system will perform a 2-second ambient channel calibration, equalize microphone sensitivities, and commence streaming real-time adaptive cancellation.
