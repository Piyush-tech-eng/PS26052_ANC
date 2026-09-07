"""Hardware integration for real-time ANC demo.

This package provides:
- ``udp_receiver``: Laptop-side UDP listener for Pi audio frames
- ``playback``: Real-time audio output via sounddevice
- ``latency_monitor``: Per-stage timing instrumentation
- ``live_metrics_overlay``: Rolling SNR/STOI display for demo
"""
