"""AI-assisted speech enhancement for the PS26052 ANC pipeline.

This package provides:
- ``models``: Pluggable enhancement backends (pretrained wrappers + custom)
- ``streaming``: Real-time overlap-add and hybrid ANC+AI engine
- ``hardware``: UDP capture/playback for Raspberry Pi integration
- ``losses``: Training loss functions (SI-SNR, perceptual, multi-res STFT)

Every enhancement model implements the common interface defined in
``ai.models.base.EnhancementModel``, so downstream streaming and hardware
code never depends on which specific model is active.
"""
