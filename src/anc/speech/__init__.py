"""Speech and noise data ingestion, mixing, and augmentation for AI-assisted ANC.

This package extends Module 7's dataset pipeline with speech-domain objects
and controlled-SNR mixing, enabling supervised training of speech-enhancement
models. It lives alongside the existing ``anc.signals`` and ``anc.scenarios``
packages without modifying Modules 1–6.
"""

from anc.speech.sources import NoiseSample, SpeechSample
from anc.speech.mixing import mix_at_snr, mix_batch

__all__ = [
    "NoiseSample",
    "SpeechSample",
    "mix_at_snr",
    "mix_batch",
]
