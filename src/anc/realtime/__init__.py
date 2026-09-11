"""Real-time streaming infrastructure for PS26052 ANC.

This package provides the persistent audio processing pipeline and its
supporting components:

- ``audio_input`` — abstract AudioInput + concrete sources (WAV, mic, Pi)
- ``audio_output`` — abstract AudioOutput + concrete sinks (WAV, speaker, Pi)
- ``ring_buffer`` — thread-safe circular buffer for audio data
- ``telemetry`` — pipeline telemetry and status reporting
- ``streaming_pipeline`` — the main continuous processing engine
"""

from anc.realtime.audio_input import AudioFrame, AudioInput
from anc.realtime.audio_output import AudioOutput
from anc.realtime.ring_buffer import AudioRingBuffer
from anc.realtime.telemetry import PipelineTelemetry, TelemetryCollector

__all__ = [
    "AudioFrame",
    "AudioInput",
    "AudioOutput",
    "AudioRingBuffer",
    "PipelineTelemetry",
    "TelemetryCollector",
]
