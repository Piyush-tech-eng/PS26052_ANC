"""Controlled-SNR mixing of speech and noise for supervised training.

Implements the core mixing equation::

    m[n] = s[n] + alpha * v[n]

where ``alpha`` is computed to hit a target SNR, producing reproducible
and auditable training samples with full provenance metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from anc.speech.sources import NoiseSample, SpeechSample


@dataclass(frozen=True)
class MixedSample:
    """One mixed speech+noise sample with full reproducibility metadata.

    Attributes
    ----------
    noisy_speech : np.ndarray
        The mixed signal m[n] = s[n] + alpha * v[n].
    clean_speech : np.ndarray
        The clean speech target s[n].
    noise_component : np.ndarray
        The scaled noise alpha * v[n].
    sampling_rate : int
        Common sampling rate.
    snr_db : float
        Target SNR used for mixing.
    alpha : float
        Computed noise scaling factor.
    speech_source_id : str
        Source ID of the speech sample.
    speaker_id : str
        Speaker identity for split grouping.
    noise_source_id : str
        Source ID of the noise sample.
    noise_family : str
        Noise category from the PS26052 taxonomy.
    mix_id : str
        Unique identifier for this mix.
    provenance : dict
        Full provenance chain.
    """

    noisy_speech: np.ndarray
    clean_speech: np.ndarray
    noise_component: np.ndarray
    sampling_rate: int
    snr_db: float
    alpha: float
    speech_source_id: str
    speaker_id: str
    noise_source_id: str
    noise_family: str
    mix_id: str
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.noisy_speech) != len(self.clean_speech):
            raise ValueError("noisy_speech and clean_speech must have equal lengths.")
        if len(self.noisy_speech) != len(self.noise_component):
            raise ValueError("noisy_speech and noise_component must have equal lengths.")


def _compute_alpha(speech: np.ndarray, noise: np.ndarray, target_snr_db: float) -> float:
    """Compute the noise scaling factor alpha to achieve a target SNR.

    SNR = 10 * log10(P_s / P_n_scaled)
    P_n_scaled = alpha^2 * P_n
    alpha = sqrt(P_s / (P_n * 10^(SNR/10)))
    """
    speech_power = float(np.mean(speech ** 2))
    noise_power = float(np.mean(noise ** 2))

    floor = np.finfo(np.float64).eps
    if speech_power < floor or noise_power < floor:
        return 0.0

    alpha = np.sqrt(speech_power / (noise_power * (10.0 ** (target_snr_db / 10.0))))
    return float(alpha)


def mix_at_snr(
    speech: SpeechSample,
    noise: NoiseSample,
    target_snr_db: float,
    *,
    mix_id: str | None = None,
    trim_to_shorter: bool = True,
) -> MixedSample:
    """Mix a speech sample with noise at a specified SNR.

    Parameters
    ----------
    speech : SpeechSample
        Clean speech signal.
    noise : NoiseSample
        Noise signal.
    target_snr_db : float
        Target signal-to-noise ratio in dB.
    mix_id : str, optional
        Unique identifier; auto-generated if not provided.
    trim_to_shorter : bool
        If True, trim both signals to the length of the shorter one.
        If False, raise an error when lengths differ.

    Returns
    -------
    MixedSample
        The mixed sample with full provenance.
    """
    if speech.sampling_rate != noise.sampling_rate:
        raise ValueError(
            f"Sampling rates must match: speech={speech.sampling_rate}, "
            f"noise={noise.sampling_rate}. Resample first."
        )

    s = speech.audio.copy()
    v = noise.audio.copy()

    if len(s) != len(v):
        if trim_to_shorter:
            min_len = min(len(s), len(v))
            s = s[:min_len]
            v = v[:min_len]
        else:
            raise ValueError(
                f"Speech length ({len(s)}) and noise length ({len(v)}) differ. "
                f"Set trim_to_shorter=True or resample/pad manually."
            )

    alpha = _compute_alpha(s, v, target_snr_db)
    noise_scaled = alpha * v
    mixed = s + noise_scaled

    if mix_id is None:
        mix_id = f"{speech.source_id}__{noise.noise_id}__snr{target_snr_db:+.0f}dB"

    provenance = {
        "speech_source_id": speech.source_id,
        "speaker_id": speech.speaker_id,
        "noise_source_id": noise.source_id,
        "noise_id": noise.noise_id,
        "noise_family": noise.noise_family,
        "target_snr_db": target_snr_db,
        "computed_alpha": alpha,
        "speech_provenance": dict(speech.provenance),
        "noise_provenance": dict(noise.provenance),
    }

    return MixedSample(
        noisy_speech=mixed,
        clean_speech=s,
        noise_component=noise_scaled,
        sampling_rate=speech.sampling_rate,
        snr_db=target_snr_db,
        alpha=alpha,
        speech_source_id=speech.source_id,
        speaker_id=speech.speaker_id,
        noise_source_id=noise.source_id,
        noise_family=noise.noise_family,
        mix_id=mix_id,
        provenance=provenance,
    )


# Default SNR ladder from Section 3.3 of the AI transformation plan
DEFAULT_SNR_LADDER_DB = (-5.0, 0.0, 5.0, 10.0, 15.0)


def mix_batch(
    speech_samples: Iterable[SpeechSample],
    noise_samples: Iterable[NoiseSample],
    snr_ladder_db: Iterable[float] = DEFAULT_SNR_LADDER_DB,
) -> list[MixedSample]:
    """Create a Cartesian product of speech × noise × SNR mixes.

    Parameters
    ----------
    speech_samples : iterable of SpeechSample
    noise_samples : iterable of NoiseSample
    snr_ladder_db : iterable of float
        Target SNR values in dB.

    Returns
    -------
    list of MixedSample
        All speech × noise × SNR combinations.
    """
    speech_list = list(speech_samples)
    noise_list = list(noise_samples)
    snr_list = list(snr_ladder_db)

    if not speech_list:
        raise ValueError("At least one speech sample is required.")
    if not noise_list:
        raise ValueError("At least one noise sample is required.")
    if not snr_list:
        raise ValueError("At least one SNR value is required.")

    results: list[MixedSample] = []
    for speech in speech_list:
        for noise in noise_list:
            for snr_db in snr_list:
                results.append(mix_at_snr(speech, noise, snr_db))
    return results
