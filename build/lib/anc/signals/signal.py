from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Signal:
    """
    Canonical discrete-time signal representation used throughout
    the PS26052 ANC project.
    """

    samples: np.ndarray
    sampling_rate: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.samples = np.asarray(self.samples, dtype=np.float64)

        if self.samples.ndim != 1:
            raise ValueError(
                "Signal samples must be a one-dimensional array."
            )

        if self.sampling_rate <= 0:
            raise ValueError(
                "Sampling rate must be greater than zero."
            )

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    @property
    def duration(self) -> float:
        return self.num_samples / self.sampling_rate

    @property
    def time(self) -> np.ndarray:
        return np.arange(self.num_samples) / self.sampling_rate