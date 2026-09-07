"""Abstract base class for all speech enhancement models.

Every enhancement backend — pretrained wrapper, fine-tuned checkpoint, or
from-scratch architecture — must implement this interface.  Downstream code
(the streaming overlap-add processor, the hybrid engine, the hardware
playback loop) depends **only** on this ABC, which is what allows hot-swapping
models without touching any integration code.

Design constraints
------------------
* Input/output are 1-D ``np.ndarray`` (mono, float64).
* The model declares its native ``sample_rate`` — callers are responsible for
  resampling to/from it (``anc.speech.resampling`` provides this).
* ``frame_size`` tells the streaming layer how large each processing chunk
  should be.  A model that can handle arbitrary lengths should return ``0``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EnhancementModel(ABC):
    """Interface contract for a single-channel speech enhancement backend."""

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model identifier (e.g. ``'rnnoise'``, ``'dtln'``)."""

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Native sample rate this model expects, in Hz."""

    @property
    @abstractmethod
    def frame_size(self) -> int:
        """Number of samples per processing frame.

        Return ``0`` if the model can process arbitrary-length inputs.
        The streaming overlap-add layer uses this to chunk the audio.
        """

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    @abstractmethod
    def enhance(self, x: np.ndarray) -> np.ndarray:
        """Enhance a mono audio segment.

        Parameters
        ----------
        x : np.ndarray
            1-D float64 mono audio at ``self.sample_rate``.

        Returns
        -------
        np.ndarray
            Enhanced audio, same length and dtype as *x*.

        Raises
        ------
        ValueError
            If *x* is not 1-D or is empty.
        """

    # ------------------------------------------------------------------
    # Convenience helpers (concrete)
    # ------------------------------------------------------------------

    def validate_input(self, x: np.ndarray, *, name: str = "x") -> np.ndarray:
        """Validate and coerce input to float64 1-D array.

        Subclasses should call this at the top of :meth:`enhance`.
        """
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional, got {arr.ndim}-D.")
        if arr.size == 0:
            raise ValueError(f"{name} must not be empty.")
        return arr

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, "
            f"sample_rate={self.sample_rate}, frame_size={self.frame_size})"
        )
