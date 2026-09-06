"""Secondary-path model loading for ANC replay.

M6.05 connects a saved secondary-path identification artifact from
Module 5 to the Module 6 ANC replay engine.

The identified FIR coefficients represent:

    S_hat(z)

and are used by FxLMS/FxNLMS to construct the filtered reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json

import numpy as np


@dataclass(frozen=True)
class ReplaySecondaryPathModel:
    """Validated secondary-path model used by ANC replay."""

    coefficients: np.ndarray
    source_path: Path
    metadata: dict[str, Any] | None = None

    @property
    def length(self) -> int:
        return int(len(self.coefficients))


def _validate_coefficients(
    coefficients: np.ndarray | list[float],
    *,
    name: str,
) -> np.ndarray:
    """Validate a one-dimensional finite FIR coefficient vector."""

    values = np.asarray(
        coefficients,
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional."
        )

    if len(values) == 0:
        raise ValueError(
            f"{name} must not be empty."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            f"{name} contains NaN or Inf."
        )

    return values.copy()


def load_identified_secondary_path(
    coefficients_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
) -> ReplaySecondaryPathModel:
    """Load a Module 5 identified secondary-path coefficient artifact.

    Parameters
    ----------
    coefficients_path:
        Path to a .npy file containing the identified FIR coefficients.

    metadata_path:
        Optional JSON metadata file describing the identification run.

    Returns
    -------
    ReplaySecondaryPathModel
        Validated model ready for run_anc_replay().
    """

    coefficients_path = Path(
        coefficients_path
    )

    if not coefficients_path.is_file():
        raise FileNotFoundError(
            "Secondary-path coefficient artifact "
            "was not found: "
            f"{coefficients_path}"
        )

    try:
        coefficients = np.load(
            coefficients_path,
            allow_pickle=False,
        )
    except (OSError, ValueError) as error:
        raise ValueError(
            "Could not load secondary-path "
            "coefficient artifact: "
            f"{coefficients_path}"
        ) from error

    coefficients = _validate_coefficients(
        coefficients,
        name="secondary-path coefficients",
    )

    metadata = None

    if metadata_path is not None:

        metadata_path = Path(
            metadata_path
        )

        if not metadata_path.is_file():
            raise FileNotFoundError(
                "Secondary-path metadata file "
                "was not found: "
                f"{metadata_path}"
            )

        try:
            with metadata_path.open(
                "r",
                encoding="utf-8",
            ) as file:

                metadata = json.load(
                    file
                )

        except json.JSONDecodeError as error:
            raise ValueError(
                "Secondary-path metadata is "
                "not valid JSON."
            ) from error

        if not isinstance(
            metadata,
            dict,
        ):
            raise ValueError(
                "Secondary-path metadata must "
                "contain a JSON object."
            )

    return ReplaySecondaryPathModel(
        coefficients=coefficients,
        source_path=coefficients_path.resolve(),
        metadata=metadata,
    )