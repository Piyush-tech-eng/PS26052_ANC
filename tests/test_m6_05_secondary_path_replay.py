from __future__ import annotations

import json

import numpy as np
import pytest

from anc.replay import (
    ReplaySecondaryPathModel,
    load_identified_secondary_path,
)


def test_load_identified_secondary_path(
    tmp_path,
) -> None:

    coefficients = np.array(
        [
            0.0,
            0.0,
            0.62,
            0.28,
            -0.10,
            0.04,
        ],
        dtype=np.float64,
    )

    coefficients_path = (
        tmp_path
        / "identified_secondary_path.npy"
    )

    np.save(
        coefficients_path,
        coefficients,
    )

    model = (
        load_identified_secondary_path(
            coefficients_path
        )
    )

    assert isinstance(
        model,
        ReplaySecondaryPathModel,
    )

    assert model.length == len(
        coefficients
    )

    np.testing.assert_allclose(
        model.coefficients,
        coefficients,
    )


def test_load_metadata(
    tmp_path,
) -> None:

    coefficients_path = (
        tmp_path
        / "identified.npy"
    )

    metadata_path = (
        tmp_path
        / "metadata.json"
    )

    coefficients = np.array(
        [
            0.0,
            0.5,
            0.2,
        ],
        dtype=np.float64,
    )

    metadata = {
        "module": "M5",
        "algorithm": "NLMS",
        "filter_length": 3,
    }

    np.save(
        coefficients_path,
        coefficients,
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
        )

    model = (
        load_identified_secondary_path(
            coefficients_path,
            metadata_path=metadata_path,
        )
    )

    assert model.metadata == metadata


def test_missing_coefficients_rejected(
    tmp_path,
) -> None:

    missing_path = (
        tmp_path
        / "missing.npy"
    )

    with pytest.raises(
        FileNotFoundError,
    ):
        load_identified_secondary_path(
            missing_path
        )


def test_empty_coefficients_rejected(
    tmp_path,
) -> None:

    path = (
        tmp_path
        / "empty.npy"
    )

    np.save(
        path,
        np.array(
            [],
            dtype=np.float64,
        ),
    )

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        load_identified_secondary_path(
            path
        )


def test_nan_coefficients_rejected(
    tmp_path,
) -> None:

    path = (
        tmp_path
        / "invalid.npy"
    )

    np.save(
        path,
        np.array(
            [
                0.0,
                np.nan,
            ],
            dtype=np.float64,
        ),
    )

    with pytest.raises(
        ValueError,
        match="NaN or Inf",
    ):
        load_identified_secondary_path(
            path
        )


def test_multidimensional_coefficients_rejected(
    tmp_path,
) -> None:

    path = (
        tmp_path
        / "matrix.npy"
    )

    np.save(
        path,
        np.zeros(
            (
                2,
                3,
            ),
            dtype=np.float64,
        ),
    )

    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        load_identified_secondary_path(
            path
        )