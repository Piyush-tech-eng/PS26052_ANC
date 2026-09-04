import numpy as np
import pytest

from anc.statistics import (
    solve_wiener_hopf,
)


def test_solves_known_wiener_system() -> None:
    """
    Construct a system with a known solution and verify
    that the solver recovers it.
    """

    R = np.array(
        [
            [2.0, 1.0],
            [1.0, 2.0],
        ]
    )

    expected_w = np.array(
        [1.0, -1.0]
    )

    p = R @ expected_w

    solution = solve_wiener_hopf(
        R,
        p,
    )

    assert np.allclose(
        solution,
        expected_w,
    )


def test_solution_satisfies_wiener_hopf_equation() -> None:
    R = np.array(
        [
            [4.0, 1.0, 0.5],
            [1.0, 4.0, 1.0],
            [0.5, 1.0, 4.0],
        ]
    )

    p = np.array(
        [1.0, 2.0, 3.0]
    )

    w = solve_wiener_hopf(
        R,
        p,
    )

    assert np.allclose(
        R @ w,
        p,
    )


def test_solution_has_correct_shape() -> None:
    R = np.eye(5)

    p = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0]
    )

    w = solve_wiener_hopf(
        R,
        p,
    )

    assert w.shape == (5,)


def test_identity_matrix_returns_p() -> None:
    R = np.eye(3)

    p = np.array(
        [2.0, -1.0, 5.0]
    )

    w = solve_wiener_hopf(
        R,
        p,
    )

    assert np.allclose(
        w,
        p,
    )


def test_rejects_non_square_matrix() -> None:
    R = np.ones(
        (2, 3)
    )

    p = np.array(
        [1.0, 2.0]
    )

    with pytest.raises(
        ValueError,
        match="square",
    ):
        solve_wiener_hopf(
            R,
            p,
        )


def test_rejects_vector_length_mismatch() -> None:
    R = np.eye(3)

    p = np.array(
        [1.0, 2.0]
    )

    with pytest.raises(
        ValueError,
        match="length must match",
    ):
        solve_wiener_hopf(
            R,
            p,
        )


def test_rejects_nonfinite_matrix() -> None:
    R = np.array(
        [
            [1.0, np.nan],
            [0.0, 1.0],
        ]
    )

    p = np.array(
        [1.0, 2.0]
    )

    with pytest.raises(
        ValueError,
        match="NaN or Inf",
    ):
        solve_wiener_hopf(
            R,
            p,
        )


def test_rejects_nonfinite_vector() -> None:
    R = np.eye(2)

    p = np.array(
        [1.0, np.inf]
    )

    with pytest.raises(
        ValueError,
        match="NaN or Inf",
    ):
        solve_wiener_hopf(
            R,
            p,
        )


def test_rejects_singular_matrix() -> None:
    R = np.array(
        [
            [1.0, 2.0],
            [2.0, 4.0],
        ]
    )

    p = np.array(
        [1.0, 2.0]
    )

    with pytest.raises(
        ValueError,
        match="singular",
    ):
        solve_wiener_hopf(
            R,
            p,
        )

