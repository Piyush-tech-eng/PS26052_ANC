import numpy as np
import pytest

from anc.adaptive import (
    NLMSFilter,
    NLMSResult,
)


def test_nlms_returns_expected_result_type() -> None:

    nlms = NLMSFilter(
        filter_length=2,
        step_size=0.1,
    )

    result = nlms.adapt(
        np.array(
            [1.0, 2.0, 3.0],
        ),
        np.array(
            [1.0, 2.0, 3.0],
        ),
    )

    assert isinstance(
        result,
        NLMSResult,
    )


def test_nlms_result_shapes() -> None:

    reference = np.array(
        [1.0, 2.0, 3.0, 4.0],
    )

    desired = np.array(
        [2.0, 4.0, 6.0, 8.0],
    )

    filter_length = 3

    nlms = NLMSFilter(
        filter_length=filter_length,
        step_size=0.1,
    )

    result = nlms.adapt(
        reference,
        desired,
    )

    assert result.output.shape == (
        len(reference),
    )

    assert result.error.shape == (
        len(reference),
    )

    assert result.squared_error.shape == (
        len(reference),
    )

    assert (
        result.coefficient_history.shape
        == (
            len(reference),
            filter_length,
        )
    )

    assert result.final_coefficients.shape == (
        filter_length,
    )


def test_nlms_squared_error_matches_error() -> None:

    reference = np.array(
        [1.0, -1.0, 2.0],
    )

    desired = np.array(
        [0.5, 1.0, -0.5],
    )

    nlms = NLMSFilter(
        filter_length=2,
        step_size=0.1,
    )

    result = nlms.adapt(
        reference,
        desired,
    )

    np.testing.assert_allclose(
        result.squared_error,
        result.error ** 2,
    )


def test_nlms_first_update_is_correct() -> None:

    reference = np.array(
        [2.0],
    )

    desired = np.array(
        [4.0],
    )

    step_size = 0.5

    epsilon = 1e-8

    nlms = NLMSFilter(
        filter_length=1,
        step_size=step_size,
        epsilon=epsilon,
    )

    result = nlms.adapt(
        reference,
        desired,
    )

    # Initial coefficient is zero.
    #
    # output = 0
    # error = desired - output = 4
    #
    # input energy = 2^2 = 4
    #
    # normalized step =
    #     0.5 / (4 + epsilon)
    #
    # coefficient update =
    #     normalized_step
    #     * error
    #     * input
    #
    expected_coefficient = (
        step_size
        / (
            epsilon
            + 4.0
        )
        * 4.0
        * 2.0
    )

    np.testing.assert_allclose(
        result.final_coefficients,
        np.array(
            [expected_coefficient],
        ),
    )


def test_nlms_learns_simple_gain() -> None:

    num_samples = 1000

    reference = np.ones(
        num_samples,
        dtype=np.float64,
    )

    desired = 2.0 * reference

    nlms = NLMSFilter(
        filter_length=1,
        step_size=0.1,
    )

    result = nlms.adapt(
        reference,
        desired,
    )

    assert abs(
        result.final_coefficients[0]
        - 2.0
    ) < 1e-3

    initial_error_power = np.mean(
        result.squared_error[
            :100
        ]
    )

    final_error_power = np.mean(
        result.squared_error[
            -100:
        ]
    )

    assert (
        final_error_power
        < initial_error_power
    )


def test_nlms_reset_sets_coefficients_to_zero() -> None:

    nlms = NLMSFilter(
        filter_length=3,
        step_size=0.1,
    )

    nlms.adapt(
        np.array(
            [1.0, 2.0, 3.0],
        ),
        np.array(
            [2.0, 4.0, 6.0],
        ),
    )

    assert not np.allclose(
        nlms.coefficients,
        0.0,
    )

    nlms.reset()

    np.testing.assert_allclose(
        nlms.coefficients,
        np.zeros(
            3,
        ),
    )


@pytest.mark.parametrize(
    (
        "filter_length",
        "step_size",
        "exception_type",
    ),
    [
        (
            0,
            0.1,
            ValueError,
        ),
        (
            -1,
            0.1,
            ValueError,
        ),
        (
            2,
            0.0,
            ValueError,
        ),
        (
            2,
            -0.1,
            ValueError,
        ),
    ],
)
def test_nlms_invalid_parameters_rejected(
    filter_length: int,
    step_size: float,
    exception_type: type[Exception],
) -> None:

    with pytest.raises(
        exception_type
    ):

        NLMSFilter(
            filter_length=filter_length,
            step_size=step_size,
        )


def test_nlms_non_finite_parameters_rejected() -> None:

    with pytest.raises(
        ValueError
    ):

        NLMSFilter(
            filter_length=2,
            step_size=np.nan,
        )

    with pytest.raises(
        ValueError
    ):

        NLMSFilter(
            filter_length=2,
            step_size=np.inf,
        )

    with pytest.raises(
        ValueError
    ):

        NLMSFilter(
            filter_length=2,
            step_size=0.1,
            epsilon=0.0,
        )


def test_nlms_mismatched_signal_lengths_rejected() -> None:

    nlms = NLMSFilter(
        filter_length=2,
        step_size=0.1,
    )

    with pytest.raises(
        ValueError
    ):

        nlms.adapt(
            np.array(
                [1.0, 2.0],
            ),
            np.array(
                [1.0],
            ),
        )


def test_nlms_empty_signal_rejected() -> None:

    nlms = NLMSFilter(
        filter_length=2,
        step_size=0.1,
    )

    with pytest.raises(
        ValueError
    ):

        nlms.adapt(
            np.array(
                [],
            ),
            np.array(
                [],
            ),
        )


def test_nlms_non_finite_input_rejected() -> None:

    nlms = NLMSFilter(
        filter_length=2,
        step_size=0.1,
    )

    with pytest.raises(
        ValueError
    ):

        nlms.adapt(
            np.array(
                [1.0, np.nan],
            ),
            np.array(
                [1.0, 2.0],
            ),
        )

    with pytest.raises(
        ValueError
    ):

        nlms.adapt(
            np.array(
                [1.0, 2.0],
            ),
            np.array(
                [1.0, np.inf],
            ),
        )