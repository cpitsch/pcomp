from typing import get_args

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pcomp.emd.core import (
    EMDBackend,
    StochasticLanguage,
    compute_distance_matrix,
    compute_emd,
    compute_emd_for_index_sample,
    emd,
    population_to_stochastic_language,
)
from pcomp.emd.distances.levenshtein import custom_postnormalized_levenshtein_distance
from pcomp.emd.extraction import BinnedServiceTimeTrace
from tests.conftest import assert_almost_equal


def test_emd_wrapper() -> None:
    costs = np.array([[0, 1], [1, 0]])

    available_backends: tuple[EMDBackend, ...] = get_args(EMDBackend)
    for backend in available_backends:
        assert (
            emd(np.array([0.5, 0.5]), np.array([0.5, 0.5]), costs, backend=backend)
            == 0.0
        )
        assert (
            emd(np.array([0.0, 1.0]), np.array([0.5, 0.5]), costs, backend=backend)
            == 0.5
        )


def test_population_to_stochastic_language():
    population = ["a"] * 4 + ["b"] * 3 + ["c"] + ["d"] * 2

    result = population_to_stochastic_language(population)

    # Different order would also be considered correct..
    expected = StochasticLanguage(
        variants=["a", "b", "c", "d"], frequencies=np.array([0.4, 0.3, 0.1, 0.2])
    )

    assert result.variants == expected.variants
    assert result.frequencies == pytest.approx(expected.frequencies)


def test_compute_emd_for_index_sample():
    # fmt: off
    dists = np.array(
        [
            [0,  0.5, 1  ],
            [1,   0,  0.5],
            [99, 999, 0  ],
        ]
    )
    # fmt: on
    index_sample = np.array([0, 1, 0])

    # The dists matrix will be projected to the first two rows
    # And the freqs will be along the lines of np.array([0.66666667, 0.33333333])
    assert compute_emd_for_index_sample(index_sample, dists, np.ones(3) / 3) == 1 / 3
    # Assertion works for these inputs, but for other inputs, floating point inaccuracies might make it fail


def test_compute_emd():
    stoch_lang_1 = StochasticLanguage([0, 1, 2], np.array([0.3, 0.4, 0.3]))
    stoch_lang_2 = StochasticLanguage([0, 1, 2], np.array([0.2, 0.8, 0.0]))

    def cost_fn(x, y):
        return float(abs(x - y))

    result_1 = compute_emd(
        stoch_lang_1,
        stoch_lang_2,
        cost_fn,
        backend="wasserstein",
        show_progress_bar=False,
    )
    result_2 = compute_emd(
        stoch_lang_1, stoch_lang_2, cost_fn, backend="pot", show_progress_bar=False
    )

    assert_almost_equal(result_1, 0.4, 1e-9)
    assert_almost_equal(
        result_1,
        result_2,
        1e-9,
        "wassertstein and pot should give approx. the same result",
    )


def test_emd_normal_example():
    """Test that the EMD without time (all times are 0) is the same as what the paper
    (Brockhoff et al. "Time-Aware Concept Drift Detection Using the Earth Mover's Distance")
    has as example for non-time EMD. The paper did not have any examples of EMD's *with*
    times, so we don't have a reference for this.
    """
    stoch_lang_1: StochasticLanguage[BinnedServiceTimeTrace] = StochasticLanguage(
        variants=[
            (("a", 0), ("b", 0), ("d", 0), ("f", 0)),
            (("a", 0), ("c", 0), ("f", 0)),
            (("a", 0), ("b", 0), ("e", 0), ("f", 0)),
        ],
        frequencies=np.array([0.5, 0.4, 0.1]),
    )
    stoch_lang_2: StochasticLanguage[BinnedServiceTimeTrace] = StochasticLanguage(
        variants=[
            (("a", 0), ("b", 0), ("d", 0), ("f", 0)),
            (("a", 0), ("c", 0), ("f", 0)),
            (("a", 0), ("b", 0), ("d", 0), ("e", 0), ("f", 0)),
        ],
        frequencies=np.array([0.5, 0.35, 0.15]),
    )

    stoch_lang_3: StochasticLanguage[BinnedServiceTimeTrace] = StochasticLanguage(
        variants=[
            (("a", 0), ("b", 0), ("d", 0), ("f", 0)),
            (("a", 0), ("c", 0), ("f", 0)),
            (("a", 0), ("b", 0), ("e", 0), ("f", 0)),
        ],
        frequencies=np.array([0.2, 0.7, 0.1]),
    )

    # Check that all available EMD backends return the correct result
    for backend in get_args(EMDBackend):
        # Assert almost equal due to floating point arithmetic. 10^-9 is a very reasonable delta.
        assert_almost_equal(
            compute_emd(
                stoch_lang_1,
                stoch_lang_2,
                custom_postnormalized_levenshtein_distance,
                backend=backend,
            ),
            0.05,
            1e-9,
        )
        assert_almost_equal(
            compute_emd(
                stoch_lang_1,
                stoch_lang_3,
                custom_postnormalized_levenshtein_distance,
                backend=backend,
            ),
            0.15,
            1e-9,
        )


def test_compute_distance_matrix():
    stoch_lang_1 = StochasticLanguage([0, 1, 2], np.array([0.3, 0.4, 0.3]))
    stoch_lang_2 = StochasticLanguage([0, 1, 2], np.array([0.2, 0.8, 0.0]))

    def cost_fn(x, y):
        return float(abs(x - y))

    result = compute_distance_matrix(
        stoch_lang_1.variants, stoch_lang_2.variants, cost_fn, show_progress_bar=False
    )

    # fmt: off
    expected = np.array([
        [0.0, 1.0, 2.0],
        [1.0, 0.0, 1.0],
        [2.0, 1.0, 0.0]
    ])
    # fmt: on
    assert_array_equal(result, expected)
