from typing import get_args

import numpy as np
import pytest

from pcomp.emd.core import (
    EMDBackend,
    StochasticLanguage,
    compute_emd_for_index_sample,
    emd,
    population_to_stochastic_language,
)


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
