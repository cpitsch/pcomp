import numpy as np

from pcomp.emd.core import (
    compute_emd_for_index_sample,
    compute_emd_for_sample,
    compute_time_distance_component,
    emd,
    population_to_stochastic_language,
)


def test_emd_wrapper():
    hist_1 = np.array([0.5, 0.5])
    hist_2 = np.array([0.5, 0.5])

    costs = np.array([[0, 1], [1, 0]])

    assert emd(np.array([0.5, 0.5]), np.array([0.5, 0.5]), costs) == 0.0
    assert emd(np.array([0, 1]), np.array([0.5, 0.5]), costs) == 0.5


def test_population_to_stochastic_language():
    population = ["a"] * 4 + ["b"] * 3 + ["c"] + ["d"] * 2

    result = population_to_stochastic_language(population)

    assert sorted(result, key=lambda x: x[0]) == [
        ("a", 0.4),
        ("b", 0.3),
        ("c", 0.1),
        ("d", 0.2),
    ]


def test_compute_emd_for_sample():
    np.random.seed(0)

    # fmt: off
    dists = np.array(
        [
            [0,  0.5, 1  ],
            [1,   0,  0.5],
            [99, 999, 0  ],
        ]
    )
    # fmt: on

    # With seed 0, the function will sample np.array([0, 1, 0])
    # So the dists matrix will be projected to the first two rows
    # And the freqs will be along the lines of np.array([0.66666667, 0.33333333])
    assert compute_emd_for_sample(dists, np.ones(3) / 3, resample_size=3) == 1 / 3
    # Assertion works for these inputs, but for other inputs, floating point inaccuracies might make it fail


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

    # Same assertion as above, but here we, of course, don't have
    # randomness due to being given the indices
    assert (
        compute_emd_for_index_sample(np.array([0, 1, 0]), dists, np.ones(3) / 3)
        == 1 / 3
    )


def test_compute_time_distance_component_simple():
    trace_1 = (("a", 1), ("b", 1), ("c", 2), ("d", 3))
    trace_2 = (("a", 1), ("c", 2), ("b", 1), ("d", 1))

    # Just d incurs a cost
    assert compute_time_distance_component(trace_1, trace_2) == 2.0


def test_compute_time_distance_component_complex():
    trace_1 = (("a", 5), ("b", 1), ("a", 1), ("c", 5), ("c", 1), ("d", 1))
    trace_2 = (("a", 1), ("b", 1), ("a", 4), ("c", 1), ("d", 1))

    # Everything is matched
    assert (
        compute_time_distance_component(trace_1, trace_2) == 1 + 5
    )  # Costs for a (5-4) and c insert of time 5
