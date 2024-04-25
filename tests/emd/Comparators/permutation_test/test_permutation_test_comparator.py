import numpy as np

from pcomp.emd.Comparators.permutation_test.permutation_test_comparator import (
    compute_permutation_test_distribution,
    get_permutation_sample,
)


def test_get_permutation_sample_is_seeded():
    SAMPLE_RANGE = 100
    DIST_SIZE = 100
    SEED = 42

    sample_1 = get_permutation_sample(SAMPLE_RANGE, DIST_SIZE, SEED)
    sample_2 = get_permutation_sample(SAMPLE_RANGE, DIST_SIZE, SEED)

    assert (sample_1 == sample_2).all()


def test_get_permutation_sample_gives_correct_dimension():
    SAMPLE_RANGE = 100
    DIST_SIZE = 100

    sample = get_permutation_sample(SAMPLE_RANGE, DIST_SIZE)
    assert sample.shape == (DIST_SIZE, SAMPLE_RANGE)


def test_permutation_test_distribution_computation():
    population_0 = [0] * 10
    population_1 = [1] * 10

    cost_fn = lambda x, y: abs(x - y)  # 0 if same, 1 if different

    DIST_SIZE = 1  # 1 Permutation sample
    SEED = 42

    result = compute_permutation_test_distribution(
        population_0, population_1, cost_fn, DIST_SIZE, SEED, show_progress_bar=False
    )

    # Expected:
    # The permutation sample is the permutation of [0,...,19] with seed 1
    # In this case:
    #
    # [[15, 9, 14, 7, 12, 10, 6, 19, 3, 0, 16, 5, 11, 18, 2, 4, 17, 1, 13, 8]]
    #
    # Indices < 10 are 0s, >=10 are 1s, so effectively, the permutation is as follows:
    # permutation = [1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
    #
    # So:
    # sample_1 = [1, 0, 1, 0, 1, 1, 0, 1, 0, 0]
    # sample_2 = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
    #
    # This gives us the unique variants and relative frequencies:
    # variants_1 = [0, 1]; freqs_1 = [0.5, 0.5]
    # variants_2 = [0, 1]; freqs_2 = [0.5, 0.5] # Trivial example then...
    #
    # Which should trivially give an EMD of 0. (Bad example I guess)

    assert result == np.array([0.0], dtype=np.float_)

    # Now, using the same permutation as before, we choose the input populations such
    # that the permutation splits the variants exactly into the two samples. That means,
    # sample_1 consists of only 0s, sample_2 consists of only 1s

    total_input = np.empty(20, dtype=np.int_)
    permutation = np.array(
        [15, 9, 14, 7, 12, 10, 6, 19, 3, 0, 16, 5, 11, 18, 2, 4, 17, 1, 13, 8]
    )
    total_input[permutation[:10]] = 0
    total_input[permutation[10:]] = 1

    # Multiply cost_fn by 7 so the EMD isn't 1 since its probably more likely to
    # accidentally get EMD 1 than EMD 7
    DIFF_COST = 7
    cost_fn_2 = lambda x, y: DIFF_COST * cost_fn(x, y)

    result_2 = compute_permutation_test_distribution(
        total_input[:10].tolist(),
        total_input[10:].tolist(),
        cost_fn_2,
        DIST_SIZE,
        SEED,
        show_progress_bar=False,
    )

    # So like this, the permutation samples are:
    # sample_1 = [0] * 10
    # sample_2 = [1] * 10
    #
    # As such, the unique variants with their relative frequencies are:
    # variants_1 = [0], freqs_1 = [1]
    # variants_2 = [1], freqs_2 = [1]
    #
    # So intuitively, in the EMD transport problem, we have a source of 1 unit and a
    # sink of 1 unit, with a cost of DIFF_COST per unit transferred, so trivially, the
    # EMD is DIFF_COST (or more generally, the value of cost_fn(0, 1)).
    assert result_2 == np.array([cost_fn_2(0, 1)], dtype=np.float_)
