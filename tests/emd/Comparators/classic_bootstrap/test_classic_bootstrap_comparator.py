import numpy as np
import pytest

from pcomp.emd.comparators.classic_bootstrap.classic_bootstrap_comparator import (
    bootstrap_emd_population_classic,
)


def test_bootstrap_dist_computation():
    population_0 = [0] * 10
    population_1 = [1] * 10

    cost_fn = lambda x, y: abs(x - y)  # 0 if same, 1 if different

    DIST_SIZE = 2
    SEED = 42

    result = bootstrap_emd_population_classic(
        population_0, population_1, cost_fn, DIST_SIZE, SEED, show_progress_bar=False
    )

    # Bootstrap sampling should boil down to:
    # gen.choice(2, (1,20), replace=True, p=[0.5, 0.5])
    # Since: 2 distinct values, each with relative frequency 0.5, total pop size 20
    #
    # This gives the result:
    # [[1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1]
    #  [1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1]]
    #
    # So in Iteration 1:
    # sample_1 = [1, 0, 1, 1, 0, 1, 1, 1, 0, 0]
    # sample_2 = [0, 1, 1, 1, 0, 0, 1, 0, 1, 1]
    #
    # This boils down to:
    # variants_1 = [0,1]; freqs_1 = [0.4, 0.6]
    # variants_2 = [0,1]; freqs_2 = [0.4, 0.6] # Again a trivial example...
    #
    # Which trivially yields EMD 0
    #
    # Iteration 2:
    # sample_1 = [1, 0, 1, 1, 1, 0, 0, 0, 0, 1]
    # sample_2 = [1, 1, 0, 0, 0, 0, 0, 0, 0, 1]
    #
    # This boils down to:
    # variants_1 = [0,1]; freqs_1 = [0.5, 0.5]
    # variants_2 = [0,1]; freqs_2 = [0.7, 0.3]
    #
    # Heuristically, after mapping all cost 0 values, we are left with
    # freqs_1 = [0.0, 0.2]
    # freqs_2 = [0.2, 0.0]
    #
    # So we map 0.2 units from 1 to 0 at cost 1 per unit, i.e., incur a cost of 0.2
    # Thus, the second EMD is 0.2

    assert result == pytest.approx(np.array([0.0, 0.2], dtype=np.float_))
