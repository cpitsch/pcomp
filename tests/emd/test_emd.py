from pcomp.emd.emd import (
    postNormalizedWeightedLevenshteinDistance,
    calc_timing_emd,
    # weightedLevenshteinDistance,
)


def test_postnormalized_weighted_lev_distance():
    trace1 = (("a", 1), ("b", 1), ("c", 2), ("d", 3))
    trace2 = (("a", 1), ("c", 2), ("b", 1), ("d", 1))
    # Solution:
    #   1) Match (a,1) and (a,1) with cost 0
    #   2) Delete (b,1) with cost 2
    #   3) Match (c,2) and (c,2) with cost 0
    #   4) Insert a (b,1) with cost 2
    #   5) Match (d,3) and (d,1) with cost 2
    # Total cost is 6
    # Alternatively: Match, Rename, Rename, Match => 6

    result = postNormalizedWeightedLevenshteinDistance(
        trace1,
        trace2,
        rename_cost=lambda x, y: 1,
        insertion_deletion_cost=lambda x: 1,
        cost_time_match_rename=lambda x, y: abs(x - y),
        cost_time_insert_delete=lambda x: x,
    )
    assert result == 1.5  # 6 / 4


# def test_weighted_lev_distance_example():
#     """Test the post-normalized Levenshtein distance with the example from the paper."""
#     trace1 = (("a", 1), ("b", 4))
#     trace2 = (("a", 4), ("b", 2))
#     trace3 = (("b", 2), ("a", 4))

#     costs = {
#         "rename_cost": lambda x, y: 2, # f_r = 2
#         "insertion_deletion_cost": lambda x: 2, #f_id = 2
#         "cost_time_match_rename": lambda x, y: abs(x - y),
#         "cost_time_insert_delete": lambda x: x,
#     }

#     # assert weightedLevenshteinDistance(trace1, trace2, **costs) == 5
#     # assert weightedLevenshteinDistance(trace1, trace3, **costs) == 4


def test_emd_normal_example():
    """Test that the EMD without time (all times are 0) is the same as what the paper has as example for non-time EMD.
    The paper did not have any examples of EMD's *with* times, so we don't have a reference for this.

    """
    distribution1 = [
        ((("a", 0), ("b", 0), ("d", 0), ("f", 0)), 50 / 100),
        ((("a", 0), ("c", 0), ("f", 0)), 40 / 100),
        ((("a", 0), ("b", 0), ("e", 0), ("f", 0)), 10 / 100),
    ]
    distribution2 = [
        ((("a", 0), ("b", 0), ("d", 0), ("f", 0)), 50 / 100),
        ((("a", 0), ("c", 0), ("f", 0)), 35 / 100),
        ((("a", 0), ("b", 0), ("d", 0), ("e", 0), ("f", 0)), 15 / 100),
    ]
    distribution3 = [
        ((("a", 0), ("b", 0), ("d", 0), ("f", 0)), 20 / 100),
        ((("a", 0), ("c", 0), ("f", 0)), 70 / 100),
        ((("a", 0), ("b", 0), ("e", 0), ("f", 0)), 10 / 100),
    ]

    # Assert almost equal due to floating point arithmetic. 10^-9 is a very reasonable delta.
    assert_almost_equal(calc_timing_emd(distribution1, distribution2), 0.05, 1e-9)
    assert_almost_equal(calc_timing_emd(distribution1, distribution3), 0.15, 1e-9)


def assert_almost_equal(a, b, delta):
    assert abs(a - b) < delta
