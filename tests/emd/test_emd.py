from datetime import timedelta
from typing import get_args

import numpy as np

from pcomp.emd.core import EMDBackend, StochasticLanguage, compute_emd
from pcomp.emd.emd import (
    custom_postnormalized_levenshtein_distance,
    extract_service_time_traces,
    extract_traces,
    post_normalized_weighted_levenshtein_distance,
)
from pcomp.utils import ensure_start_timestamp_column

#### Test Extraction ####


def test_service_time_trace_extraction_simple_event_log(simple_event_log):
    internal_log = ensure_start_timestamp_column(simple_event_log)

    service_time_traces = extract_service_time_traces(
        internal_log,
        activity_key="concept:name",
        start_time_key="start_timestamp",
        end_time_key="time:timestamp",
    )

    day = timedelta(days=1).total_seconds()
    expected = [(("a", 0 * day), ("b", 1 * day), ("c", 1 * day), ("d", 0 * day))]
    assert service_time_traces == expected


def test_service_time_trace_extraction_event_log(event_log):
    internal_log = ensure_start_timestamp_column(event_log)

    service_time_traces = extract_service_time_traces(
        internal_log,
        activity_key="concept:name",
        start_time_key="start_timestamp",
        end_time_key="time:timestamp",
    )

    day = timedelta(days=1).total_seconds()

    # If we order by completion timestamp, we get b, a_2, a_1 for the second case
    expected = [
        (("a", 2 * day), ("c", 1 * day), ("b", 4 * day), ("d", 0 * day)),
        (("b", 2 * day), ("a", 2 * day), ("a", 5 * day)),
    ]
    assert service_time_traces == expected


def test_control_flow_extraction_simple_event_log(simple_event_log):
    # Filters to only retain complete events
    traces = extract_traces(simple_event_log, filter_complete_lifecycle=True)
    expected = [("a", "b", "c", "d")]
    assert traces == expected


def test_control_flow_extraction_event_log(event_log):
    # Filters to only retain complete events
    traces = extract_traces(event_log, filter_complete_lifecycle=True)
    expected = [
        ("a", "c", "b", "d"),
        ("b", "a", "a"),
    ]
    assert traces == expected


#### Test Edit Distances


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

    result = post_normalized_weighted_levenshtein_distance(
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
    stoch_lang_1 = StochasticLanguage(
        variants=[
            (("a", 0), ("b", 0), ("d", 0), ("f", 0)),
            (("a", 0), ("c", 0), ("f", 0)),
            (("a", 0), ("b", 0), ("e", 0), ("f", 0)),
        ],
        frequencies=np.array([0.5, 0.4, 0.1]),
    )
    stoch_lang_2 = StochasticLanguage(
        variants=[
            (("a", 0), ("b", 0), ("d", 0), ("f", 0)),
            (("a", 0), ("c", 0), ("f", 0)),
            (("a", 0), ("b", 0), ("d", 0), ("e", 0), ("f", 0)),
        ],
        frequencies=np.array([0.5, 0.35, 0.15]),
    )

    stoch_lang_3 = StochasticLanguage(
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


def assert_almost_equal(a, b, delta):
    assert abs(a - b) < delta
