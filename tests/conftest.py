from copy import deepcopy

import pandas as pd
import pytest

from tests.testing_utilities import Case, create_event_log


@pytest.fixture
def event_log():
    # d completes without a corresponding start event
    #  --> Should use the timestamp of the "complete" event as start_timestamp
    case1 = Case(
        "case1",
        ["a", "b", "a", "c", "c", "b", "d"],
        [
            "start",  # start a
            "start",  # start b
            "complete",  # complete a
            "start",  # start c
            "complete",  # complete c
            "complete",  # complete b
            "complete",  # complete d
        ],
        [1, 2, 3, 4, 5, 6, 7],
        ["i1_1", "i1_2", "i1_1", "i1_3", "i1_3", "i1_2", "i1_4"],
    )

    # a completes while there are multiple started a's --> Ambiguity
    # --> If no instance key is given, or it isn't present in the log, the first started shall be used
    # --> If an instance key is given and present in the log, the assignment is not ambiguous, and here the first completion of a belongs to the second start of a
    case2 = Case(
        "case2",
        ["a", "b", "a", "b", "a", "a"],
        ["start", "start", "start", "complete", "complete", "complete"],
        [7, 8, 9, 10, 11, 12],
        # The first complete a belongs to the second started a
        ["i2_1", "i2_2", "i2_3", "i2_2", "i2_3", "i2_1"],
    )

    return create_event_log([case1, case2])


@pytest.fixture
def event_log_invalid_instances():
    return create_event_log(
        [
            Case(  # Two start a's with the same ID - should raise error
                "case",
                ["a", "b", "a", "b", "a", "a"],
                ["start", "start", "start", "complete", "complete", "complete"],
                [7, 8, 9, 10, 11, 12],
                ["i1", "i2", "i1", "i2", "i1", "i1"],
            )
        ]
    )


def create_large_log(seed_cases: list[Case], num_repeats: int) -> pd.DataFrame:
    """Create a large event log by repeating the given cases multiple times with different case ids"""
    cases: list[Case] = []
    for i in range(num_repeats):
        altered_cases = [deepcopy(case) for case in seed_cases]
        for case in altered_cases:
            case.caseid = f"{case.caseid}_repetition_{i}"
        cases += altered_cases
    return create_event_log(cases)


@pytest.fixture
def large_event_log():
    return create_large_log(
        [
            Case(
                "case1",
                ["a", "b", "a", "c", "c", "b", "d"],
                [
                    "start",  # start a
                    "start",  # start b
                    "complete",  # complete a
                    "start",  # start c
                    "complete",  # complete c
                    "complete",  # complete b
                    "complete",  # complete d
                ],
                [1, 2, 3, 4, 5, 6, 7],
                ["i1_1", "i1_2", "i1_1", "i1_3", "i1_3", "i1_2", "i1_4"],
            ),
            Case(
                "case2",
                ["a", "b", "a", "b", "a", "a"],
                ["start", "start", "start", "complete", "complete", "complete"],
                [7, 8, 9, 10, 11, 12],
                # The first complete a belongs to the second started a
                ["i2_1", "i2_2", "i2_3", "i2_2", "i2_3", "i2_1"],
            ),
        ],
        100,
    )  # 100 * 2 cases


@pytest.fixture
def simple_event_log():
    case = Case(
        caseid="case1",
        activities=["a", "b", "c", "b", "c", "d"],
        lifecycle_transitions=[
            "complete",
            "start",
            "start",
            "complete",
            "complete",
            "complete",
        ],
        timestamps=[1, 2, 2, 3, 3, 4],
        event_instances=["i1", "i2", "i3", "i2", "i3", "i4"],
    )
    return create_event_log([case])


def assert_almost_equal(a, b, delta, msg: str | None = None):
    if msg is None:
        assert abs(a - b) < delta
    else:
        assert abs(a - b) < delta, msg
