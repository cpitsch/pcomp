from pcomp.utils import (
    convert_lifecycle_eventlog_to_start_timestamp_eventlog,
    constants,
)
import pandas as pd
from dataclasses import dataclass
from typing import Optional
import pytest


@dataclass
class Case:
    caseid: str
    activities: list[str]
    lifecycle_transitions: list[str]
    timestamps: list[int]
    event_instances: Optional[list[str]] = None
    start_timestamps: Optional[list[int]] = None


def create_event_log(cases: list[Case]) -> pd.DataFrame:
    base_date = pd.Timestamp(2023, 1, 1)
    date_increment = pd.Timedelta(days=1)

    def drop_none(d: dict) -> dict:
        return {k: v for k, v in d.items() if v is not None}

    df_rows = [
        drop_none(
            {
                constants.DEFAULT_TRACEID_KEY: case.caseid,
                constants.DEFAULT_NAME_KEY: activity,
                constants.DEFAULT_LIFECYCLE_KEY: transition,
                constants.DEFAULT_TIMESTAMP_KEY: base_date
                + (date_increment * timestamp),
                constants.DEFAULT_INSTANCE_KEY: case.event_instances[i]
                if case.event_instances is not None
                else None,
                constants.DEFAULT_START_TIMESTAMP_KEY: base_date
                + (date_increment * case.start_timestamps[i])
                if case.start_timestamps is not None
                else None,
            }
        )
        for case in cases
        for i, (activity, transition, timestamp) in enumerate(
            zip(case.activities, case.lifecycle_transitions, case.timestamps)
        )
    ]
    df = pd.DataFrame(df_rows)
    return df


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
        altered_cases = [case for case in seed_cases]
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


def test_lifecycle_log_conversion_instance_key_none(event_log):
    """Test without using instance key, so pass in instance_key=None. Thus, always match a complete event to the first corresponding starting event"""

    result = convert_lifecycle_eventlog_to_start_timestamp_eventlog(
        event_log, instance_key=None
    ).sort_values(
        constants.DEFAULT_TIMESTAMP_KEY
    )  # Sort to compare regardless of order
    expected = create_event_log(
        [
            Case(
                "case1",
                ["a", "c", "b", "d"],
                ["complete"] * 4,
                [3, 5, 6, 7],
                ["i1_1", "i1_3", "i1_2", "i1_4"],
                [1, 4, 2, 7],
            ),
            Case(
                "case2",
                ["b", "a", "a"],
                ["complete"] * 3,
                [10, 11, 12],
                ["i2_2", "i2_3", "i2_1"],
                [8, 7, 9],
            ),
        ]
    )

    assert result.equals(expected)


def test_lifecycle_log_conversion_no_instance_key(event_log):
    """Test without using instance key. Similar to `test_lifecycle_log_conversion_instance_key_none`,
    but here the instance key is not present in the log, and we don't explicitly pass `None` in
    """
    dropped_log = event_log.drop(columns=[constants.DEFAULT_INSTANCE_KEY])
    result = convert_lifecycle_eventlog_to_start_timestamp_eventlog(
        dropped_log
    ).sort_values(
        constants.DEFAULT_TIMESTAMP_KEY
    )  # Sort to compare regardless of order
    expected = create_event_log(
        [
            Case(
                "case1",
                ["a", "c", "b", "d"],
                ["complete"] * 4,
                [3, 5, 6, 7],
                start_timestamps=[1, 4, 2, 7],
            ),
            Case(
                "case2",
                ["b", "a", "a"],
                ["complete"] * 3,
                [10, 11, 12],
                start_timestamps=[8, 7, 9],
            ),
        ]
    )

    assert result.equals(expected)


def test_lifecycle_log_conversion_with_instance(event_log):
    """Same test case as above, but in the below example, the existence of the instance key changes which a is mapped to which"""
    result = convert_lifecycle_eventlog_to_start_timestamp_eventlog(
        event_log
    ).sort_values(
        constants.DEFAULT_TIMESTAMP_KEY
    )  # Sort to compare regardless of order
    expected = create_event_log(
        [
            Case(
                "case1",
                ["a", "c", "b", "d"],
                ["complete"] * 4,
                [3, 5, 6, 7],
                ["i1_1", "i1_3", "i1_2", "i1_4"],
                [1, 4, 2, 7],
            ),
            Case(
                "case2",
                ["b", "a", "a"],
                ["complete"] * 3,
                [10, 11, 12],
                ["i2_2", "i2_3", "i2_1"],
                [8, 9, 7],
            ),
        ]
    )
    assert result.equals(expected)


def test_invalid_log_multiple_matches(event_log_invalid_instances):
    """Test that if the instance id is used, but there are multiple start events with the same instance id, an error is raised"""

    with pytest.raises(
        ValueError,
        match="Multiple starting events found for single complete event of instance id i1",
    ):
        convert_lifecycle_eventlog_to_start_timestamp_eventlog(
            event_log_invalid_instances
        )
