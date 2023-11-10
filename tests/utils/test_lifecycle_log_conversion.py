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
    # event_instances: Optional[list[str]] = None
    event_instances: Optional[list[str]] = None


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

    return pd.DataFrame(
        {
            constants.DEFAULT_TRACEID_KEY: (["case1"] * len(case1.activities))
            + (["case2"] * len(case2.activities)),
            constants.DEFAULT_NAME_KEY: case1.activities + case2.activities,
            constants.DEFAULT_LIFECYCLE_KEY: case1.lifecycle_transitions
            + case2.lifecycle_transitions,
            constants.DEFAULT_TIMESTAMP_KEY: case1.timestamps + case2.timestamps,
            constants.DEFAULT_INSTANCE_KEY: case1.event_instances
            + case2.event_instances,
        }
    )


@pytest.fixture
def event_log_invalid_instances():
    case = Case(  # Two start a's with the same ID - should raise error
        "case",
        ["a", "b", "a", "b", "a", "a"],
        ["start", "start", "start", "complete", "complete", "complete"],
        [7, 8, 9, 10, 11, 12],
        ["i1", "i2", "i1", "i2", "i1", "i1"],
    )

    return pd.DataFrame(
        {
            constants.DEFAULT_TRACEID_KEY: [case.caseid] * len(case.activities),
            constants.DEFAULT_NAME_KEY: case.activities,
            constants.DEFAULT_LIFECYCLE_KEY: case.lifecycle_transitions,
            constants.DEFAULT_TIMESTAMP_KEY: case.timestamps,
            constants.DEFAULT_INSTANCE_KEY: case.event_instances,
        }
    )


def test_lifecycle_log_conversion_instance_key_none(event_log):
    """Test without using instance key, so pass in instance_key=None. Thus, always match a complete event to the first corresponding starting event"""

    result = convert_lifecycle_eventlog_to_start_timestamp_eventlog(
        event_log, instance_key=None
    ).sort_values(
        constants.DEFAULT_TIMESTAMP_KEY
    )  # Sort to compare regardless of order
    expected = pd.DataFrame(
        {
            constants.DEFAULT_TRACEID_KEY: ["case1"] * 4 + ["case2"] * 3,
            constants.DEFAULT_NAME_KEY: ["a", "c", "b", "d"] + ["b", "a", "a"],
            constants.DEFAULT_LIFECYCLE_KEY: ["complete"] * 4 + ["complete"] * 3,
            constants.DEFAULT_TIMESTAMP_KEY: [3, 5, 6, 7] + [10, 11, 12],
            constants.DEFAULT_INSTANCE_KEY: ["i1_1", "i1_3", "i1_2", "i1_4"]
            + ["i2_2", "i2_3", "i2_1"],
            constants.DEFAULT_START_TIMESTAMP_KEY: [1, 4, 2, 7] + [8, 7, 9],
        }
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
    expected = pd.DataFrame(
        {
            constants.DEFAULT_TRACEID_KEY: ["case1"] * 4 + ["case2"] * 3,
            constants.DEFAULT_NAME_KEY: ["a", "c", "b", "d"] + ["b", "a", "a"],
            constants.DEFAULT_LIFECYCLE_KEY: ["complete"] * 4 + ["complete"] * 3,
            constants.DEFAULT_TIMESTAMP_KEY: [3, 5, 6, 7] + [10, 11, 12],
            constants.DEFAULT_START_TIMESTAMP_KEY: [1, 4, 2, 7] + [8, 7, 9],
        }
    )

    assert result.equals(expected)


def test_lifecycle_log_conversion_with_instance(event_log):
    """Same test case as above, but in the below example, the existence of the instance key changes which a is mapped to which"""
    result = convert_lifecycle_eventlog_to_start_timestamp_eventlog(
        event_log
    ).sort_values(
        constants.DEFAULT_TIMESTAMP_KEY
    )  # Sort to compare regardless of order
    expected = pd.DataFrame(
        {
            constants.DEFAULT_TRACEID_KEY: ["case1"] * 4 + ["case2"] * 3,
            constants.DEFAULT_NAME_KEY: ["a", "c", "b", "d"] + ["b", "a", "a"],
            constants.DEFAULT_LIFECYCLE_KEY: ["complete"] * 4 + ["complete"] * 3,
            constants.DEFAULT_TIMESTAMP_KEY: [3, 5, 6, 7] + [10, 11, 12],
            constants.DEFAULT_INSTANCE_KEY: ["i1_1", "i1_3", "i1_2", "i1_4"]
            + ["i2_2", "i2_3", "i2_1"],
            constants.DEFAULT_START_TIMESTAMP_KEY: [1, 4, 2, 7] + [8, 9, 7],
        }
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
