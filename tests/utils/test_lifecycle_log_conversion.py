from dataclasses import dataclass
from typing import Optional

import pandas as pd
import pytest

from pcomp.utils import (
    constants,
    convert_lifecycle_eventlog_to_start_timestamp_eventlog,
)
from tests.testing_utilities import Case, create_event_log


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
