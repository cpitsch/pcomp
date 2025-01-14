import pytest

from pcomp.utils.constants import (
    DEFAULT_INSTANCE_KEY,
    DEFAULT_LIFECYCLE_KEY,
    DEFAULT_TIMESTAMP_KEY,
)
from pcomp.utils.preparation import ensure_start_timestamp_column
from tests.testing_utilities import Case, create_event_log


def test_lifecycle_log_conversion_instance_key_none(event_log):
    """
    Test without using instance key. Thus, always match a complete event to the first
    corresponding starting event.
    """

    result = ensure_start_timestamp_column(
        event_log.drop(columns=[DEFAULT_INSTANCE_KEY])
    ).sort_values(
        DEFAULT_TIMESTAMP_KEY
    )  # Sort to compare regardless of order
    expected = create_event_log(
        [
            Case(
                "case1",
                ["a", "c", "b", "d"],
                ["complete"] * 4,
                [3, 5, 6, 7],
                ["0", "0", "0", "0"],
                [1, 4, 2, 7],
            ),
            Case(
                "case2",
                ["b", "a", "a"],
                ["complete"] * 3,
                [10, 11, 12],
                ["0", "0", "1"],
                [8, 7, 9],
            ),
        ]
    )

    expected[DEFAULT_INSTANCE_KEY] = expected[DEFAULT_INSTANCE_KEY].astype("object")
    assert result.equals(expected)


def test_lifecycle_log_conversion_with_instance(event_log):
    """
    Same test case as above, but in the below example, the existence of the instance
    key changes which a is mapped to which
    """
    result = ensure_start_timestamp_column(event_log).sort_values(
        DEFAULT_TIMESTAMP_KEY
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


def test_lifecycle_log_conversion_no_lifecycle(event_log):
    """
    Test without using lifecycle information. Thus, all events are interpreteed as a complete
    event, meaning that all stat timestamps will be equal to the completion timestamps
    """
    result = ensure_start_timestamp_column(
        event_log.drop(columns=[DEFAULT_INSTANCE_KEY, DEFAULT_LIFECYCLE_KEY])
    ).sort_values(
        DEFAULT_TIMESTAMP_KEY
    )  # Sort to compare regardless of order

    expected = create_event_log(
        [
            Case(
                "case1",
                ["a", "b", "a", "c", "c", "b", "d"],
                ["complete"] * 7,
                [1, 2, 3, 4, 5, 6, 7],
                ["0", "0", "1", "0", "1", "1", "0"],
                # Start timestamps should be identical to completion
                [1, 2, 3, 4, 5, 6, 7],
            ),
            Case(
                "case2",
                ["a", "b", "a", "b", "a", "a"],
                ["complete"] * 6,
                [7, 8, 9, 10, 11, 12],
                ["0", "0", "1", "1", "2", "3"],
                # Start timestamps should be identical to completion
                [7, 8, 9, 10, 11, 12],
            ),
        ]
    )

    # Sort columns for comparison
    result = result.reindex(sorted(result.columns), axis=1)
    expected = expected.reindex(sorted(expected.columns), axis=1)

    expected[DEFAULT_INSTANCE_KEY] = expected[DEFAULT_INSTANCE_KEY].astype("object")

    assert result.equals(expected)


def test_invalid_log_multiple_matches(event_log_invalid_instances):
    """
    Test that if the instance id is used, but there are multiple start events with
    the same instance id, an error is raised
    """

    with pytest.raises(
        ValueError,
        match="Multiple starting events found for single complete event of instance id i1",
    ):
        ensure_start_timestamp_column(event_log_invalid_instances)
