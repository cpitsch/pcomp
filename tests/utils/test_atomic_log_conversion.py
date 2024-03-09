import pandas as pd
import pydantic
import pytest

from pcomp.utils import (
    constants,
    convert_atomic_eventlog_to_lifecycle_eventlog,
    convert_atomic_eventlog_to_start_timestamp_eventlog,
)
from tests.testing_utilities import Case, create_event_log


class AtomicCase(pydantic.BaseModel):
    caseid: str
    activities: list[str]
    timestamps: list[int]


def create_atomic_log(cases: list[AtomicCase]) -> pd.DataFrame:
    base_date = pd.Timestamp(2023, 1, 1)
    date_increment = pd.Timedelta(days=1)

    df_rows = [
        {
            constants.DEFAULT_TRACEID_KEY: case.caseid,
            constants.DEFAULT_NAME_KEY: activity,
            constants.DEFAULT_TIMESTAMP_KEY: base_date + (date_increment * timestamp),
        }
        for case in cases
        for (activity, timestamp) in zip(case.activities, case.timestamps)
    ]

    return pd.DataFrame(df_rows)


@pytest.fixture
def atomic_event_log() -> pd.DataFrame:
    return create_atomic_log(
        [
            AtomicCase(
                caseid="case1", activities=["a", "b", "c", "d"], timestamps=[1, 2, 3, 4]
            ),
        ]
    )


def test_atomic_to_lifecycle(atomic_event_log):
    converted = convert_atomic_eventlog_to_lifecycle_eventlog(atomic_event_log)

    expected = create_event_log(
        [
            Case(
                caseid="case1",
                activities=["a", "b", "c", "d"],
                lifecycle_transitions=["complete"] * 4,
                timestamps=[1, 2, 3, 4],
                event_instances=[f"case1:{i+1}" for i in range(4)],
            )
        ]
    )

    # Sort columns to compare regardless of order
    assert converted.sort_index(axis=1).equals(expected.sort_index(axis=1))


def test_atomic_to_start_timestamp_eventlog(atomic_event_log):
    converted = convert_atomic_eventlog_to_start_timestamp_eventlog(atomic_event_log)

    expected = create_event_log(
        [
            Case(
                caseid="case1",
                activities=["a", "b", "c", "d"],
                lifecycle_transitions=["DUMMY_TRANSITION"]
                * 4,  # Will be dropped, just needed to create the dataframe
                timestamps=[1, 2, 3, 4],
                start_timestamps=[1, 2, 3, 4],
            )
        ]
    ).drop(columns=[constants.DEFAULT_LIFECYCLE_KEY])

    # Sort columns to compare regardless of order
    assert converted.sort_index(axis=1).equals(expected.sort_index(axis=1))
