from dataclasses import dataclass
from typing import Optional

import pandas as pd

from pcomp.utils import constants


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
