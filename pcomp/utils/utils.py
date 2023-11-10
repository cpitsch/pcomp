from pm4py import read_xes
from pm4py.utils import sample_cases

import pandas as pd
from pandas import DataFrame

from . import constants


def import_log(path: str, show_progress_bar: bool = False) -> DataFrame:
    return read_xes(path, show_progress_bar=show_progress_bar)


def log_len(log: DataFrame, traceid_key: str = constants.DEFAULT_TRACEID_KEY) -> int:
    return len(log[traceid_key].unique())


def split_log_cases(
    log: DataFrame, frac: float, traceid_key: str = constants.DEFAULT_TRACEID_KEY
) -> tuple[DataFrame, DataFrame]:
    num_cases = log_len(log)
    num_sample_cases = int(num_cases * frac)

    sample1 = sample_cases(log, num_sample_cases, case_id_key=traceid_key)
    sample2 = log[~log[traceid_key].isin(sample1[traceid_key].unique())]

    return sample1, sample2


def convert_lifecycle_eventlog_to_start_timestamp_eventlog(
    log: DataFrame,
    traceid_key: str = constants.DEFAULT_TRACEID_KEY,
    activity_key: str = constants.DEFAULT_NAME_KEY,
    timestamp_key: str = constants.DEFAULT_TIMESTAMP_KEY,
    lifecycle_key: str = constants.DEFAULT_LIFECYCLE_KEY,
    instance_key: str = constants.DEFAULT_INSTANCE_KEY,
) -> DataFrame:
    """Convert an event log consisting of events with lifecycle information and timestamp to an event log with start_timestamp and complete_timestamp.

    So if before there was a "start" event and a "complete" event with timestamps, this is combined into one event with "start_timestamp" as the timestamp of the "start" event and "time:timestamp" as the timestamp of the "complete" event.

    Args:
        log (DataFrame): The event log.
        traceid_key (str, optional): _description_. Defaults to "case:concept:name".
        activity_key (str, optional): _description_. Defaults to "concept:name".
        timestamp_key (str, optional): _description_. Defaults to "time:timestamp".
        lifecycle_key (str, optional): _description_. Defaults to "lifecycle:transition".
        instance_key (str, optional): The key for the id to tell different executions of activities apart. Defaults to "concept:instance". If this column is not present, \
                                        a complete event is always matched to the first found start event in the order of the case.
    Returns:
        DataFrame: The converted event log
    """
    use_instance = instance_key is not None and instance_key in log.columns
    new_df = log.iloc[:0].copy()  # Create empty copy of log to append altered cases to
    new_df[constants.DEFAULT_START_TIMESTAMP_KEY] = pd.Series(
        dtype=log[timestamp_key].dtype
    )  # Create start timestamp column with the same dtype as the timestamp column
    for _, case_df in log.sort_values(timestamp_key).groupby(traceid_key):
        start_events = case_df[case_df[lifecycle_key] == "start"].copy()
        end_events = case_df[
            case_df[lifecycle_key].isin(["complete", "ate_abort"])
        ].copy()

        # Match each complete event to its start event (if there is one), otherwise use the timestamp of the complete event
        ## Afterwards, the "end" events will remain, with a new attribute "start_timestamp" that contains the timestamp of the corresponding start event (or their own timestamp if there is no start event --> Instant execution)
        def find_start_timestamp(
            row: pd.Series, start_events: DataFrame, use_instance: bool
        ):
            matching_events = start_events[
                start_events[activity_key] == row[activity_key]
            ]
            if use_instance:
                matching_events = matching_events[
                    matching_events[instance_key] == row[instance_key]
                ]
            if len(matching_events) == 0:
                return row[timestamp_key]
            elif use_instance and len(matching_events) > 1:
                raise ValueError(
                    f"Multiple starting events found for single complete event of instance id {row[instance_key]}"
                )
            else:
                matched_event = matching_events.sort_values(timestamp_key).iloc[0]
                # Remove the matched event from the start events, so it cannot be matched again
                start_events.drop(matched_event.name, inplace=True)
                return matched_event[timestamp_key]

        end_events[constants.DEFAULT_START_TIMESTAMP_KEY] = end_events.apply(
            lambda row: find_start_timestamp(row, start_events, use_instance), axis=1
        )
        new_df = pd.concat([new_df, end_events])
    return new_df.reset_index(drop=True)
