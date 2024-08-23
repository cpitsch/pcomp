import logging

import pandas as pd
from pandas import DataFrame
from pm4py import read_xes  # type: ignore
from pm4py.objects.log.util import dataframe_utils
from tqdm.auto import tqdm

from . import constants


def import_log(
    path: str, show_progress_bar: bool = False, variant: str | None = None
) -> DataFrame:
    """Import an event log using pm4py.
    This function is a wrapper around pm4py.read_xes.

    Args:
        path (str): The path to the event log file.
        show_progress_bar (bool, optional): Should a progress bar be shown for the impor?
            Defaults to False.
        variant (str, optional): The pm4py import variant to use. See `pm4py.read_xes`.
            Defaults to None (use the pm4py default).

    Returns:
        DataFrame: The imported event log.
    """
    return read_xes(path, show_progress_bar=show_progress_bar, variant=variant)


def log_len(log: DataFrame, traceid_key: str = constants.DEFAULT_TRACEID_KEY) -> int:
    """Compute the number of cases in the event log. (Number of unique trace ids)

    Args:
        log (DataFrame): The event log.
        traceid_key (str, optional): The column name for the trace id. Defaults to
            constants.DEFAULT_TRACEID_KEY.

    Returns:
        int: The number of unqiue trace ids in the event log.
    """
    return log[traceid_key].nunique()


def split_log_cases(
    log: DataFrame,
    frac: float,
    seed: int | None = None,
    traceid_key: str = constants.DEFAULT_TRACEID_KEY,
) -> tuple[DataFrame, DataFrame]:
    """Split an event log into two parts, with the first part containing `frac` of the
    cases and the second part containing the rest.

    Args:
        log (DataFrame): The event log.
        frac (float): The fraction of cases to assign to the first split. For instance,
            0.5 splits the event log in two halves.
        seed (int, optional): The seed for the random number generator. Defaults to None.
        traceid_key (str, optional): The column name for the trace id. Defaults to
            "case:concept:name".

    Returns:
        tuple[DataFrame, DataFrame]: The two splits of the event log.
    """
    num_cases = log_len(log)
    num_sample_cases = int(num_cases * frac)

    sample1 = sample_cases(log, num_sample_cases, seed, traceid_key)
    sample2 = log[~log[traceid_key].isin(sample1[traceid_key].unique())]

    return sample1, sample2


def sample_cases(
    log: pd.DataFrame,
    num_cases: int,
    seed: int | None = None,
    traceid_key: str = constants.DEFAULT_TRACEID_KEY,
) -> pd.DataFrame:
    """Sample a number of cases from the event log.

    Args:
        log (pd.DataFrame): The event log.
        num_cases (int): The number of cases to sample. Should not be larger than the
            event log.
        seed (int | None, optional): The seed to use for shuffling. Defaults to None
            (random).
        traceid_key (str, optional): The column name for the case id. Defaults to
            "case:concept:name".

    Returns:
        pd.DataFrame: The sampled event log
    """
    shuffled_log = log.sample(frac=1, random_state=seed)

    return dataframe_utils.sample_dataframe(
        shuffled_log,
        {
            "max_no_cases": num_cases,
            "deterministic": True,
            dataframe_utils.Parameters.CASE_ID_KEY: traceid_key,
            "pm4py:param:case_id_key": traceid_key,
        },
    )


def convert_lifecycle_eventlog_to_start_timestamp_eventlog(
    log: DataFrame,
    traceid_key: str = constants.DEFAULT_TRACEID_KEY,
    activity_key: str = constants.DEFAULT_NAME_KEY,
    timestamp_key: str = constants.DEFAULT_TIMESTAMP_KEY,
    lifecycle_key: str = constants.DEFAULT_LIFECYCLE_KEY,
    instance_key: str = constants.DEFAULT_INSTANCE_KEY,
) -> DataFrame:
    """Convert an event log consisting of events with lifecycle information and timestamp
    to an event log with start_timestamp and complete_timestamp.

    So if before there was a "start" event and a "complete" event with timestamps, this
    is combined into one event with "start_timestamp" as the timestamp of the "start"
    event and "time:timestamp" as the timestamp of the "complete" event.

    Args:
        log (DataFrame): The event log.
        traceid_key (str, optional): The column name for the trace id. Defaults to
            "case:concept:name".
        activity_key (str, optional): The column name for the activity label. Defaults
            to "concept:name".
        timestamp_key (str, optional): The column name for the timestamp. Defaults to
            "time:timestamp".
        lifecycle_key (str, optional): The column name for the lifecycle information.
            Defaults to "lifecycle:transition".
        instance_key (str, optional): The key for the id to tell different executions of
            activities apart. Defaults to "concept:instance". If this column is not
            present, a complete event is always matched to the first found start event
            in the order of the case.
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

        # Match each complete event to its start event (if there is one), otherwise use
        # the timestamp of the complete event
        ## Afterwards, the "end" events will remain, with a new attribute "start_timestamp"
        ## that contains the timestamp of the corresponding start event (or their own
        ## timestamp if there is no start event --> Instant execution)
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
    return new_df.sort_values(by=timestamp_key, ascending=True).reset_index(drop=True)


def convert_atomic_eventlog_to_lifecycle_eventlog(
    log: DataFrame,
    traceid_key: str = constants.DEFAULT_TRACEID_KEY,
    lifecycle_key: str = constants.DEFAULT_LIFECYCLE_KEY,
    instance_key: str = constants.DEFAULT_INSTANCE_KEY,
) -> DataFrame:
    """Convert an event log with no lifecycle information or start timestamps into an
    equivalent one with lifecycle information.
    In particular, this adds a lifecycle column with value "complete" for each event, and
    an event instance column with unique values (signifying each event corresponds to a
    different activity execution).

    The event instance id generated has the form `<caseid>:<idx>` where `idx` is the
    index of this event in the case

    Args:
        log (DataFrame): The Event Log.
        traceid_key (str, optional): The column name for the trace id. Defaults to
            "case:concept:name".
        lifecycle_key (str, optional): The column name to use for the lifecycle information.
            Defaults to "lifecycle:transition". If this column already exists, it will be overwritten.
        instance_key (str, optional): The column name to use for the instance information.
            Defaults to "concept:instance". If this column already exists, it will be overwritten.
    Returns:
        DataFrame: The converted event log.
    """
    new_log = log.copy()
    new_log[lifecycle_key] = "complete"
    return add_event_instance_id_to_log(new_log, traceid_key, instance_key)


def convert_atomic_eventlog_to_start_timestamp_eventlog(
    log: DataFrame,
    timestamp_key: str = constants.DEFAULT_TIMESTAMP_KEY,
) -> DataFrame:
    """Convert an event log with no lifecycle information or start timestamps into an
    equivalent one with a "start_timestamp" column for each event.
    In particular, this adds a new column "start_timestamp" with the same values as the
    "time:timestamp" column.

    Args:
        log (DataFrame): The Event Log.
        timestamp_key (str, optional): The column name for the timestamp of the event.
            Defaults to "time:timestamp".

    Returns:
        DataFrame: The converted event log
    """
    new_log = log.copy()
    new_log[constants.DEFAULT_START_TIMESTAMP_KEY] = new_log[timestamp_key]
    return new_log


def add_event_instance_id_to_log(
    log: pd.DataFrame,
    traceid_key: str = constants.DEFAULT_TRACEID_KEY,
    instance_key: str = constants.DEFAULT_INSTANCE_KEY,
) -> pd.DataFrame:
    """Add instance ids to the event log. Adds a unique instance id to each event.

    Args:
        log (pd.DataFrame): The event log.
        traceid_key (str, optional): The column for the trace ids. Defaults to
            "case:concept:name".
        instance_key (str, optional): The column for the instance ids. Defaults to
            "concept:instance".

    Returns:
        pd.DataFrame: The event log with instance ids.
    """
    new_log = log.copy()
    new_log[instance_key] = log.groupby(by=traceid_key).cumcount() + 1
    new_log[instance_key] = (
        new_log[traceid_key] + ":" + new_log[instance_key].astype(str)
    )

    return new_log


def add_duration_column_to_log(
    log: pd.DataFrame,
    start_time_key: str = constants.DEFAULT_START_TIMESTAMP_KEY,
    end_time_key: str = constants.DEFAULT_TIMESTAMP_KEY,
    duration_key: str = "@pcomp:duration",
) -> pd.DataFrame:
    """Compute the duration (in seconds) of each event based on its start- and end
    timestamp.

    Args:
        log (pd.DataFrame): The event log.
        start_time_key (str, optional): The column name for the start timestamp.
            Defaults to constants.DEFAULT_START_TIMESTAMP_KEY.
        end_time_key (str, optional): The column name for the completion timestamp.
            Defaults to constants.DEFAULT_TIMESTAMP_KEY.
        duration_key (str, optional): The column name to write the durations in. Defaults
            to "@pcomp:duration".

    Returns:
        pd.DataFrame: The changed event log

    Raises:
        ValueError: If the given start_time_key or end_time_key are not in the event log
    """
    if start_time_key not in log.columns or end_time_key not in log.columns:
        raise ValueError(
            "Event Log must contain a start timestamp key and end timestamp key to compute the duration"
        )
    new_log = log.copy()
    new_log[duration_key] = (
        new_log[end_time_key] - new_log[start_time_key]
    ).dt.total_seconds()

    return new_log


def ensure_start_timestamp_column(
    df: pd.DataFrame,
    start_timestamp_key: str = constants.DEFAULT_START_TIMESTAMP_KEY,
    lifecycle_key: str = constants.DEFAULT_LIFECYCLE_KEY,
) -> pd.DataFrame:
    """Ensure that the event log has a start timestamp column.
    If it doesn't, try creating one using lifecycle information.
    If no lifecycle information is present, try interpret the event log as an atomic
    event log and create a start timestamp using the end timestamps (same start timestamp
    and end timestamp for each event).

    Args:
        df (pd.DataFrame): The event log.
        start_timestamp_key (str, optional): The column for the start timestamps of events.
            Defaults to "start_timestamp".
        lifecycle_key (str, optional): The column for lifecycle information. Defaults to
            "lifecycle:transition".

    Returns:
        pd.DataFrame: The (altered) event log with a start timestamp column.
    """
    if start_timestamp_key not in df.columns:
        if lifecycle_key in df.columns:
            return convert_lifecycle_eventlog_to_start_timestamp_eventlog(df)
        else:
            return convert_atomic_eventlog_to_start_timestamp_eventlog(df)
    else:
        return df


def pretty_format_duration(seconds: float) -> str:
    """Format a duration in seconds as a string in the format HH:MM:SS

    Args:
        seconds (float): The duration in seconds.

    Returns:
        str: The formatted duration.
    """
    num_hours = int(seconds // 3600)
    num_minutes = int((seconds % 3600) // 60)
    num_seconds = seconds % 60
    return f"{num_hours:02}:{num_minutes:02}:{'0' if num_seconds < 10 else ''}{num_seconds:02.2f}"


def enable_logging(level: int = logging.INFO):
    """Set the logging level so that the respective log messages are shown.

    Args:
        level (int, optional): The logging level to use. Defaults to logging.INFO.
    """
    logging.basicConfig(level=level)


class DevNullProgressBar:
    """
    A dummy progress bar that does nothing when updated/closed. Used when no progress bar
    should be shown.
    """

    def __init__(self, *args, **_):
        if len(args) > 0:
            self.data = args[0]

    def __iter__(self):
        for i in self.data:
            yield i

    def update(self, amount: int = 1):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()  # Does nothing anyway


def create_progress_bar(
    show_progress_bar: bool = True, *args, **kwargs
) -> tqdm | DevNullProgressBar:
    """Helper function to create a progress bar. If `show_progress_bar` is False, a dummy
    progress bar is returned that does nothing when updated/closed.

    Args:
        show_progress_bar (bool, optional): Return a real or dummy progress bar?
            Defaults to True.
        *args: Arguments to pass to the progress bar constructor
        **kwargs: Keyword arguments to pass to the progress bar constructor

    Returns:
        tqdm | DevNullProgressBar: The created (dummy) progress bar.
    """
    if show_progress_bar:
        return tqdm(*args, **kwargs)
    else:
        return DevNullProgressBar(*args, **kwargs)
