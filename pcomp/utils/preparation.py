import pandas as pd

from pcomp.utils.constants import (
    DEFAULT_INSTANCE_KEY,
    DEFAULT_LIFECYCLE_KEY,
    DEFAULT_NAME_KEY,
    DEFAULT_START_TIMESTAMP_KEY,
    DEFAULT_TIMESTAMP_KEY,
    DEFAULT_TRACEID_KEY,
    END_LIFECYLCES,
    START_LIFECYLCES,
)


def infer_event_instance_id(
    log: pd.DataFrame,
    instance_id_key: str = DEFAULT_INSTANCE_KEY,
    traceid_key: str = DEFAULT_TRACEID_KEY,
    activity_key: str = DEFAULT_NAME_KEY,
    timestamp_key: str = DEFAULT_TIMESTAMP_KEY,
    lifecycle_key: str = DEFAULT_LIFECYCLE_KEY,
) -> pd.DataFrame:
    """Infer the instance id by matching the first start event of an activity with the
    first complete event of that activity in the same trace. The instance id is a unique
    identifier used to pair the start and complete events of an activity.

    Does not raise an error if a start event has no corresponding complete event!

    Args:
        log (pd.DataFrame): The event log.
        instance_id_key (str, optional): The column name to use for the inferred instance
            id. Defaults to "concept:instance".
        traceid_key (str, optional): The column name for the trace-id. Defaults
            to "case:concept:name".
        activity_key (str, optional): The column name for the activity label.
            Defaults to "concept:name.
        timestamp_key (str, optional): The column name for the timestamp of an
            event. Defaults to "time:timestamp".
        lifecycle_key (str, optional): The column name for the lifecycle
            information of an event. Defaults to "lifecycle:transition".

    Returns:
        pd.DataFrame: The event log with thew added instance id key.
    """
    for column in [traceid_key, activity_key, timestamp_key, lifecycle_key]:
        if column not in log.columns:
            raise ValueError(f"Column {column} required to infer event instance id.")

    new_log = log.sort_values(
        timestamp_key, ascending=True
    )  # Reverse sorting: later (complete) events first # TODO: This comment seems outdated

    # Give each complete event a unique id
    new_log[instance_id_key] = None  # -1

    is_complete_event = new_log[lifecycle_key].isin(END_LIFECYLCES)
    is_start_event = new_log[lifecycle_key].isin(START_LIFECYLCES)
    # Each complete event gets its index of all complete events with this activity
    new_log.loc[is_complete_event, instance_id_key] = (
        new_log[is_complete_event].groupby([traceid_key, activity_key]).cumcount()
    )
    # Do the same for start events
    new_log.loc[is_start_event, instance_id_key] = (
        new_log[is_start_event].groupby([traceid_key, activity_key]).cumcount()
    )
    # --> First "a" start event gets 0, first "a" complete event gets 0 --> match
    # --> Second "a" start event gets 1, second "a" complete event gets 1 --> match
    # --> etc.

    new_log[instance_id_key] = new_log[instance_id_key].astype(str)

    return new_log


def fold_instance_id_log_to_partial_order_log(
    log: pd.DataFrame,
    instance_id_key: str = DEFAULT_INSTANCE_KEY,
    traceid_key: str = DEFAULT_TRACEID_KEY,
    activity_key: str = DEFAULT_NAME_KEY,
    timestamp_key: str = DEFAULT_TIMESTAMP_KEY,
    start_timestamp_key: str = DEFAULT_START_TIMESTAMP_KEY,
    lifecycle_key: str = DEFAULT_LIFECYCLE_KEY,
) -> pd.DataFrame:
    """Merge events with the same instance id, case, and activity_key into one event with
    a start- and complete timestamp.

    Args:
        log (pd.DataFrame): The event log.
        instance_id_key (str, optional): The column name for the instance id of an event.
            Defaults to "concept:instance".
        traceid_key (str, optional): The column name for the trace-id. Defaults to
            "case:concept:name".
        activity_key (str, optional): The column name for the activity label. Defaults
            to "concept:name".
        timestamp_key (str, optional): The column name for the completion timestamp.
            Defaults to "time:timestamp".
        start_timestamp_key (str, optional): The column name for the start timestamp.
            Defaults to "start_timestamp".
        lifecycle_key (str, optional): The column name for the lifecycle information of
            an event. Defaults to "lifecycle:transition".

    Returns:
        pd.DataFrame: The event log with merged events.

    Raises:
        ValueError: The provided instance id column is not present in the event log
        ValueError: The instance ids pair multiple start events with a single complete
            event
        ValueError: Two complete events found with same trace id, activity, and instance
            id.
    """
    if instance_id_key not in log.columns:
        raise ValueError(f'Instance id column "{instance_id_key}" not found in log')

    new_log = log.sort_values(
        by=[traceid_key, activity_key, instance_id_key], ascending=True
    )

    for (trace_id, _, iid), group_df in new_log.groupby(
        by=[traceid_key, activity_key, instance_id_key], sort=False
    ):
        if len(group_df) == 1:
            continue

        complete_events = group_df[group_df[lifecycle_key].isin(END_LIFECYLCES)]
        start_events = group_df[group_df[lifecycle_key].isin(START_LIFECYLCES)]

        if len(start_events) > 1:
            raise ValueError(
                f"Multiple starting events found for single complete event of instance id {iid}"
            )
        if len(complete_events) > 1:
            raise ValueError(
                f"Multiple complete events found for same activity in case {trace_id}"
            )

        new_log.loc[complete_events.index, start_timestamp_key] = start_events[
            timestamp_key
        ].values

    new_log = new_log[~new_log[lifecycle_key].isin(START_LIFECYLCES)]
    # Fill in the start timestamp for complete events without a start event
    new_log[start_timestamp_key] = new_log[start_timestamp_key].fillna(
        new_log[timestamp_key]
    )
    return new_log


def ensure_start_timestamp_column(
    log: pd.DataFrame,
    lifecycle_key: str = DEFAULT_LIFECYCLE_KEY,
    instance_key: str = DEFAULT_INSTANCE_KEY,
    traceid_key: str = DEFAULT_TRACEID_KEY,
    activity_key: str = DEFAULT_NAME_KEY,
    timestamp_key: str = DEFAULT_TIMESTAMP_KEY,
) -> pd.DataFrame:
    """Ensure that the event log has the "start_timestamp" column. If it is not present
    in the log, add the start_timestamp column by doing the following:

        1. If instance id's ("concept:instance") are present, compute the start_timestamp
           for the complete event as the timestamp of the corresponding start event
        2. If no instance id's are present, infer them by pairing the earliest start event
           with the earliest complete event, etc.
        3. If no lifecycle information is present, give all events "complete" lifecycle
           transitions.

    Args:
        log (pd.DataFrame): The event log.
        lifecycle_key (str, optional): The column for the lifecycle information of an event.
            Defaults to "lifecycle:transition".
        instance_key (str, optional): The column for the instance id of an event. Used
            to pair start- and end events for executions of the same activity. If this
            column does not exist, infer using `infer_event_instance_id`. Defaults to
            "concept:instance".
        traceid_key (str, optional): The column name for the trace-id. Defaults
            to "case:concept:name".
        activity_key (str, optional): The column name for the activity label.
            Defaults to "concept:name.
        timestamp_key (str, optional): The column name for the timestamp of an
            event. Defaults to "time:timestamp".

    Returns:
        pd.DataFrame: The event log with the added "start_timestamp" column. If it was
            already in the event log, a copy of the log is returned.
    """
    if DEFAULT_START_TIMESTAMP_KEY in log.columns:
        return log.copy()

    if lifecycle_key not in log.columns:
        # The event log is atomic, so all events are complete events
        log[lifecycle_key] = "complete"
    if instance_key not in log.columns:
        log = infer_event_instance_id(
            log,
            instance_id_key=instance_key,
            traceid_key=traceid_key,
            activity_key=activity_key,
            timestamp_key=timestamp_key,
            lifecycle_key=lifecycle_key,
        )
    return (
        fold_instance_id_log_to_partial_order_log(
            log,
            instance_id_key=instance_key,
            traceid_key=traceid_key,
            activity_key=activity_key,
            timestamp_key=timestamp_key,
            start_timestamp_key=DEFAULT_START_TIMESTAMP_KEY,
            lifecycle_key=lifecycle_key,
        )
        .sort_values(by=timestamp_key)
        .reset_index(drop=True)
    )
