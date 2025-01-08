# This is an application of the concept drift detection approach from the paper
# "Time-aware Concept Drift Detection Using the Earth Mover’s Distance" by Brockhoff et al.
# Here, we use the idea of this approach to compare two event logs w.r.t. timed control-flow

from collections.abc import Callable

import pandas as pd
from strsimpy.levenshtein import Levenshtein  # type: ignore
from strsimpy.weighted_levenshtein import WeightedLevenshtein  # type: ignore

from pcomp.binning import BinnerManager
from pcomp.utils import add_duration_column_to_log, constants

Trace = tuple[str, ...]

ServiceTimeEvent = tuple[str, float]
ServiceTimeTrace = tuple[ServiceTimeEvent, ...]

BinnedServiceTimeEvent = tuple[str, int]
BinnedServiceTimeTrace = tuple[BinnedServiceTimeEvent, ...]


def extract_service_time_traces(
    log: pd.DataFrame,
    activity_key: str = constants.DEFAULT_NAME_KEY,
    start_time_key: str = constants.DEFAULT_START_TIMESTAMP_KEY,
    end_time_key: str = constants.DEFAULT_TIMESTAMP_KEY,
    traceid_key: str = constants.DEFAULT_TRACEID_KEY,
    tiebreaker_key: str | None = constants.DEFAULT_NAME_KEY,
) -> list[ServiceTimeTrace]:
    """Extract a list of ServiceTimeTrace from the Event Log. This is an abstraction
    from the event log, where a case is considered a sequence (tuple) of tuples of 1)
    Activity and 2) Duration.

    The event log must be an interval event log, i.e. each event has a start and end
    timestamp.

    Args:
        log (pd.DataFrame): The event log.
        activity_key (str, optional): The key for the activity label in the event log.
            Defaults to constants.DEFAULT_NAME_KEY.
        start_time_key (str, optional): The key for the start timestamp in the event log.
            Defaults to constants.DEFAULT_START_TIMESTAMP_KEY.
        end_time_key (str, optional): The key for the end timestamp in the event log.
            Defaults to constants.DEFAULT_TIMESTAMP_KEY.
        traceid_key (str, optional): The key for the trace id in the event log. Defaults
            to constants.DEFAULT_TRACEID_KEY.
        tiebreaker_key (str | None, optional): The key to use for sorting two events with
            identical completion timestamps. Ensures consistent ordering across traces.
            Defaults to constants.DEFAULT_NAME_KEY.

    Returns:
        list[ServiceTimeTrace]: The list of ServiceTimeTraces extracted from the event log.
    """
    # Also sort by activity key to ensure that events with identical completion timestamp
    # are ordered consistently across traces
    sort_by = [end_time_key]
    if tiebreaker_key is not None:
        sort_by.append(tiebreaker_key)

    if "@pcomp:duration" not in log.columns:
        log = add_duration_column_to_log(
            log,
            start_time_key=start_time_key,
            end_time_key=end_time_key,
            duration_key="@pcomp:duration",
        )

    # For each case a tuple containing for each event a tuple of 1) Activity and 2) Duration
    return extract_trace_with_numerical_attribute(
        log, "@pcomp:duration", activity_key, traceid_key, end_time_key, tiebreaker_key
    )


def extract_traces(
    log: pd.DataFrame,
    traceid_key: str = constants.DEFAULT_TRACEID_KEY,
    activity_key: str = constants.DEFAULT_NAME_KEY,
    complete_timestamp_key: str = constants.DEFAULT_TIMESTAMP_KEY,
    tiebreaker_key: str = constants.DEFAULT_NAME_KEY,
    lifecycle_column: str = constants.DEFAULT_LIFECYCLE_KEY,
    filter_complete_lifecycle: bool = True,
) -> list[Trace]:
    """Extract the traces from the event log. Traces are sequences of executed activities.

    Args:
        log (pd.DataFrame): The event log.
        traceid_key (str, optional): The column name for the case id. Defaults to
            "case:concept:name".
        activity_key (str, optional): The column name for the activity label. Defaults
            to "concept:name".
        complete_timestamp_key (str, optional): The column name for the completion timestamp
            of an event. Defaults to "time:timestamp".
        tiebreaker_key (str, optional): The key to use to order events with the same
            completion timestamp. Defaults to "concept:name" (The activity label).
        lifecycle_key (str, optional): The column name for the lifecycle information. Used
            for retaining only complete events if `filter_complete_lifecycle` is True.
            Defaults to "lifecycle:transition".
            completion timestamp. Defaults to "concept:name" (The activity label).
        filter_complete_lifecycle (bool, optional): Filter the event log to only contain
            complete events. Defaults to true

    Returns:
        list[Trace]: The extracted traces
    """
    sort_by = [complete_timestamp_key]
    if tiebreaker_key is not None:
        sort_by.append(tiebreaker_key)

    if lifecycle_column in log.columns and filter_complete_lifecycle:
        log = log[log[lifecycle_column] == "complete"]

    return (
        log.sort_values(by=sort_by)
        .groupby(by=traceid_key, sort=False)[activity_key]
        .apply(tuple)
        .tolist()
    )


def extract_trace_with_numerical_attribute(
    log: pd.DataFrame,
    attribute_column: str,
    activity_key: str = constants.DEFAULT_NAME_KEY,
    traceid_key: str = constants.DEFAULT_TRACEID_KEY,
    end_time_key: str = constants.DEFAULT_TIMESTAMP_KEY,
    tiebreaker_key: str | None = constants.DEFAULT_NAME_KEY,
) -> list[ServiceTimeTrace]:
    sort_by = [end_time_key]
    if tiebreaker_key is not None:
        sort_by.append(tiebreaker_key)

    # For each case a tuple containing for each event a tuple of
    # 1) Activity and 2) The value in the attribute_column
    return (
        log.sort_values(by=sort_by)
        .groupby(by=traceid_key, sort=False)[[activity_key, attribute_column]]
        .apply(
            lambda group: tuple(  # type: ignore [arg-type]
                group.itertuples(index=False, name=None)  # type: ignore
            )
        )
        .tolist()
    )


def extract_binned_service_time_traces(
    log: pd.DataFrame,
    binner_manager: BinnerManager,
    activity_key: str = constants.DEFAULT_NAME_KEY,
    start_time_key: str = constants.DEFAULT_START_TIMESTAMP_KEY,
    end_time_key: str = constants.DEFAULT_TIMESTAMP_KEY,
) -> list[BinnedServiceTimeTrace]:
    """Extract the traces from the event log with a special focus on service times, i.e.,
    a view concerned only with the executed activity, and how long its execution took.
    Only defined for Interval Event Logs, i.e. each event has a Start and Complete
    timestamp

    Args:
        log (EventLog): The event log
        binner_manager (BinnerManager): The binner manager to use for binning the
            service times.
        activity_key (str, optional): The key for the activity value in the event log.
            Defaults to xes.DEFAULT_NAME_KEY.
        start_time_key (str, optional): The key in the event log for the start timestamp
            of the event. Defaults to xes.DEFAULT_START_TIMESTAMP_KEY.
        end_time_key (str, optional): The key in the event log for the completion
            timestamp of the event. Defaults to xes.DEFAULT_TIMESTAMP_KEY.
    Returns:
        list[ServiceTimeTrace]: A sequence of traces, represented as a tuple of activity
            and binned duration. Same order as in the original event log.
    """
    service_time_traces = extract_service_time_traces(
        log, activity_key, start_time_key, end_time_key
    )
    return [
        tuple(
            # Bin the duration with the binner associated to the activitiy
            (activity, binner_manager.bin(activity, duration))
            for activity, duration in trace
        )
        for trace in service_time_traces
    ]


def extract_binned_trace_with_numerical_attribute(
    log: pd.DataFrame,
    binner_manager: BinnerManager,
    attribute_column: str,
    activity_key: str = constants.DEFAULT_NAME_KEY,
    traceid_key: str = constants.DEFAULT_TRACEID_KEY,
    end_time_key: str = constants.DEFAULT_TIMESTAMP_KEY,
    tiebreaker_key: str | None = constants.DEFAULT_NAME_KEY,
) -> list[BinnedServiceTimeTrace]:
    traces = extract_trace_with_numerical_attribute(
        log, attribute_column, activity_key, traceid_key, end_time_key, tiebreaker_key
    )

    return [
        tuple(
            # Bin the attribute value with the binner associated to the activitiy
            (activity, binner_manager.bin(activity, value))
            for activity, value in trace
        )
        for trace in traces
    ]


def levenshtein_distance(trace_1: Trace, trace_2: Trace) -> float:
    """Compute the levenshtein distance between two traces (lists of strings). A wrapper
    around the function provided by strsimpy.

    Args:
        trace_1 (Trace): The first trace.
        trace_2 (Trace): The second trace.

    Returns:
        float: The levenshtein edit distance between the traces.
    """
    return Levenshtein().distance(trace_1, trace_2)


def postnormalized_levenshtein_distance(trace_1: Trace, trace_2: Trace) -> float:
    """Compute the postnormalized levenshtein distance between two traces by dividing their
    levenshtein edit distance by the larger length.

    Args:
        trace_1 (Trace): The first trace.
        trace_2 (Trace): The second trace.

    Returns:
        float: The postnormalized levenshtein distance.
    """
    return levenshtein_distance(trace_1, trace_2) / max(len(trace_1), len(trace_2))


def weighted_levenshtein_distance(
    trace1: BinnedServiceTimeTrace,
    trace2: BinnedServiceTimeTrace,
    rename_cost: Callable[[str, str], float],
    insertion_deletion_cost: Callable[[str], float],
    cost_time_match_rename: Callable[[int, int], float],
    cost_time_insert_delete: Callable[[int], float],
) -> float:
    """Compute the levenshtein distance with custom weights. Using strsimpy

    Args:
        trace1 (BinnedServiceTimeTrace): The first trace.
        trace2 (BinnedServiceTimeTrace): The second trace.
        rename_cost (Callable[[str], float]): Custom Cost.
        insertion_deletion_cost (Callable[[str], float]): Custom Cost.
        cost_time_match_rename (Callable[[int], float]): Custom Cost.
        cost_time_insert_delete (Callable[[int], float]): Custom Cost.

    Returns:
        float: The computed weighted Levenshtein distance.
    """

    def ins_del_cost(x: BinnedServiceTimeEvent) -> float:
        return insertion_deletion_cost(x[0]) + cost_time_insert_delete(x[1])

    def substitution_cost(
        x: BinnedServiceTimeEvent, y: BinnedServiceTimeEvent
    ) -> float:
        cost_rename = (
            0 if x[0] == y[0] else rename_cost(x[0], y[0])
        )  # Allow matching activity while renaming time
        return cost_rename + cost_time_match_rename(x[1], y[1])

    dist = WeightedLevenshtein(
        insertion_cost_fn=ins_del_cost,
        deletion_cost_fn=ins_del_cost,
        substitution_cost_fn=substitution_cost,
    ).distance(trace1, trace2)
    return dist


def post_normalized_weighted_levenshtein_distance(
    trace1: BinnedServiceTimeTrace,
    trace2: BinnedServiceTimeTrace,
    rename_cost: Callable[[str, str], float],
    insertion_deletion_cost: Callable[[str], float],
    cost_time_match_rename: Callable[[int, int], float],
    cost_time_insert_delete: Callable[[int], float],
) -> float:
    """Compute the post-normalized weighted Levenshtein distance. This is used for the
    calculation of a "time-aware" EMD.

    Args:
        trace1 (BinnedServiceTimeTrace): The first trace.
        trace2 (BinnedServiceTimeTrace): The second trace.
        rename_cost (Callable[[str, str], float]): Custom Cost.
        insertion_deletion_cost (Callable[[str], float]): Custom Cost.
        cost_time_match_rename (Callable[[int, int], float]): Custom Cost.
        cost_time_insert_delete (Callable[[int], float]): Custom Cost.

    Returns:
        float: The post-normalized weighted Levenshtein distance.
    """
    return weighted_levenshtein_distance(
        trace1,
        trace2,
        rename_cost=rename_cost,
        insertion_deletion_cost=insertion_deletion_cost,
        cost_time_match_rename=cost_time_match_rename,
        cost_time_insert_delete=cost_time_insert_delete,
    ) / max(len(trace1), len(trace2))
