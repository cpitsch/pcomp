# This is an application of the concept drift detection approach from the paper
# "Time-aware Concept Drift Detection Using the Earth Mover’s Distance" by Brockhoff et al.
# Here, we use the idea of this approach to compare two event logs w.r.t. timed control-flow

from collections.abc import Callable
from functools import cache
from typing import Any, Optional, cast

import pandas as pd
from pandas import DataFrame
from strsimpy.weighted_levenshtein import WeightedLevenshtein  # type: ignore

from pcomp.binning import BinnerFactory, BinnerManager, KMeans_Binner
from pcomp.emd.core import (
    BootstrappingStyle,
    EMD_ProcessComparator,
    EMDBackend,
    bootstrap_emd_population,
    compute_emd,
    population_to_stochastic_language,
)
from pcomp.utils import constants, ensure_start_timestamp_column

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
) -> list[ServiceTimeTrace]:
    """Extract a list of ServiceTimeTrace from the Event Log. This is an abstraction from the event log, where a case is considered a sequence (tuple)
    of tuples of 1) Activity and 2) Duration.

    The event log must be an interval event log, i.e. each event has a start and end timestamp.

    Args:
        log (pd.DataFrame): The event log.
        activity_key (str, optional): The key for the activity label in the event log. Defaults to constants.DEFAULT_NAME_KEY.
        start_time_key (str, optional): The key for the start timestamp in the event log. Defaults to constants.DEFAULT_START_TIMESTAMP_KEY.
        end_time_key (str, optional): The key for the end timestamp in the event log. Defaults to constants.DEFAULT_TIMESTAMP_KEY.
        traceid_key (str, optional): The key for the trace id in the event log. Defaults to constants.DEFAULT_TRACEID_KEY.

    Returns:
        list[ServiceTimeTrace]: The list of ServiceTimeTraces extracted from the event log.
    """
    # For each case a tuple containing for each event a tuple of 1) Activity and 2) Duration
    return (
        log.sort_values(by=end_time_key)
        .groupby(traceid_key, sort=False)  # sort=False to retain trace order
        .apply(
            lambda group_df: tuple(  # type: ignore [arg-type, return-value]
                (
                    evt[activity_key],
                    (evt[end_time_key] - evt[start_time_key]).total_seconds(),
                )
                for (_, evt) in cast(
                    pd.DataFrame, group_df
                )  # Tell typing that group_df is a Dataframe since pandas typing is weird
                .sort_values(by=end_time_key)
                .iterrows()
            )
        )
        .tolist()
    )


def extract_traces_activity_service_times(
    log: pd.DataFrame,
    binner_manager: BinnerManager,
    activity_key: str = constants.DEFAULT_NAME_KEY,
    start_time_key: str = constants.DEFAULT_START_TIMESTAMP_KEY,
    end_time_key: str = constants.DEFAULT_TIMESTAMP_KEY,
) -> list[BinnedServiceTimeTrace]:
    """Extract the traces from the event log with a special focus on service times, i.e., a view concerned only with the executed activity, and how long its execution took. Only defined for Interval Event Logs, i.e. each event has a Start and Complete timestamp

    Args:
        log (EventLog): The event log
        binner_manager (BinnerManager): The binner manager to use for binning the service times.
        activity_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.
        start_time_key (str, optional): The key in the event log for the start timestamp of the event. Defaults to xes.DEFAULT_START_TIMESTAMP_KEY.
        end_time_key (str, optional): The key in the event log for the completion timestamp of the event. Defaults to xes.DEFAULT_TIMESTAMP_KEY.
    Returns:
        list[ServiceTimeTrace]: A sequence of traces, represented as a tuple of activity and binned duration. Same order as in the original event log.
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


def custom_levenshtein_distance(
    trace1: BinnedServiceTimeTrace, trace2: BinnedServiceTimeTrace
) -> float:
    """Calculate the weighted levenshtein distance with the following weights:

        - Rename cost: 1
        - Insertion/Deletion cost: 1
        - Time Match/Rename cost: Absolute difference between the times
        - Time Insert/Delete cost: x (The value of the time)

        Thanks to not taking (lambda) cost functions as inputs, caching (hashing) can work correctly and save time.
        For the implementation with caching, see `cached_custom_levenshtein_distance`.

    Args:
        trace1 (BinnedServiceTimeTrace): The first trace.
        trace2 (BinnedServiceTimeTrace): The second trace.

    Returns:
        float: The computed weighted Levenshtein distance for these costs.
    """
    return weighted_levenshtein_distance(
        trace1,
        trace2,
        rename_cost=lambda *_: 1,
        insertion_deletion_cost=lambda _: 1,
        cost_time_match_rename=lambda x, y: abs(x - y),
        cost_time_insert_delete=lambda x: x,
    )


@cache
def cached_custom_levenshtein_distance(
    trace1: BinnedServiceTimeTrace, trace2: BinnedServiceTimeTrace
) -> float:
    """A cached version of `custom_levenshtein_distance`."""
    return custom_levenshtein_distance(trace1, trace2)


def custom_postnormalized_levenshtein_distance(
    trace1: BinnedServiceTimeTrace,
    trace2: BinnedServiceTimeTrace,
) -> float:
    """Calculate the postnmormalized weighted levenshtein distance with the following weights:

        - Rename cost: 1
        - Insertion/Deletion cost: 1
        - Time Match/Rename cost: Absolute difference between the times
        - Time Insert/Delete cost: x (The value of the time)

        After computing the distance, it is divided by the length of the longer trace.

        Thanks to not taking (lambda) cost functions as inputs, caching (hashing) can work correctly and could save time.
        For caching, see `cached_custom_postnormalized_levenshtein_distance`.

    Args:
        trace1 (BinnedServiceTimeTrace): The first trace.
        trace2 (BinnedServiceTimeTrace): The second trace.

    Returns:
        float: The computed postnormalized weighted Levenshtein distance for these costs.
    """
    return post_normalized_weighted_levenshtein_distance(
        trace1,
        trace2,
        rename_cost=lambda *_: 1,
        insertion_deletion_cost=lambda _: 1,
        cost_time_match_rename=lambda x, y: abs(x - y),
        cost_time_insert_delete=lambda x: x,
    )


@cache
def cached_custom_postnormalized_levenshtein_distance(
    trace1: BinnedServiceTimeTrace, trace2: BinnedServiceTimeTrace
) -> float:
    """A cached version of `custom_postnormalized_levenshtein_distance`."""
    return custom_postnormalized_levenshtein_distance(trace1, trace2)


def post_normalized_weighted_levenshtein_distance(
    trace1: BinnedServiceTimeTrace,
    trace2: BinnedServiceTimeTrace,
    rename_cost: Callable[[str, str], float],
    insertion_deletion_cost: Callable[[str], float],
    cost_time_match_rename: Callable[[int, int], float],
    cost_time_insert_delete: Callable[[int], float],
) -> float:
    """Compute the post-normalized weighted Levenshtein distance. This is used for the calculation of a "time-aware" EMD.

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


def calc_timing_emd(
    distribution_1: list[tuple[BinnedServiceTimeTrace, float]],
    distribution_2: list[tuple[BinnedServiceTimeTrace, float]],
) -> float:
    """Calculate the Earth Mover's Distance between two populations of BinnedServiceTimeTraces.

    Args:
        distribution_1 (list[tuple[BinnedServiceTimeTrace, float]]): The population for the first event log.
        distribution_2 (list[tuple[BinnedServiceTimeTrace, float]]): The population for the second event log.

    Returns:
        float: The computed (Time-Aware) Earth Mover's Distance.
    """
    return compute_emd(
        distribution_1,
        distribution_2,
        custom_postnormalized_levenshtein_distance,
    )


def log_to_stochastic_language(
    log: pd.DataFrame,
    binner_manager: BinnerManager,
    activity_key: str = constants.DEFAULT_NAME_KEY,
    start_time_key: str = constants.DEFAULT_START_TIMESTAMP_KEY,
    end_time_key: str = constants.DEFAULT_TIMESTAMP_KEY,
) -> list[tuple[BinnedServiceTimeTrace, float]]:
    """Extract the stochastic language of BinnedServiceTimeTraces from an event log.

    This is done by extracting ServiceTimeTraces, then binning the durations and finally computing the relative frequencies.

    Args:
        log (pd.DataFrame): The event log.
        binner_manager (BinnerManager): The binner manager to use to bin the activity service times.
        activity_key (str, optional): The key for the activity label in the event log. Defaults to constants.DEFAULT_NAME_KEY.
        start_time_key (str, optional): The key for the start timestamp in the event log. Defaults to constants.DEFAULT_START_TIMESTAMP_KEY.
        end_time_key (str, optional): The key for the end timestamp. Defaults to constants.DEFAULT_TIMESTAMP_KEY.

    Returns:
        list[tuple[BinnedServiceTimeTrace, float]]: The extracted stochastic language.
    """
    population: list[BinnedServiceTimeTrace] = extract_traces_activity_service_times(
        log, binner_manager, activity_key, start_time_key, end_time_key
    )

    return population_to_stochastic_language(population)


def compare_logs_emd(
    log_1: pd.DataFrame,
    log_2: pd.DataFrame,
    binner_manager: BinnerManager | None = None,
    num_bins: int = 3,
    activity_key: str = constants.DEFAULT_NAME_KEY,
    start_time_key: str = constants.DEFAULT_START_TIMESTAMP_KEY,
    end_time_key: str = constants.DEFAULT_TIMESTAMP_KEY,
    seed: int | None = None,
) -> float:
    """Compare two event logs w.r.t. control-flow and timing using the Earth Mover's Distance.

    Both event logs must contain a start and end timestamp for each event (by default "start_timestamp" and "time:timestamp", respectively)

    Args:
        log_1 (pd.DataFrame): The first event log.
        log_2 (pd.DataFrame): The second event log.
        binner_manager (BinnerManager, optional): The binner to use to bin the activtiy service times. If none provided, one will be trained on the data using a KMeans++ binner with `num_bins` bins.
        num_bins (int, optional): The number of bins to cluster the service times into (Using K-Means++). Defaults to 3 (intuitively "slow", "medium", "fast").
        activity_key (str, optional): The name of the column containing the label of the executed activity. Defaults to "concept:name".
        start_time_key (str, optional): The name of the column containing the start timestamp of the event. Defaults to "start_timestamp".
        end_time_key (str, optional): The name of the column containing the (end-) timestamp of the event. Defaults to "time:timestamp".
        seed (int | None, optional): The seed used for the clustering of the timestamps for binning. Defaults to None (no seeds used).

    Returns:
        float: The earth mover's distance between the two event logs.
    """

    if binner_manager is None:
        binner_manager = BinnerManager(
            [
                evt
                for trace in extract_service_time_traces(
                    log_1, activity_key, start_time_key, end_time_key
                )
                for evt in trace
            ],
            KMeans_Binner,
            k=num_bins,
            seed=seed,
        )

    dist1: list[tuple[BinnedServiceTimeTrace, float]] = log_to_stochastic_language(
        log_1, binner_manager, activity_key, start_time_key, end_time_key
    )

    dist2: list[tuple[BinnedServiceTimeTrace, float]] = log_to_stochastic_language(
        log_2, binner_manager, activity_key, start_time_key, end_time_key
    )

    emd = compute_emd(dist1, dist2, custom_postnormalized_levenshtein_distance)
    return emd


def process_comparison_emd(
    log_1: pd.DataFrame,
    log_2: pd.DataFrame,
    num_bins: int = 3,
    bootstrapping_dist_size: int = 10_000,
    resample_size: Optional[int] = None,
    activity_key: str = constants.DEFAULT_NAME_KEY,
    lifecycle_key: str = constants.DEFAULT_LIFECYCLE_KEY,
    start_time_key: str = constants.DEFAULT_START_TIMESTAMP_KEY,
    end_time_key: str = constants.DEFAULT_TIMESTAMP_KEY,
    seed: int | None = None,
) -> float:
    """Compare two event logs w.r.t timed-control-flow using a Time Aware Earth Mover's Distance.

    Args:
        log_1 (pd.DataFrame): The first event log.
        log_2 (pd.DataFrame): The second event log.
        num_bins (int, optional): The number of bins to use for binning . Defaults to 3.
        bootstrapping_dist_size (int, optional): The number of self-distances to compute for bootstrapping. Defaults to 10_000.
        resample_size (Optional[int], optional): The size of samples to compute the self-distances for in bootstrapping. Defaults to None.
        activity_key (str, optional): The key for the activity label in the event log. Defaults to constants.DEFAULT_NAME_KEY.
        lifecycle_key (str, optional): The key for the lifecycle transition in the event log. Defaults to constants.DEFAULT_LIFECYCLE_KEY.
        start_time_key (str, optional): The key for the start timestamp in the event log. Defaults to constants.DEFAULT_START_TIMESTAMP_KEY.
        end_time_key (str, optional): The key for the end timestamp in the event log. Defaults to constants.DEFAULT_TIMESTAMP_KEY.
        seed (int | None, optional): The seed to use for binning. Defaults to None (no seed used).

    Returns:
        float: The computed p-value.
    """
    # Simplify lifecycle event logs to event logs with start- and end- timestamps if necessary
    log_1 = ensure_start_timestamp_column(log_1, start_time_key, lifecycle_key)
    log_2 = ensure_start_timestamp_column(log_2, start_time_key, lifecycle_key)

    # Convert cases in event logs to traces concerning activity and service time
    traces_1 = extract_service_time_traces(
        log_1, activity_key, start_time_key, end_time_key
    )

    binner_manager = BinnerManager(
        [evt for trace in traces_1 for evt in trace],
        KMeans_Binner,
        k=num_bins,
        seed=seed,
    )

    log_1_traces = extract_traces_activity_service_times(
        log_1, binner_manager, activity_key, start_time_key, end_time_key
    )  # We sample from in the bootstrapping

    log_1_stochastic_language = population_to_stochastic_language(log_1_traces)

    emd = compute_emd(
        log_1_stochastic_language,
        log_to_stochastic_language(  # log_2_stochastic_language
            log_2, binner_manager, activity_key, start_time_key, end_time_key
        ),
        cached_custom_postnormalized_levenshtein_distance,
    )

    # Bootstrap a p-value
    # Compute samples of EMD's of the logs with themselves to gauge a "normal" EMD
    self_emds = bootstrap_emd_population(
        log_1_traces,
        custom_postnormalized_levenshtein_distance,
        bootstrapping_dist_size,
        resample_size,
        show_progress_bar=True,
    )

    # Clear the cache for the levenshtein distance
    cached_custom_postnormalized_levenshtein_distance.cache_clear()

    num_larger_or_equal_bootstrap_dists = len([d for d in self_emds if d >= emd])
    return num_larger_or_equal_bootstrap_dists / bootstrapping_dist_size


class Timed_Levenshtein_EMD_Comparator(EMD_ProcessComparator[BinnedServiceTimeTrace]):
    """An implementation of the EMD_ProcessComparator for comparing event logs w.r.t. the timed-control-flow
    using a weighted post-normalized levenshtein distance as the cost function.
    """

    binner_manager: BinnerManager

    def __init__(
        self,
        log_1: DataFrame,
        log_2: DataFrame,
        bootstrapping_dist_size: int = 10000,
        resample_size: int | float | None = None,
        verbose: bool = True,
        cleanup_on_del: bool = True,
        bootstrapping_style: BootstrappingStyle = "replacement sublogs",
        emd_backend: EMDBackend = "wasserstein",
        seed: int | None = None,
        weighted_time_cost: bool = False,
        binner_factory: BinnerFactory | None = None,
        binner_args: dict[str, Any] | None = None,
    ):
        super().__init__(
            log_1,
            log_2,
            bootstrapping_dist_size,
            resample_size,
            verbose,
            cleanup_on_del,
            bootstrapping_style,
            emd_backend,
            seed,
        )
        self.weighted_time_cost = weighted_time_cost

        # Default to KMeans_Binner with 3 bins
        self.binner_factory = binner_factory or KMeans_Binner
        self.binner_args = binner_args or (
            {
                "k": 3,
            }
            if self.binner_factory == KMeans_Binner
            else {}
        )

    def extract_representations(
        self, log_1: DataFrame, log_2: DataFrame
    ) -> tuple[list[BinnedServiceTimeTrace], list[BinnedServiceTimeTrace]]:
        """Extract the service time traces from the event logs and bin their activity service times."""
        traces_1 = extract_service_time_traces(log_1)

        self.binner_manager = BinnerManager(
            [evt for trace in traces_1 for evt in trace],
            self.binner_factory,
            seed=self.seed,
            show_training_progress_bar=self.verbose,
            **self.binner_args,
        )

        return (
            extract_traces_activity_service_times(log_1, self.binner_manager),
            extract_traces_activity_service_times(log_2, self.binner_manager),
        )

    def cost_fn(
        self, item1: BinnedServiceTimeTrace, item2: BinnedServiceTimeTrace
    ) -> float:
        """
        If `weighted_time_cost` is True, the time costs are all weighted by the maximum possible time difference, `num_bins - 1`.
        As such, the maximum cost that can be incurred due to time differences in each event is 1.
        """
        if self.weighted_time_cost:
            return post_normalized_weighted_levenshtein_distance(
                item1,
                item2,
                rename_cost=lambda *_: 1,
                insertion_deletion_cost=lambda _: 1,
                cost_time_match_rename=lambda x, y: abs(x - y)
                / max(self.binner_manager.num_bins - 1, 1),
                cost_time_insert_delete=lambda x: x
                / max(self.binner_manager.num_bins - 1, 1),
            ) / (2 if self.binner_manager.num_bins > 0 else 1)
        else:
            return custom_postnormalized_levenshtein_distance(item1, item2)

    def cleanup(self) -> None:
        pass
