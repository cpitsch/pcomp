# This is an application of the concept drift detection approach from the paper
# "Time-aware Concept Drift Detection Using the Earth Mover’s Distance" by Brockhoff et al.
# Here, we use the idea of this approach to compare two event logs w.r.t. timed control-flow

import math
from collections import Counter
from collections.abc import Callable
from functools import cache
from typing import Optional
from itertools import groupby

import numpy as np
import pandas as pd
import wasserstein  # type: ignore
from sklearn.cluster import kmeans_plusplus  # type: ignore
from tqdm.auto import tqdm  # type: ignore
from strsimpy.weighted_levenshtein import WeightedLevenshtein  # type: ignore

from pcomp.utils import constants, log_len, ensure_start_timestamp_column

ServiceTimeEvent = tuple[str, float]
ServiceTimeTrace = tuple[ServiceTimeEvent, ...]

BinnedServiceTimeEvent = tuple[str, int]
BinnedServiceTimeTrace = tuple[BinnedServiceTimeEvent, ...]


class KMeans_Binner:
    def __init__(
        self,
        data: list[ServiceTimeTrace],
        num_bins: int,
        seed: int | None = None,
    ):
        self.data = data
        self.seed = seed
        self.num_bins = num_bins

        if self.seed is not None:
            np.random.seed(self.seed)

        sample_indices = np.random.choice(
            range(len(self.data)),
            size=math.ceil(0.2 * len(self.data)),
        )

        durations: dict[str, list[float]] = {
            key: [dur for _, dur in subiter]
            for key, subiter in groupby(
                sorted(
                    [
                        evt
                        for idx, trace in enumerate(self.data)
                        for evt in trace
                        if idx in sample_indices
                    ]
                ),
                lambda x: x[0],
            )
        }

        # Get a clustering for the service times of each activity

        self.centroids = {
            act: kmeans_plusplus(
                np.array(durs).reshape(-1, 1),
                n_clusters=self.num_bins,
                n_local_trials=10,
                random_state=seed,
            )[0]
            for act, durs in durations.items()
        }

    def _closestCentroid1D(self, point: float, centroids: np.ndarray) -> int:
        mindex = 0
        minval = centroids[0]
        for idx, centroid in enumerate(centroids):
            dist = abs(centroid - point)
            if dist < minval:
                mindex = idx
                minval = dist
        return mindex

    def bin_event(self, event: ServiceTimeEvent) -> BinnedServiceTimeEvent:
        act, dur = event
        return (act, self._closestCentroid1D(dur, self.centroids[act]))

    def bin_trace(self, trace: ServiceTimeTrace) -> BinnedServiceTimeTrace:
        return tuple(self.bin_event(evt) for evt in trace)

    def bin_log(self, traces: list[ServiceTimeTrace]) -> list[BinnedServiceTimeTrace]:
        return [self.bin_trace(trace) for trace in traces]


def extract_service_time_traces(
    log: pd.DataFrame,
    activity_key: str = constants.DEFAULT_NAME_KEY,
    start_time_key: str = constants.DEFAULT_START_TIMESTAMP_KEY,
    end_time_key: str = constants.DEFAULT_TIMESTAMP_KEY,
) -> list[ServiceTimeTrace]:
    """Extract a list of ServiceTimeTrace from the Event Log. This is an abstraction from the event log, where a case is considered a sequence (tuple)
    of tuples of 1) Activity and 2) Duration.

    The event log must be an interval event log, i.e. each event has a start and end timestamp.

    Args:
        log (pd.DataFrame): The event log.
        activity_key (str, optional): The key for the activity label in the event log. Defaults to constants.DEFAULT_NAME_KEY.
        start_time_key (str, optional): The key for the start timestamp in the event log. Defaults to constants.DEFAULT_START_TIMESTAMP_KEY.
        end_time_key (str, optional): The key for the end timestamp in the event log. Defaults to constants.DEFAULT_TIMESTAMP_KEY.

    Returns:
        list[ServiceTimeTrace]: The list of ServiceTimeTraces extracted from the event log.
    """
    # For each case a tuple containing for each event a tuple of 1) Activity and 2) Duration
    return (
        log.groupby("case:concept:name")
        .apply(
            lambda group_df: tuple(  # type: ignore [arg-type, return-value]
                (
                    evt[activity_key],
                    (evt[end_time_key] - evt[start_time_key]).total_seconds(),
                )
                for (_, evt) in group_df.sort_values(by=end_time_key).iterrows()
            )
        )
        .tolist()
    )


def extract_traces_activity_service_times(
    log: pd.DataFrame,
    binner: KMeans_Binner,
    activity_key: str = constants.DEFAULT_NAME_KEY,
    start_time_key: str = constants.DEFAULT_START_TIMESTAMP_KEY,
    end_time_key: str = constants.DEFAULT_TIMESTAMP_KEY,
) -> list[BinnedServiceTimeTrace]:
    """Extract the traces from the event log with a special focus on service times, i.e., a view concerned only with the executed activity, and how long its execution took. Only defined for Interval Event Logs, i.e. each event has a Start and Complete timestamp

    Args:
        log (EventLog): The event log
        binner (KMeans_Binner): The binner to use for binning the service times.
        num_bins (int, optional): The number of bins to cluster the service times into (Using K-Means). Defaults to 3 (intuitively "slow", "medium", "fast").
        traceid_key (str, optional): The key for the trace id in the event log. Defaults to xes.DEFAULT_TRACEID_KEY.
        activity_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.
        start_time_key (str, optional): The key in the event log for the start timestamp of the event. Defaults to xes.DEFAULT_START_TIMESTAMP_KEY.
        end_time_key (str, optional): The key in the event log for the completion timestamp of the event. Defaults to xes.DEFAULT_TIMESTAMP_KEY.
        seed (int | None, optional): The seed for the random number generator for numpy sampling and scipy kmeans++
    Returns:
        np.ndarray[ServiceTimeTrace]: A sequence of traces, represented as a tuple of Activities, but here the activities are tuples of the activity name and how long that activity took to complete. Same order as in the original event log.
    """
    return binner.bin_log(
        extract_service_time_traces(log, activity_key, start_time_key, end_time_key)
    )


@cache
def weighted_levenshtein_distance(
    trace1: BinnedServiceTimeTrace,
    trace2: BinnedServiceTimeTrace,
    rename_cost: Callable[[str, str], int],
    insertion_deletion_cost: Callable[[str], int],
    cost_time_match_rename: Callable[[int, int], int],
    cost_time_insert_delete: Callable[[int], int],
) -> int:
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

    def ins_del_cost(x: BinnedServiceTimeEvent) -> int:
        return insertion_deletion_cost(x[0]) + cost_time_insert_delete(x[1])

    def substitution_cost(x: BinnedServiceTimeEvent, y: BinnedServiceTimeEvent) -> int:
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


@cache
def custom_levenshtein_distance(
    trace1: BinnedServiceTimeTrace, trace2: BinnedServiceTimeTrace
) -> float:
    """Calculate the weighted levenshtein distance with the following weights:

        - Rename cost: 1
        - Insertion/Deletion cost: 1
        - Time Match/Rename cost: Absolute difference between the times
        - Time Insert/Delete cost: x (The value of the time)

        Thanks to not taking cost functions as inputs, caching (hashing) works correctly and saves time.

    Args:
        trace1 (BinnedServiceTimeTrace): The first trace.
        trace2 (BinnedServiceTimeTrace): The second trace.

    Returns:
        float: The computed weighted Levenshtein distance for these costs.
    """
    return weighted_levenshtein_distance(
        trace1,
        trace2,
        rename_cost=lambda x, y: 1,
        insertion_deletion_cost=lambda x: 1,
        cost_time_match_rename=lambda x, y: abs(x - y),
        cost_time_insert_delete=lambda x: x,
    )


@cache
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

        Thanks to not taking cost functions as inputs, caching (hashing) works correctly and saves time.

    Args:
        trace1 (BinnedServiceTimeTrace): The first trace.
        trace2 (BinnedServiceTimeTrace): The second trace.

    Returns:
        float: The computed postnormalized weighted Levenshtein distance for these costs.
    """
    return post_normalized_weighted_levenshtein_distance(
        trace1,
        trace2,
        rename_cost=lambda x, y: 1,
        insertion_deletion_cost=lambda x: 1,
        cost_time_match_rename=lambda x, y: abs(x - y),
        cost_time_insert_delete=lambda x: x,
    )


def post_normalized_weighted_levenshtein_distance(
    trace1: BinnedServiceTimeTrace,
    trace2: BinnedServiceTimeTrace,
    rename_cost: Callable[[str, str], int],
    insertion_deletion_cost: Callable[[str], int],
    cost_time_match_rename: Callable[[int, int], int],
    cost_time_insert_delete: Callable[[int], int],
) -> float:
    """Compute the post-normalized weighted Levenshtein distance. This is used for the calculation of a "time-aware" EMD.

    Args:
        trace1 (BinnedServiceTimeTrace): The first trace.
        trace2 (BinnedServiceTimeTrace): The second trace.
        rename_cost (Callable[[str, str], int]): Custom Cost.
        insertion_deletion_cost (Callable[[str], int]): Custom Cost.
        cost_time_match_rename (Callable[[int, int], int]): Custom Cost.
        cost_time_insert_delete (Callable[[int], int]): Custom Cost.

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
    distribution1: list[tuple[BinnedServiceTimeTrace, float]],
    distribution2: list[tuple[BinnedServiceTimeTrace, float]],
) -> float:
    """Calculate the Earth Mover's Distance between two populations of BinnedServiceTimeTraces.

    Args:
        distribution1 (list[tuple[BinnedServiceTimeTrace, float]]): The population for the first event log.
        distribution2 (list[tuple[BinnedServiceTimeTrace, float]]): The population for the second event log.

    Returns:
        float: The computed (Time-Aware) Earth Mover's Distance.
    """
    distances: dict[tuple[int, int], float] = dict()
    for i, (trace1, _) in enumerate(distribution1):
        for j, (trace2, _) in enumerate(distribution2):
            if (i, j) not in distances:
                distances[(i, j)] = custom_postnormalized_levenshtein_distance(
                    trace1,
                    trace2,
                )

    solver = wasserstein.EMD()
    dists = np.ones((len(distribution1), len(distribution2)))
    for i in range(len(distribution1)):
        for j in range(len(distribution2)):
            dists[i, j] = distances[(i, j)]

    return solver(
        [freq for _, freq in distribution1], [freq for _, freq in distribution2], dists
    )


def population_to_stochastic_language(
    population: list[BinnedServiceTimeTrace],
) -> list[tuple[BinnedServiceTimeTrace, float]]:
    """Convert a population (list of traces) to a stochastic language (list of traces with frequencies).

    Args:
        population (list[BinnedServiceTimeTrace]): The population to convert.

    Returns:
        list[tuple[BinnedServiceTimeTrace, float]]: The stochastic language.
    """
    pop_len = len(population)
    return [(trace, freq / pop_len) for trace, freq in Counter(population).items()]


def log_to_stochastic_language(
    log: pd.DataFrame,
    binner: KMeans_Binner,
    activity_key: str = constants.DEFAULT_NAME_KEY,
    start_time_key: str = constants.DEFAULT_START_TIMESTAMP_KEY,
    end_time_key: str = constants.DEFAULT_TIMESTAMP_KEY,
) -> list[tuple[BinnedServiceTimeTrace, float]]:
    """Extract the stochastic language of BinnedServiceTimeTraces from an event log.

    This is done by extracting ServiceTimeTraces, then binning the durations and finally computing the relative frequencies.

    Args:
        log (pd.DataFrame): The event log.
        binner (KMeans_Binner): The binner to use to bin the activity service times.
        activity_key (str, optional): The key for the activity label in the event log. Defaults to constants.DEFAULT_NAME_KEY.
        start_time_key (str, optional): The key for the start timestamp in the event log. Defaults to constants.DEFAULT_START_TIMESTAMP_KEY.
        end_time_key (str, optional): The key for the end timestamp. Defaults to constants.DEFAULT_TIMESTAMP_KEY.

    Returns:
        list[tuple[BinnedServiceTimeTrace, float]]: The extracted stochastic language.
    """
    population: list[BinnedServiceTimeTrace] = extract_traces_activity_service_times(
        log, binner, activity_key, start_time_key, end_time_key
    )

    return population_to_stochastic_language(population)


def compare_logs_emd(
    log_1: pd.DataFrame,
    log_2: pd.DataFrame,
    binner: Optional[KMeans_Binner] = None,
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
        binner (KMeans_Binner, optional): The binner to use to bin the activtiy service times. If none provided, one will be trained on the data using `num_bins` bins.
        num_bins (int, optional): The number of bins to cluster the service times into (Using K-Means). Defaults to 3 (intuitively "slow", "medium", "fast").
        activity_key (str, optional): The name of the column containing the label of the executed activity. Defaults to "concept:name".
        start_time_key (str, optional): The name of the column containing the start timestamp of the event. Defaults to "start_timestamp".
        end_time_key (str, optional): The name of the column containing the (end-) timestamp of the event. Defaults to "time:timestamp".
        seed (int | None, optional): The seed used for the clustering of the timestamps for binning. Defaults to None (no seeds used).

    Returns:
        float: The earth mover's distance between the two event logs.
    """

    if binner is None:
        binner = KMeans_Binner(
            extract_service_time_traces(
                log_1, activity_key, start_time_key, end_time_key
            )
            + extract_service_time_traces(
                log_2, activity_key, start_time_key, end_time_key
            ),
            num_bins,
            seed,
        )

    dist1: list[tuple[BinnedServiceTimeTrace, float]] = log_to_stochastic_language(
        log_1, binner, activity_key, start_time_key, end_time_key
    )

    dist2: list[tuple[BinnedServiceTimeTrace, float]] = log_to_stochastic_language(
        log_2, binner, activity_key, start_time_key, end_time_key
    )

    emd = calc_timing_emd(dist1, dist2)
    return emd


def _sample_with_replacement(
    items: list[BinnedServiceTimeTrace], n: int
) -> list[BinnedServiceTimeTrace]:
    sampled_indices = np.random.choice(range(len(items)), n, replace=True)
    return [items[idx] for idx in sampled_indices]


def process_comparison_emd(
    log_1: pd.DataFrame,
    log_2: pd.DataFrame,
    num_bins: int = 3,
    bootstrapping_dist_size: int = 10_000,
    resample_size: Optional[int] = None,
    traceid_key: str = constants.DEFAULT_TRACEID_KEY,
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
        traceid_key (str, optional): The key for the case id in the event log. Defaults to constants.DEFAULT_TRACEID_KEY.
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
    traces_2 = extract_service_time_traces(
        log_2, activity_key, start_time_key, end_time_key
    )

    binner = KMeans_Binner(traces_1 + traces_2, num_bins, seed)

    log_1_traces = extract_traces_activity_service_times(
        log_1, binner, activity_key, start_time_key, end_time_key
    )  # We sample from in the bootstrapping

    log_1_stochastic_language = population_to_stochastic_language(log_1_traces)

    emd = calc_timing_emd(
        log_1_stochastic_language,
        log_to_stochastic_language(  # log_2_stochastic_language
            log_2, binner, activity_key, start_time_key, end_time_key
        ),
    )

    # Bootstrap a p-value
    # Compute samples of EMD's of the logs with themselves to gauge a "normal" EMD
    self_emds = []

    if resample_size is None:
        resample_size = log_len(log_1, traceid_key)

    for _ in tqdm(
        range(bootstrapping_dist_size), desc="Bootstrapping Distribution for P-Value"
    ):
        # Compute an EMD of the event log to itself
        language = population_to_stochastic_language(
            _sample_with_replacement(log_1_traces, resample_size)
        )
        self_emds.append(calc_timing_emd(log_1_stochastic_language, language))

    # Clear the cache for the levenshtein distance
    custom_postnormalized_levenshtein_distance.cache_clear()

    num_larger_or_equal_bootstrap_dists = len([d for d in self_emds if d >= emd])
    return num_larger_or_equal_bootstrap_dists / bootstrapping_dist_size
