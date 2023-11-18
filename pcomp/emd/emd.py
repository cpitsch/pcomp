# This is an application of the concept drift detection approach from the paper
# "Time-aware Concept Drift Detection Using the Earth Mover’s Distance" by Brockhoff et al.
# Here, we use the idea of this approach to compare two event logs w.r.t. timed control-flow

import math
from collections import Counter
from collections.abc import Callable
from functools import cache

import numpy as np
import pandas as pd
import wasserstein
from scipy.stats import mannwhitneyu
from sklearn.cluster import kmeans_plusplus
from tqdm.auto import tqdm

from pcomp.utils import constants, log_len, split_log_cases


import pydantic


class ServiceTimeEvent(pydantic.BaseModel):
    activity: str
    duration: float


class ServiceTimeTrace(pydantic.BaseModel):
    events: tuple[ServiceTimeEvent, ...]


def extract_traces_activity_service_times(
    log: pd.DataFrame,
    num_bins: int = 3,
    traceid_key: str = constants.DEFAULT_TRACEID_KEY,
    activity_key: str = constants.DEFAULT_NAME_KEY,
    start_time_key: str = constants.DEFAULT_START_TIMESTAMP_KEY,
    end_time_key: str = constants.DEFAULT_TIMESTAMP_KEY,
    seed: int | None = None,
) -> np.ndarray[ServiceTimeTrace]:
    """Extract the traces from the event log with a special focus on service times, i.e., a view concerned only with the executed activity, and how long its execution took. Only defined for Interval Event Logs, i.e. each event has a Start and Complete timestamp

    Args:
        log (EventLog): The event log
        num_bins (int, optional): The number of bins to cluster the service times into (Using K-Means). Defaults to 3 (intuitively "slow", "medium", "fast").
        traceid_key (str, optional): The key for the trace id in the event log. Defaults to xes.DEFAULT_TRACEID_KEY.
        activity_key (str, optional): The key for the activity value in the event log. Defaults to xes.DEFAULT_NAME_KEY.
        start_time_key (str, optional): The key in the event log for the start timestamp of the event. Defaults to xes.DEFAULT_START_TIMESTAMP_KEY.
        end_time_key (str, optional): The key in the event log for the completion timestamp of the event. Defaults to xes.DEFAULT_TIMESTAMP_KEY.
        seed (int | None, optional): The seed for the random number generator for numpy sampling and scipy kmeans++
    Returns:
        np.ndarray[ServiceTimeTrace]: A sequence of traces, represented as a tuple of Activities, but here the activities are tuples of the activity name and how long that activity took to complete. Same order as in the original event log.
    """

    # For each case a tuple containing for each event a tuple of 1) Activity and 2) Duration
    out = (
        log.groupby("case:concept:name")
        .apply(
            lambda group_df: ServiceTimeTrace(  # type: ignore [arg-type, return-value]
                events=tuple(
                    (
                        ServiceTimeEvent(
                            activity=evt[activity_key],
                            duration=(
                                evt[end_time_key] - evt[start_time_key]
                            ).total_seconds(),
                        )
                    )
                    for (_, evt) in group_df.sort_values(by=end_time_key).iterrows()
                )
            )
        )
        .to_numpy()
    )

    # Now cluster the durations for binning
    # Use pareto principle; 20% of cases represent 80% of interesting behavior; Sample 20% only
    if seed is not None:
        np.random.seed(seed)
    sample = np.random.choice(out, size=math.ceil(0.2 * log_len(log, traceid_key)))
    durations = np.array([dur for trace in sample for _, dur in trace])
    durations = durations.reshape(-1, 1)  # Reshape to 2D array for sklearn
    centroid, _ = kmeans_plusplus(
        durations, n_clusters=num_bins, n_local_trials=10, random_state=seed
    )

    # Bin the actual data
    def _closestCentroid1D(point, centroids):
        mindex = 0
        minval = centroids[0]
        for idx, centroid in enumerate(centroids):
            dist = abs(centroid - point)
            if dist < minval:
                mindex = idx
                minval = dist
        return mindex

    for index in range(len(out)):
        trace = out[index]
        out[index] = tuple(
            # Assigning
            (act, _closestCentroid1D(dur, centroid))
            for act, dur in trace
        )
    return out


@cache
def weightedLevenshteinDistance(
    trace1: tuple[tuple[str, int], ...],
    trace2: tuple[tuple[str, int], ...],
    rename_cost: Callable[[str, str], int],
    insertion_deletion_cost: Callable[[str], int],
    cost_time_match_rename: Callable[[int, int], int],
    cost_time_insert_delete: Callable[[int], int],
) -> int:
    """Compute the levenshtein distance with custom weights.

    Args:
        trace1 (tuple[str]): The first trace.
        trace2 (tuple[str]): The second trace.
        rename_cost (Callable[[str], float]): Custom Cost.
        insertion_deletion_cost (Callable[[str], float]): Custom Cost.
        cost_time_match_rename (Callable[[int], float]): Custom Cost.
        cost_time_insert_delete (Callable[[int], float]): Custom Cost.

    Returns:
        float: The computed weighted Levenshtein distance.
    """
    if trace1 == trace2:
        return 0
    elif len(trace1) == 0:
        # Insert the traces
        ret = 0
        for act, dur in trace2:
            ret += insertion_deletion_cost(act) + cost_time_insert_delete(dur)
        return ret
    elif len(trace2) == 0:
        ret = 0
        for act, dur in trace1:
            ret += insertion_deletion_cost(act) + cost_time_insert_delete(dur)
        return ret

    act1, time1 = trace1[-1]
    act2, time2 = trace2[-1]

    dist1 = weightedLevenshteinDistance(
        trace1[:-1],
        trace2[:-1],
        rename_cost,
        insertion_deletion_cost,
        cost_time_match_rename,
        cost_time_insert_delete,
    )
    dist2 = weightedLevenshteinDistance(
        trace1,
        trace2[:-1],
        rename_cost,
        insertion_deletion_cost,
        cost_time_match_rename,
        cost_time_insert_delete,
    )
    dist3 = weightedLevenshteinDistance(
        trace1[:-1],
        trace2,
        rename_cost,
        insertion_deletion_cost,
        cost_time_match_rename,
        cost_time_insert_delete,
    )

    distance = min(
        dist1 + rename_cost(act1, act2) + cost_time_match_rename(time1, time2),
        dist2 + insertion_deletion_cost(act2) + cost_time_insert_delete(time2),
        dist3 + insertion_deletion_cost(act1) + cost_time_insert_delete(time1),
    )
    if act1 == act2:  # Also allow match
        return min(dist1 + cost_time_match_rename(time1, time2), distance)
    else:
        return distance


def postNormalizedWeightedLevenshteinDistance(
    trace1: tuple[tuple[str, int], ...],
    trace2: tuple[tuple[str, int], ...],
    rename_cost: Callable[[str, str], int],
    insertion_deletion_cost: Callable[[str], int],
    cost_time_match_rename: Callable[[int, int], int],
    cost_time_insert_delete: Callable[[int], int],
) -> float:
    """Compute the post-normalized weighted Levenshtein distance. This is used for the calculation of a "time-aware" EMD.

    Args:
        trace1 (tuple[tuple[str, int]]): The first trace.
        trace2 (tuple[tuple[str, int]]): The second trace.
        rename_cost (Callable[[str, str], int]): Custom Cost.
        insertion_deletion_cost (Callable[[str], int]): Custom Cost.
        cost_time_match_rename (Callable[[int, int], int]): Custom Cost.
        cost_time_insert_delete (Callable[[int], int]  ): Custom Cost.

    Returns:
        float: The post-normalized weighted Levenshtein distance.
    """
    return weightedLevenshteinDistance(
        trace1,
        trace2,
        rename_cost=rename_cost,
        insertion_deletion_cost=insertion_deletion_cost,
        cost_time_match_rename=cost_time_match_rename,
        cost_time_insert_delete=cost_time_insert_delete,
    ) / max(len(trace1), len(trace2))


def calc_timing_emd(
    distribution1: list[tuple[ServiceTimeTrace, float]],
    distribution2: list[tuple[ServiceTimeTrace, float]],
) -> float:
    distances: dict[tuple[int, int], float] = dict()
    progress = tqdm(total=len(distribution1) * len(distribution2))
    for i, (trace1, _) in enumerate(distribution1):
        for j, (trace2, _) in enumerate(distribution2):
            if (i, j) not in distances:
                distances[(i, j)] = postNormalizedWeightedLevenshteinDistance(
                    trace1,
                    trace2,
                    rename_cost=lambda x, y: 1,
                    insertion_deletion_cost=lambda x: 1,
                    cost_time_match_rename=lambda x, y: abs(x - y),
                    cost_time_insert_delete=lambda x: x,
                )
            progress.update()

    solver = wasserstein.EMD()
    dists = np.ones((len(distribution1), len(distribution2)))
    for i in range(len(distribution1)):
        for j in range(len(distribution2)):
            dists[i, j] = distances[(i, j)]

    return solver(
        [freq for _, freq in distribution1], [freq for _, freq in distribution2], dists
    )


def compare_logs_emd(
    log_1: pd.DataFrame,
    log_2: pd.DataFrame,
    traceid_key: str = constants.DEFAULT_TRACEID_KEY,
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
        traceid_key (str, optional): The name of the column containing the case id. Defaults to "case:concept:name".
        activity_key (str, optional): The name of the column containing the label of the executed activity. Defaults to "concept:name".
        start_time_key (str, optional): The name of the column containing the start timestamp of the event. Defaults to "start_timestamp".
        end_time_key (str, optional): The name of the column containing the (end-) timestamp of the event. Defaults to "time:timestamp".
        seed (int | None, optional): The seed used for the clustering of the timestamps for binning. Defaults to None (no seeds used).

    Returns:
        float: The earth mover's distance between the two event logs.
    """

    pop1: Counter[ServiceTimeTrace] = Counter(
        extract_traces_activity_service_times(
            log_1, 3, traceid_key, activity_key, start_time_key, end_time_key, seed
        )
    )
    pop2: Counter[ServiceTimeTrace] = Counter(
        extract_traces_activity_service_times(
            log_2, 3, traceid_key, activity_key, start_time_key, end_time_key, seed
        )
    )

    len_log1 = log_len(log_1, traceid_key)
    len_log2 = log_len(log_2, traceid_key)

    dist1 = [(trace, freq / len_log1) for trace, freq in pop1.items()]
    dist2 = [(trace, freq / len_log2) for trace, freq in pop2.items()]

    emd = calc_timing_emd(dist1, dist2)
    weightedLevenshteinDistance.cache_clear()
    return emd


def process_comparison_emd(
    log_1: pd.DataFrame,
    log_2: pd.DataFrame,
    traceid_key: str = constants.DEFAULT_TRACEID_KEY,
    activity_key: str = constants.DEFAULT_NAME_KEY,
    start_time_key: str = constants.DEFAULT_START_TIMESTAMP_KEY,
    end_time_key: str = constants.DEFAULT_TIMESTAMP_KEY,
    seed: int | None = None,
) -> float:
    emd = compare_logs_emd(
        log_1, log_2, traceid_key, activity_key, start_time_key, end_time_key, seed
    )

    # Compute samples of EMD's of the logs with themselves to gauge a "normal" EMD
    emds = []
    for _ in range(5):
        # Compute an EMD of the event log to itself
        emds.append(
            compare_logs_emd(
                *split_log_cases(log_1, 0.5, traceid_key),
                traceid_key,
                activity_key,
                start_time_key,
                end_time_key,
                seed
            )
        )

    # Perform a test to compare the calculated EMD `emd` with the samples from `emds1`
    # Like this test if the distance we calculated could belong to the distribution of distances of the log to itself
    # If the p-value is small enough, this means that we calculated a distance to an event log that is different from the event log itself
    statistic, p_value = mannwhitneyu(
        emds, [emd], alternative="two-sided"
    )  # Alternatively, we could maybe use ttest_1samp: `statistic, p_value = ttest_1samp(emds, emd)`
    # statistic, p_value = ttest_1samp(emds, emd)
    return p_value
