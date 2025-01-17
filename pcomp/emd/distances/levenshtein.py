from collections.abc import Callable

from strsimpy.levenshtein import Levenshtein  # type: ignore
from strsimpy.weighted_levenshtein import WeightedLevenshtein  # type: ignore

from pcomp.emd.extraction import BinnedServiceTimeEvent, BinnedServiceTimeTrace, Trace


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


def custom_postnormalized_levenshtein_distance(
    trace1: BinnedServiceTimeTrace,
    trace2: BinnedServiceTimeTrace,
) -> float:
    """Calculate the postnmormalized weighted levenshtein distance with the following
    weights:

        - Rename cost: 1
        - Insertion/Deletion cost: 1
        - Time Match/Rename cost: Absolute difference between the times
        - Time Insert/Delete cost: x (The value of the time)

        After computing the distance, it is divided by the length of the longer trace.

        Thanks to not taking (lambda) cost functions as inputs, caching (hashing) can
        work correctly.

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
