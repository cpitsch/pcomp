"""
Core functionality pertaining to EMD computations for various kinds of samples of the data.
"""

import logging
from collections import Counter
from dataclasses import dataclass
from itertools import zip_longest
from timeit import default_timer
from typing import Callable, Generic, Literal, TypeVar

import numpy as np
import ot  # type: ignore
import wasserstein  # type: ignore

from pcomp.utils import create_progress_bar, pretty_format_duration
from pcomp.utils.typing import Numpy1DArray, NumpyMatrix

T = TypeVar("T")

# Literal Types
EMDBackend = Literal["wasserstein", "ot", "pot"]


@dataclass
class StochasticLanguage(Generic[T]):
    variants: list[T]
    frequencies: Numpy1DArray[np.float_]


def compute_emd(
    distribution_1: StochasticLanguage[T],
    distribution_2: StochasticLanguage[T],
    cost_fn: Callable[[T, T], float],
    backend: EMDBackend = "wasserstein",
    show_progress_bar: bool = True,
) -> float:
    """Compute the Earth Mover's Distance between two distributions.

    Args:
        distribution_1 (StochasticLanguage[T]): The first distribution. All distinct
            behavior with their relative frequencies.
        distribution_2 (StochasticLanguage[T]): The second distribution.
        cost_fn (Callable[[T, T], float]): A function to compute the transport cost
            between two items.
        backend (EMDBackend, optional): The backend to use for EMD computation. Defaults
            to "wasserstein" (use the "wasserstein" module). Alternatively, "ot" or
            "pot" will use the "Python Optimal Transport" package.
        show_progress_bar (bool, optional): Show a progress bar for distance
            computation. Defaults to True.

    Returns:
        float: The computed Earth Mover's Distance.
    """
    logger = logging.getLogger("@pcomp")

    dists_start = default_timer()

    dists = compute_distance_matrix(
        distribution_1.variants,
        distribution_2.variants,
        cost_fn,
        show_progress_bar=show_progress_bar,
    )

    dists_end = default_timer()

    logs_emd = emd(
        distribution_1.frequencies,
        distribution_2.frequencies,
        dists,
        backend=backend,
    )
    emds_end = default_timer()

    dists_dur = dists_end - dists_start
    emds_dur = emds_end - dists_end
    total_time = emds_end - dists_start
    logger.info(
        f"compute_emd:Distances between logs took {pretty_format_duration(dists_end - dists_start)} ({(dists_dur / total_time * 100):.2f}%)"
    )
    logger.info(
        f"compute_emd:EMD between logs took {pretty_format_duration(emds_end - dists_end)} ({(emds_dur / total_time * 100):.2f}%)"
    )
    logger.info(
        f"compute_emd:Logs EMD took a total of {pretty_format_duration(total_time)}"
    )

    return logs_emd


def emd(
    freqs_1: Numpy1DArray[np.float_],
    freqs_2: Numpy1DArray[np.float_],
    dists: NumpyMatrix[np.float_],
    backend: EMDBackend = "wasserstein",
    fall_back: bool = True,
) -> float:
    """A wrapper around the EMD computation.

    Args:
        freqs_1 (Numpy1DArray[np.float_]): 1D histogram of the first distribution. All
            positive, sums up to 1.
        freqs_2 (Numpy1DArray[np.float_]): 1D histogram of the second distribution. All
            positive, sums up to 1.
        dists (NumpyMatrix[np.float_]): The cost matrix.
        backend ("wasserstein" | "ot" | "pot"): The backend to use to compute the EMD.
            Defaults to "wasserstein" (use the wasserstein package). Alternatively,
            "ot"/"pot" refers to the Python Optimal Transport package.
        fall_back (bool, optional): If the wasserstein package is used and an error is
            thrown, fall back to the ot package. Defaults to True.
    Returns:
        float: The computed Earth Mover's Distance.
    """
    if backend == "wasserstein":
        try:
            solver = wasserstein.EMD()
            return solver(freqs_1, freqs_2, dists)
        except Exception as e:
            logging.getLogger("@pcomp").warning(
                f"Error thrown by wasserstein package: \"{e}\". {' Falling back to ot package.' if fall_back else ''}",
            )
            if fall_back:
                # Apparently, the wasserstein package sometimes runs into issues on small inputs
                # In that case, we fall back to the ot package
                return ot.emd2(freqs_1, freqs_2, dists)
            else:
                raise e
    elif backend in ["ot", "pot"]:
        # This seems to be slower than the wasserstein package
        # But this could be due to different settings, such as num processes, etc.
        return ot.emd2(freqs_1, freqs_2, dists)
    else:
        raise ValueError(
            f"Invalid backend: {backend}. Must be 'wasserstein', 'ot', or 'pot'."
        )


def population_to_stochastic_language(
    population: list[T],
) -> StochasticLanguage[T]:
    """Convert a population to a stochastic language object.

    Args:
        population (list[T]): The population to convert.

    Returns:
        StochasticLanguage[T]: The stochastic language.
    """
    pop_len = len(population)
    stochastic_language = [
        (item, freq / pop_len) for item, freq in Counter(population).items()
    ]

    return StochasticLanguage(
        variants=[item for item, _ in stochastic_language],
        frequencies=np.array(
            [freq for _, freq in stochastic_language], dtype=np.float_
        ),
    )


def compute_emd_for_sample(
    dists: NumpyMatrix[np.float_],
    reference_frequencies: Numpy1DArray[np.float_],
    resample_size: int,
    emd_backend: EMDBackend = "wasserstein",
) -> float:
    """Sample a sample of size `resample_size` from the population with replacement and
    compute the EMD between the sample and the source population.

    Args:
        dists (NumpyMatrix[np.float_]): The distance matrix of the source population to
            itself.
        reference_frequencies (Numpy1DArray[np.float_]): The 1D histogram of the source
            population.
        resample_size (int): The size of the sample to draw from the population.
        emd_backend (EMDBackend, optional): The backend to use for EMD computation.
            Defaults to "wasserstein" (use the "wasserstein" module).

    Returns:
        float: The computed EMD.
    """
    # TODO: This doesn't respect the frequencies of the variants. All variants (rows) are created equal...
    sample_indices = np.random.choice(dists.shape[0], resample_size, replace=True)

    deduplicated_indices, counts = np.unique(sample_indices, return_counts=True)
    dists_for_sample = dists[deduplicated_indices, :]

    return emd(
        counts / resample_size,
        reference_frequencies,
        dists_for_sample,
        backend=emd_backend,
    )


def compute_emd_for_index_sample(
    indices: Numpy1DArray[np.int_],
    dists: NumpyMatrix[np.float_],
    reference_frequencies: Numpy1DArray[np.float_],
    emd_backend: EMDBackend = "wasserstein",
) -> float:
    """Given a sample of indices of rows in the distance matrix, compute the EMD between
    the sample and the source population (columns).

    Args:
        indices (Numpy1DArray[np.int_]): The sampled indices (possibly containing
            duplicates)
        dists (NumpyMatrix[np.float_]): The distance matrix.
        reference_frequencies (Numpy1DArray[np.float_]): 1D Histogram of the source
            population. Used for EMD computation
        emd_backend (EMDBackend, optional): The backend to use for EMD computation.
            Defaults to "wasserstein" (Use the "wasserstein" package).

    Returns:
        float: The computed EMD.
    """
    deduplicated_indices, counts = np.unique(indices, return_counts=True)
    return emd(
        counts / len(indices),
        reference_frequencies,
        dists[deduplicated_indices],
        backend=emd_backend,
    )


def compute_emd_for_split_sample(
    dists: NumpyMatrix[np.float_], emd_backend: EMDBackend = "wasserstein"
) -> float:
    """Randomly split the population in two and compute the EMD between the two halves.

    Args:
        dists (NumpyMatrix[np.float_]): The distance matrix.
        emd_backend (EMDBackend, optional): The backend to use to compute the EMD.
            Defaults to "wasserstein" (use the "wasserstein" module).

    Returns:
        float: The computed EMD.
    """
    sample_1_indices = np.random.choice(
        dists.shape[0], dists.shape[0] // 2, replace=False
    )
    sample_2_indices = np.setdiff1d(
        range(dists.shape[0]), sample_1_indices
    )  # Complement of sample_1_indices

    deduplicated_indices_1, counts_1 = np.unique(sample_1_indices, return_counts=True)
    deduplicated_indices_2, counts_2 = np.unique(sample_2_indices, return_counts=True)

    return emd(
        counts_1 / deduplicated_indices_1.size,
        counts_2 / deduplicated_indices_2.size,
        dists[deduplicated_indices_1, :][:, deduplicated_indices_2],
        backend=emd_backend,
    )


def compute_distance_matrix(
    population_1: list[T],
    population_2: list[T],
    cost_fn: Callable[[T, T], float],
    show_progress_bar: bool = True,
) -> NumpyMatrix[np.float_]:
    """Compute the distance matrix for two populations.

    Args:
        population_1 (list[T]): The first population of items.
        population_2 (list[T]): The second population of items.
        cost_fn (Callable[[T, T], float]): The cost function to compute the distance
            between two items.
        show_progress_bar (bool, optional): Show a progress bar? Defaults to True.

    Returns:
        NumpyMatrix[np.float_]: The distance matrix. The (i, j)-th element is the
            distance between the i-th element of population_1 and the j-th element of
            population_2.
    """
    dists = np.empty((len(population_1), len(population_2)), dtype=float)

    with create_progress_bar(
        show_progress_bar,
        total=dists.shape[0] * dists.shape[1],
        desc=f"Computing Distance Matrix ({dists.shape[0]}x{dists.shape[1]})",
    ) as dists_progress_bar:
        # TODO: Distance calculation could be parallelized
        for i, item1 in enumerate(population_1):
            for j, item2 in enumerate(population_2):
                dists[i, j] = cost_fn(item1, item2)
                dists_progress_bar.update()

    return dists


Event = tuple[str, float | int]
Trace = tuple[Event, ...]


def compute_time_distance_component(trace_1: Trace, trace_2: Trace) -> float:
    """Compute the time distance component of the edit distance between two traces.
    Used as an alternative to including the time distance in edit distance cost function.

    Computed by first matching equally labelled events and summing up the absolute time
    differences.
    For duplicate labels, the time differences are sorted and matched in order of
    increasing duration.
    Then, the remaining events are also sorted by duration and matched in order of
    increasing duration.

    Args:
        trace_1 (Trace): The first trace.
        trace_2 (Trace): The second trace.

    Returns:
        float: The computed time distance component.
    """

    # Sort traces alphabetically, then by duration
    # Complexity (Tim-sort): O(n log n)
    t_1 = sorted(list(trace_1))
    t_2 = sorted(list(trace_2))

    # Now iterate through the traces matching equally labelled events
    # The secondary sort by duration ensures that within equally labelled activities,
    # we match small durations to small durations and large durations to large durations
    # Complexity: O(n)
    index_1 = 0
    index_2 = 0
    matched_cost = 0.0
    not_matched_durs_1: list[float] = []
    not_matched_durs_2: list[float] = []
    while index_1 < len(t_1) and index_2 < len(t_2):
        activity_1, duration_1 = t_1[index_1]
        activity_2, duration_2 = t_2[index_2]

        if activity_1 == activity_2:
            # We have a match, add the time difference to the distance
            matched_cost += abs(duration_1 - duration_2)
            index_1 += 1
            index_2 += 1
        elif activity_1 < activity_2:
            not_matched_durs_1.append(duration_1)
            index_1 += 1

        else:
            not_matched_durs_2.append(duration_2)
            index_2 += 1

    # For the non-matched events, we sort by timestamp for the same reason as above
    # Complexity (Tim-Sort): O(n log n)
    not_matched_durs_1 = sorted(not_matched_durs_1)
    not_matched_durs_2 = sorted(not_matched_durs_2)

    return matched_cost + sum(
        abs(dur_1 - dur_2)
        for dur_1, dur_2 in zip_longest(
            not_matched_durs_1, not_matched_durs_2, fillvalue=0.0
        )
    )
