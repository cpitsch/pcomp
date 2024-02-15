import logging
from abc import ABC, abstractmethod
from collections import Counter
from itertools import zip_longest
from timeit import default_timer
from typing import Callable, Generic, Literal, TypeVar

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import ot  # type: ignore
import pandas as pd
import wasserstein  # type: ignore
from tqdm.auto import tqdm

from pcomp.utils import ensure_start_timestamp_column, pretty_format_duration
from pcomp.utils.typing import Numpy1DArray, NumpyMatrix

T = TypeVar("T")

# Literal Types
BootstrappingStyle = Literal["split sampling", "replacement sublogs"]
EMDBackend = Literal["wasserstein", "ot", "pot"]


class EMD_ProcessComparator(ABC, Generic[T]):
    log_1: pd.DataFrame
    log_2: pd.DataFrame

    behavior_1: list[T]
    behavior_2: list[T]

    def __init__(
        self,
        log_1: pd.DataFrame,
        log_2: pd.DataFrame,
        bootstrapping_dist_size: int = 10_000,
        resample_size: int | None = None,
        verbose: bool = True,
        cleanup_on_del: bool = True,
        bootstrapping_style: BootstrappingStyle = "replacement sublogs",
        emd_backend: EMDBackend = "wasserstein",
    ):
        """Create an instance.

        Args:
            log_1 (pd.DataFrame): The first event log in the comparison.
            log_2 (pd.DataFrame): The second event log in the comparison.
            bootstrapping_dist_size (int, optional): The number of samples to compute the Self-EMD for. Defaults to 10_000.
            resample_size (int | None, optional): The size of each sample for the Self-EMDs. Defaults to None.
            verbose (bool, optional): If True, show progress bars. Defaults to True.
            cleanup_on_del (bool, optional): If True, call `cleanup` upon destruction, e.g., when the object goes out of scope. Defaults to True.
            bootstrapping_style ("split sampling" | "replacement sublogs", optional): The strategy to use for bootstrapping the null distribution. The strategies work as follows:

            - "replacement sublogs": Randomly sample sublogs of `resample_size` of log_1, and compute their EMD to log_1. This is done `bootstrapping_dist_size` times.
            - "split sampling": Randomly split the log_1 in two, and compute the EMD between the two halves. This is done `bootstrapping_dist_size` times.

            emd_backend (EMDBackend, optional): The backend to use for EMD computation. Defaults to "wasserstein" (use the "wasserstein" module). Alternatively, "ot" or "pot" will
            use the "Python Optimal Transport" package.
        """
        self.log_1 = ensure_start_timestamp_column(log_1)
        self.log_2 = ensure_start_timestamp_column(log_2)
        self.bootstrapping_dist_size = bootstrapping_dist_size
        self.resample_size = resample_size
        self.verbose = verbose
        self.cleanup_on_del = cleanup_on_del
        self.bootstrapping_style = bootstrapping_style
        self.emd_backend = emd_backend

    def __del__(self):
        if self.cleanup_on_del:
            self.cleanup()

    @abstractmethod
    def extract_representations(
        self, log_1: pd.DataFrame, log_2: pd.DataFrame
    ) -> tuple[list[T], list[T]]:
        """Extract the behavior from the event log and do all data processing (binning, etc.). This can be a list of traces, or a list of graphs, etc.

        Args:
            log_1 (pd.DataFrame): The first event log.
            log_2 (pd.DataFrame): The second event log.

        Returns:
            tuple[list[T], list[T]]: The behavior extracted from the first and second event log, respectively.
        """
        pass

    @abstractmethod
    def cost_fn(self, item1: T, item2: T) -> float:
        """Compute the cost between two items.

        Args:
            item1 (T): The first item,
            item2 (T): The second item.

        Returns:
            float: The computed cost between the two items.
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup function to call after the comparison is done. For instance, clear caches, etc."""
        pass

    @property
    def logs_emd(self) -> float:
        """
        The Earth Mover's Distance between the two logs. Computed in `compare`.
        If `compare` has not been called, accessing this will raise a ValueError.
        """
        if not hasattr(self, "_logs_emd"):
            raise ValueError("Must call `compare` before accessing `logs_emd`.")
        return self._logs_emd

    @property
    def bootstrapping_distribution(self) -> list[float]:
        """
        The bootstrapping distribution of EMDs of the log to itself. Computed in `compare`.
        If `compare` has not been called, accessing this will raise a ValueError.
        """
        if not hasattr(self, "_bootstrapping_distribution"):
            raise ValueError(
                "Must call `compare` before accessing `bootstrapping_distribution`."
            )
        return self._bootstrapping_distribution

    def compare(self) -> float:
        """Apply the full pipeline to compare the event logs.

        1. Extract the representations from the event logs using `self.extract_representations`.
        2. Compute the EMD between the two representations.
        3. Bootstrap a Null distribution of EMDs (EMDs of samples of log_1 to log_1).
        4. Compute the p-value.
        Returns:
            float: The computed p-value.
        """
        self.behavior_1, self.behavior_2 = self.extract_representations(
            self.log_1, self.log_2
        )

        emd = compute_emd(
            population_to_stochastic_language(self.behavior_1),
            population_to_stochastic_language(self.behavior_2),
            self.cost_fn,
        )

        if self.bootstrapping_style == "replacement sublogs":
            self_emds = bootstrap_emd_population(
                self.behavior_1,
                self.cost_fn,
                bootstrapping_dist_size=self.bootstrapping_dist_size,
                resample_size=len(self.behavior_1),
                emd_backend=self.emd_backend,
                show_progress_bar=self.verbose,
            )
        elif self.bootstrapping_style == "split sampling":
            self_emds = bootstrap_emd_population_split_sampling(
                self.behavior_1,
                self.cost_fn,
                bootstrapping_dist_size=self.bootstrapping_dist_size,
                emd_backend=self.emd_backend,
                show_progress_bar=self.verbose,
            )
        else:
            raise ValueError(
                f"Invalid bootstrapping style: {self.bootstrapping_style}. Must be 'replacement sublogs' or 'split sampling'."
            )

        num_larger_or_equal_bootstrap_dists = len([d for d in self_emds if d >= emd])

        self._logs_emd = emd
        self._bootstrapping_distribution = self_emds

        return num_larger_or_equal_bootstrap_dists / self.bootstrapping_dist_size


def compute_emd(
    distribution1: list[tuple[T, float]],
    distribution2: list[tuple[T, float]],
    cost_fn: Callable[[T, T], float],
) -> float:
    """Compute the Earth Mover's Distance between two distributions.

    Args:
        distribution1 (list[tuple[T, float]]): The first distribution.
        distribution2 (list[tuple[T, float]]): The second distribution.
        cost_fn (Callable[[T, T], float]): A function to compute the transport cost between two items.

    Returns:
        float: The computed Earth Mover's Distance.
    """
    dists = np.empty((len(distribution1), len(distribution2)), dtype=float)
    for i, (item1, _) in enumerate(distribution1):
        for j, (item2, _) in enumerate(distribution2):
            dists[i, j] = cost_fn(item1, item2)

    return emd(
        np.array([freq for _, freq in distribution1]),
        np.array([freq for _, freq in distribution2]),
        dists,
    )


def emd(
    freqs_1: Numpy1DArray[np.float_],
    freqs_2: Numpy1DArray[np.float_],
    dists: NumpyMatrix[np.float_],
    backend: EMDBackend = "wasserstein",
) -> float:
    """A wrapper around the EMD computation call.

    Args:
        freqs_1 (Numpy1DArray[np.float_]): 1D histogram of the first distribution. All positive, sums up to 1.
        freqs_2 (Numpy1DArray[np.float_]): 1D histogram of the second distribution. All positive, sums up to 1.
        dists (NumpyMatrix[np.float_]): The cost matrix.
        backend ("wasserstein" | "ot" | "pot"): The backend to use to compute the EMD. Defaults to "wasserstein" (use the wasserstein package). "ot"/"pot" refers to the Python Optimal Transport package.

    Returns:
        float: The computed Earth Mover's Distance.
    """
    if backend == "wasserstein":
        solver = wasserstein.EMD()
        return solver(freqs_1, freqs_2, dists)
    elif backend in ["ot", "pot"]:
        # This seems to be slower than the wasserstein package
        # But this could be due to different settings, such as num processes, etc.
        return ot.emd2(freqs_1, freqs_2, dists)
    else:
        raise ValueError(
            f"Invalid backend: {backend}. Must be 'wasserstein', 'ot', or 'pot'."
        )


def _sample_with_replacement(items: list[T], n: int) -> list[T]:
    """Sample with replacement from a list of items.

    Args:
        items (list[T]): The items to sample from.
        n (int): The size of the sample.

    Returns:
        list[T]: The sampled items.
    """
    sampled_indices = np.random.choice(range(len(items)), n, replace=True)
    return [items[idx] for idx in sampled_indices]


def _split_sampling(items: list[T]) -> tuple[list[T], list[T]]:
    """Split a population into two random non-overlapping halves.

    Args:
        items (list[T]): The population to split.

    Returns:
        tuple[list[T], list[T]]: The two halves of the population.
    """
    sampled_indices = np.random.choice(
        range(len(items)), len(items) // 2, replace=False
    )
    population_1 = [items[idx] for idx in sampled_indices]
    population_2 = [
        item for idx, item in enumerate(items) if idx not in sampled_indices
    ]

    return population_1, population_2


def population_to_stochastic_language(
    population: list[T],
) -> list[tuple[T, float]]:
    """Convert a population to a stochastic language (list of items with relative frequencies).

    Args:
        population (list[T]): The population to convert.

    Returns:
        list[tuple[T, float]]: The stochastic language.
    """
    pop_len = len(population)
    return [(item, freq / pop_len) for item, freq in Counter(population).items()]


def compute_emd_for_sample(
    dists: NumpyMatrix[np.float_],
    reference_frequencies: Numpy1DArray[np.float_],
    resample_size: int,
    emd_backend: EMDBackend = "wasserstein",
) -> float:
    """Sample a sample of size `resample_size` from the population with replacement and compute the EMD between the sample and the source population.

    Args:
        dists (NumpyMatrix[np.float_]): The distance matrix of the source population to itself.
        reference_frequencies (Numpy1DArray[np.float_]): The 1D histogram of the source population.
        resample_size (int): The size of the sample to draw from the population.
        emd_backend (EMDBackend, optional): The backend to use for EMD computation. Defaults to "wasserstein" (use the "wasserstein" module).

    Returns:
        float: The computed EMD.
    """
    sample_indices = np.random.choice(dists.shape[0], resample_size, replace=True)

    deduplicated_indices, counts = np.unique(sample_indices, return_counts=True)
    dists_for_sample = dists[deduplicated_indices, :]

    return emd(
        counts / resample_size,
        reference_frequencies,
        dists_for_sample,
        backend=emd_backend,
    )


def compute_emd_for_split_sample(
    dists: NumpyMatrix[np.float_], emd_backend: EMDBackend = "wasserstein"
) -> float:
    """Randomly split the population in two and compute the EMD between the two halves.

    Args:
        dists (NumpyMatrix[np.float_]): The distance matrix.
        emd_backend (EMDBackend, optional): The backend to use to compute the EMD. Defaults to "wasserstein" (use the "wasserstein" module).

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
        cost_fn (Callable[[T, T], float]): The cost function to compute the distance between two items.
        show_progress_bar (bool, optional): Show a progress bar? Defaults to True.

    Returns:
        NumpyMatrix[np.float_]: The distance matrix. The (i, j)-th element is the distance between the i-th element of population_1 and the j-th element of population_2.
    """
    dists = np.empty((len(population_1), len(population_2)), dtype=float)

    if show_progress_bar:
        dists_progress_bar = tqdm(
            total=dists.shape[0] * dists.shape[1],
            desc="Computing Distance Matrix...",
        )

    # TODO: Could parallelize distance calculation
    for i, item1 in enumerate(population_1):
        for j, item2 in enumerate(population_2):
            dists[i, j] = cost_fn(item1, item2)
            if show_progress_bar:
                dists_progress_bar.update()
    if show_progress_bar:
        dists_progress_bar.close()

    return dists


def bootstrap_emd_population(
    population: list[T],
    cost_fn: Callable[[T, T], float],
    bootstrapping_dist_size: int = 10_000,
    resample_size: int | None = None,
    emd_backend: EMDBackend = "wasserstein",
    show_progress_bar: bool = True,
) -> list[float]:
    """Compute a distribution of EMDs from a population to samples of itself.
    Computed by sampling samples of size `resample_size` with replacement from the population.
    Then, an EMD is computed between the population and the sample.	This is repeated `bootstrapping_dist_size` times.

    Args:
        population (list[T]): The population. A list of items.
        cost_fn (Callable[[T, T], float]): A function to compute the cost between two items.
        bootstrapping_dist_size (int, optional): The number of EMDs to compute. Defaults to 10_000.
        resample_size (int | None, optional): The size of the samples. Defaults to None.
        emd_backend (EMDBackend, optional): The backend to use to compute the EMD. Defaults to "wasserstein" (use the "wasserstein" module).
        show_progress_bar (bool, optional): Whether to show a progress bar for the sampling progress. Defaults to True.

    Returns:
        list[float]: The list of computed EMDs.
    """

    if resample_size is None:
        resample_size = len(population)

    reference_stochastic_language = population_to_stochastic_language(population)
    reference_freqs = np.array([freq for _, freq in reference_stochastic_language])

    dists_start = default_timer()
    # Precompute all distances since statistically, every pair of traces will be needed at least once
    dists = compute_distance_matrix(population, population, cost_fn, show_progress_bar)
    dists_end = default_timer()

    if show_progress_bar:
        bootstrapping_progress = tqdm(
            range(bootstrapping_dist_size),
            desc="Bootstrapping EMD Null Distribution",
        )

    emds: list[float] = []
    for _ in range(bootstrapping_dist_size):
        emds.append(
            compute_emd_for_sample(
                dists, reference_freqs, resample_size, emd_backend=emd_backend
            )
        )
        if show_progress_bar:
            bootstrapping_progress.update()
    if show_progress_bar:
        bootstrapping_progress.close()

    emds_end = default_timer()

    _log_bootstrapping_performance(dists_start, dists_end, emds_end)

    return emds


def bootstrap_emd_population_split_sampling(
    population: list[T],
    cost_fn: Callable[[T, T], float],
    bootstrapping_dist_size: int = 10_000,
    emd_backend: EMDBackend = "wasserstein",
    show_progress_bar: bool = True,
) -> list[float]:
    """Compute a distribution of EMDs from a population to samples of itself.
    Computed by randomly splitting the population in two, and then computing the EMD between
    the two halves.	This is repeated `bootstrapping_dist_size` times.

    Args:
        population (list[T]): The population. A list of items.
        cost_fn (Callable[[T, T], float]): A function to compute the cost between two items.
        bootstrapping_dist_size (int, optional): The number of EMDs to compute. Defaults to 10_000.
        emd_backend (EMDBackend, optional): The backend to use to compute the EMD. Defaults to "wasserstein".
        show_progress_bar (bool, optional): Whether to show a progress bar for the sampling progress. Defaults to True.

    Returns:
        list[float]: The list of computed EMDs.
    """
    emds: list[float] = []

    dists_start = default_timer()
    # Precompute all distances since statistically, every pair of traces will be needed at least once
    dists = compute_distance_matrix(population, population, cost_fn, show_progress_bar)
    dists_end = default_timer()

    if show_progress_bar:
        progress_bar = tqdm(
            total=bootstrapping_dist_size, desc="Bootstrapping EMD Null Distribution"
        )

    for _ in range(bootstrapping_dist_size):
        emds.append(compute_emd_for_split_sample(dists, emd_backend=emd_backend))
        if show_progress_bar:
            progress_bar.update()

    emds_end = default_timer()

    _log_bootstrapping_performance(dists_start, dists_end, emds_end)

    return emds


def _log_bootstrapping_performance(
    dists_start: float, dists_end: float, emds_end: float
):
    """Log performance information about the bootstrapping stage using the logging module.

        Uses the "@pcomp" logger. (`logging.getLogger("@pcomp")`).

    Args:
        dists_start (float): The start time of the distance computation (Output of `timeit.default_timer()`).
        dists_end (float): The end time of the distance computation (Output of `timeit.default_timer()`).
        emds_end (float): The end time of the EMD computation (Output of `timeit.default_timer()`).
    """
    total_time = emds_end - dists_start
    dists_dur = dists_end - dists_start
    emds_dur = emds_end - dists_end

    logger = logging.getLogger("@pcomp")

    logger.info(
        f"bootstrap_emd_population:Distances took {pretty_format_duration(dists_dur)} ({(dists_dur / total_time * 100):.2f}%)"
    )
    logger.info(
        f"bootstrap_emd_population:EMDs took {pretty_format_duration(emds_end - dists_end)} ({(emds_dur / total_time * 100):.2f}%)"
    )
    logger.info(
        f"bootstrap_emd_population:Bootstrapping total time: {pretty_format_duration(total_time)}"
    )


Event = tuple[str, float | int]
Trace = tuple[Event, ...]


def compute_time_distance_component(trace_1: Trace, trace_2: Trace) -> float:
    """Compute the time distance component of the edit distance between two traces. Used as an
    alternative to including the time distance in edit distance  cost function.

    Computed by first matching equally labelled events and summing the absolute time differences. For
    duplicate labels, the time differences are sorted and matched in order of increasing duration.
    Then, the remaining events are also sorted by duration and matched in order of increasing duration.

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


def plot_emd_result(bootstrapping_distribution: list[float], logs_emd: float) -> Figure:
    """Plot the bootstrapping distribution and the EMD between the two logs.

    Args:
        bootstrapping_distribution (list[float]): The bootstrapping distribution of EMDs of the log to itself.
        logs_emd (float): The EMD between the two logs.

    Returns:
        plt.figure: The corresponding figure.
    """
    fig, ax = plt.subplots()
    ax.hist(
        bootstrapping_distribution,
        bins=50,
        edgecolor="black",
        alpha=0.7,
        label=r"$D_{l_1l_1}$",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Earth Mover's Distance")
    ax.set_ylabel("Frequency")
    ax.axvline(
        logs_emd,
        color="red",
        linestyle="--",
        linewidth=2,
        label=r"$d_{l_1l_2}$",
    )
    ax.legend(fontsize=12, loc="upper right")

    return fig
