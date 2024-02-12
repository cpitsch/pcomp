from collections import Counter
from typing import Callable, Literal, TypeVar, Generic
import numpy as np
import pandas as pd
import wasserstein  # type: ignore
from tqdm.auto import tqdm
from abc import ABC, abstractmethod
from itertools import zip_longest
import matplotlib.pyplot as plt

from pcomp.utils import ensure_start_timestamp_column

T = TypeVar("T")


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
        bootstrapping_style: Literal[
            "split sampling", "replacement sublogs"
        ] = "replacement sublogs",
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
        """
        self.log_1 = ensure_start_timestamp_column(log_1)
        self.log_2 = ensure_start_timestamp_column(log_2)
        self.bootstrapping_dist_size = bootstrapping_dist_size
        self.resample_size = resample_size
        self.verbose = verbose
        self.cleanup_on_del = cleanup_on_del
        self.bootstrapping_style = bootstrapping_style

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
        if not hasattr(self, "_logs_emd"):
            raise ValueError("Must call `compare` before accessing `logs_emd`.")
        return self._logs_emd

    @property
    def bootstrapping_distribution(self) -> list[float]:
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
                show_progress_bar=self.verbose,
            )
        elif self.bootstrapping_style == "split sampling":
            self_emds = bootsrap_emd_population_split_sampling(
                self.behavior_1,
                self.cost_fn,
                bootstrapping_dist_size=self.bootstrapping_dist_size,
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

    solver = wasserstein.EMD()
    return solver(
        [freq for _, freq in distribution1],
        [freq for _, freq in distribution2],
        dists,
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


def bootstrap_emd_population(
    population: list[T],
    cost_fn: Callable[[T, T], float],
    bootstrapping_dist_size: int = 10_000,
    resample_size: int | None = None,
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
        show_progress_bar (bool, optional): Whether to show a progress bar for the sampling progress. Defaults to True.

    Returns:
        list[float]: The list of computed EMDs.
    """
    emds: list[float] = []
    reference_stochastic_language = population_to_stochastic_language(population)

    if show_progress_bar:
        progress_bar = tqdm(
            total=bootstrapping_dist_size, desc="Bootstrapping EMD Null Distribution"
        )

    if resample_size is None:
        resample_size = len(population)

    for _ in range(bootstrapping_dist_size):
        stoch_language = population_to_stochastic_language(
            _sample_with_replacement(population, resample_size)
        )
        emds.append(compute_emd(reference_stochastic_language, stoch_language, cost_fn))

        if show_progress_bar:
            progress_bar.update()

    return emds


def bootsrap_emd_population_split_sampling(
    population: list[T],
    cost_fn: Callable[[T, T], float],
    bootstrapping_dist_size: int = 10_000,
    show_progress_bar: bool = True,
) -> list[float]:
    """Compute a distribution of EMDs from a population to samples of itself.
    Computed by randomly splitting the population in two, and then computing the EMD between
    the two halves.	This is repeated `bootstrapping_dist_size` times.

    Args:
        population (list[T]): The population. A list of items.
        cost_fn (Callable[[T, T], float]): A function to compute the cost between two items.
        bootstrapping_dist_size (int, optional): The number of EMDs to compute. Defaults to 10_000.
        show_progress_bar (bool, optional): Whether to show a progress bar for the sampling progress. Defaults to True.

    Returns:
        list[float]: The list of computed EMDs.
    """
    emds: list[float] = []

    if show_progress_bar:
        progress_bar = tqdm(
            total=bootstrapping_dist_size, desc="Bootstrapping EMD Null Distribution"
        )

    for _ in range(bootstrapping_dist_size):
        sample_1, sample_2 = _split_sampling(population)
        lang_1 = population_to_stochastic_language(sample_1)
        lang_2 = population_to_stochastic_language(sample_2)
        emds.append(compute_emd(lang_1, lang_2, cost_fn))

        if show_progress_bar:
            progress_bar.update()

    return emds


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


def plot_emd_result(
    bootstrapping_distribution: list[float], logs_emd: float
) -> plt.figure:
    fig, ax = plt.subplots()
    ax.hist(
        bootstrapping_distribution,
        bins=50,
        edgecolor="black",
        alpha=0.7,
        label=r"$D_{l_1l_1}$",
    )
    ax.xticks([])
    ax.yticks([])
    ax.xlabel("Earth Mover's Distance")
    ax.ylabel("Frequency")
    ax.axvline(
        logs_emd,
        color="red",
        linestyle="--",
        linewidth=2,
        label=r"$d_{l_1l_2}$",
    )
    ax.legend(fontsize=12, loc="upper right")

    return fig
