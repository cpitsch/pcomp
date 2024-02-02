from collections import Counter
from typing import Callable, TypeVar, Generic
import numpy as np
import pandas as pd
import wasserstein  # type: ignore
from tqdm.auto import tqdm
from abc import ABC, abstractmethod

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
    ):
        """Create an instance.

        Args:
            log_1 (pd.DataFrame): The first event log in the comparison.
            log_2 (pd.DataFrame): The second event log in the comparison.
            bootstrapping_dist_size (int, optional): The number of samples to compute the Self-EMD for. Defaults to 10_000.
            resample_size (int | None, optional): The size of each sample for the Self-EMDs. Defaults to None.
            verbose (bool, optional): If True, show progress bars. Defaults to True.
        """
        self.log_1 = ensure_start_timestamp_column(log_1)
        self.log_2 = ensure_start_timestamp_column(log_2)
        self.bootstrapping_dist_size = bootstrapping_dist_size
        self.resample_size = resample_size
        self.verbose = verbose

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

        self_emds = bootstrap_emd_population(
            self.behavior_1,
            self.cost_fn,
            bootstrapping_dist_size=self.bootstrapping_dist_size,
            resample_size=len(self.behavior_1),
            show_progress_bar=self.verbose,
        )

        num_larger_or_equal_bootstrap_dists = len([d for d in self_emds if d >= emd])
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
