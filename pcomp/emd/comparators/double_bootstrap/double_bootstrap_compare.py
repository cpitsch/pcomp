"""
Compare two event logs by first bootstrapping a distribution of distances between Log 1
and samples of itself. Then, bootstrap a distribution of distances between samples of Log 1
and Log 2. Then, use the mean of the second distribution to compute a p-value in the same way
as the classic bootstrap.

In essence, the key difference is that instead of using the EMD between Log 1 and Log 2,
we bootstrap a distribution of these distances, and use the mean instead.

NOTE: This technique has no foundation in statistics, and is just the result of experimentation.
Also, preliminary experiments showed that this technique _does not work well_.
"""
from abc import ABC, abstractmethod
from statistics import mean
from timeit import default_timer
from typing import Callable, Generic, Literal, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from pcomp.emd.comparators.bootstrap.bootstrap_comparator import (
    _log_bootstrapping_performance,
    bootstrap_emd_population_resample_split_sampling,
)
from pcomp.emd.core import (
    EMDBackend,
    compute_distance_matrix,
    emd,
    population_to_stochastic_language,
)
from pcomp.utils import log_len
from pcomp.utils.typing import Numpy1DArray
from pcomp.utils.utils import create_progress_bar

T = TypeVar("T")


DoubleBootstrapStyle = Literal["sample_smaller_log_size", "splitted_resampling"]


class DoubleBootstrapEMDComparator(ABC, Generic[T]):
    log_1: pd.DataFrame
    log_2: pd.DataFrame

    bootstrapping_dist_size: int
    verbose: bool
    cleanup_on_del: bool
    bootstrapping_style: DoubleBootstrapStyle
    emd_backend: EMDBackend
    seed: int | None

    behavior_1: list[T]
    behavior_2: list[T]
    log_1_self_emds: list[float]
    emds_log_1_log_2: list[float]

    def __init__(
        self,
        log_1: pd.DataFrame,
        log_2: pd.DataFrame,
        bootstrapping_dist_size: int = 10_000,
        verbose: bool = True,
        cleanup_on_del: bool = True,
        bootstrapping_style: DoubleBootstrapStyle = "sample_smaller_log_size",
        emd_backend: EMDBackend = "wasserstein",
        seed: int | None = None,
    ):
        """Create an instance.

        Args:
            log_1 (pd.DataFrame): The first event log in the comparison.
            log_2 (pd.DataFrame): The second event log in the comparison.
            bootstrapping_dist_size (int, optional): The number of samples for
                bootstrapping distributions. Defaults to 10_000.
            verbose (bool, optional): If True, show progress bars. Defaults to True.
            cleanup_on_del (bool, optional): If True, call `cleanup` upon destruction,
                e.g., when the object goes out of scope. Defaults to True.
            bootstrapping_style (DoubleBootstrapStyle, optional): The bootstrapping
                style to use. Defaults to "sample_smaller_log_size". The options are:

                - "sample_smaller_log_size": Commpare two samples of `log_1` of size `len(log_2)` for self-EMDs bootstrapping. Then,
                    draw samples of size `len(log_2)` from `log_1` and and `log_2` (with replacement) and compare them.
                - "splitted_resampling": Splitt `log_1` into two disjunct halves. Then, draw samples of size `resample_size` from each half
                    (with replacement) and compare them. Then, draw samples of size `len(log_2)` from `log_1` and and `log_2`
                    (with replacement) and compare them.

            emd_backend (EMDBackend, optional): The backend to use for EMD computation.
                Defaults to "wasserstein" (use the "wasserstein" module). Alternatively,
                "ot" or "pot" will use the "Python Optimal Transport" package.
            seed (int, optional): The seed to use for sampling in the bootstrapping phase
        """
        self.log_1 = log_1
        self.log_2 = log_2

        # Ensure that log_1 is the larger event log
        if log_len(log_1) < log_len(log_2):
            self.log_1, self.log_2 = self.log_2, self.log_1

        self.bootstrapping_dist_size = bootstrapping_dist_size

        self.verbose = verbose
        self.cleanup_on_del = cleanup_on_del
        self.bootstrapping_style = bootstrapping_style
        self.emd_backend = emd_backend

        self.seed = seed

    def __del__(self):
        if self.cleanup_on_del:
            self.cleanup()

    @abstractmethod
    def extract_representations(
        self, log_1: pd.DataFrame, log_2: pd.DataFrame
    ) -> tuple[list[T], list[T]]:
        """Extract the behavior from the event log and do all data processing (binning,
        etc.). This can be a list of traces, or a list of graphs, etc.

        Args:
            log_1 (pd.DataFrame): The first event log.
            log_2 (pd.DataFrame): The second event log.

        Returns:
            tuple[list[T], list[T]]: The behavior extracted from the first and second
                event log, respectively.
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
        """
        Cleanup function to call after the comparison is done. For instance, clear
        caches, etc.
        """
        pass

    @property
    def reference_emds_distribution(self) -> list[float]:
        """
        The boostrapped distribution of self-emds of log_1 to itself. Computed in `compare`.
        If `compare` has not been called, accessing this will raise a ValueError.
        """
        if not hasattr(self, "_reference_emds_distribution"):
            raise ValueError("Must call `compare` before accessing `pval`.")
        return self._reference_emds_distribution

    @property
    def logs_emds_distribution(self) -> list[float]:
        """
        The distribution of emds from log_1 to log_2. Computed in `compare`.
        If `compare` has not been called, accessing this will raise a ValueError.
        """
        if not hasattr(self, "_logs_emds_distribution"):
            raise ValueError("Must call `compare` before accessing `pval`.")
        return self._logs_emds_distribution

    @property
    def pval(self) -> float:
        """
        The p-value from the comparison. Computed in `compare`.
        If `compare` has not been called, accessing this will raise a ValueError.
        """
        if not hasattr(self, "_pval"):
            raise ValueError("Must call `compare` before accessing `pval`.")
        return self._pval

    def compare(self) -> float:
        """Apply the full pipeline to compare the event logs.

        1. Extract the representations from the event logs using `self.extract_representations`.
        2. Compute the EMD between the two representations.
        3. Bootstrap a Null-distribution of EMDs (EMDs of samples of log_1 to log_1).
        4. Bootstrap a distribution of EMDs from log_1 to log_2.
        5. Compute the p-value.

        Returns:
            float: The computed p-value.
        """
        self.behavior_1, self.behavior_2 = self.extract_representations(
            self.log_1, self.log_2
        )

        # DO_NEW_BOOTSTRAPPING = False
        if self.bootstrapping_style == "splitted_resampling":
            self._reference_emds_distribution = (
                bootstrap_emd_distribution_splitted_resampling(
                    self.behavior_1,
                    cost_fn=self.cost_fn,
                    bootstrapping_dist_size=self.bootstrapping_dist_size,
                    resample_size=min(len(self.behavior_1) // 2, len(self.behavior_2)),
                    seed=self.seed,
                    emd_backend=self.emd_backend,
                    show_progress_bar=self.verbose,
                )
            )
            self._logs_emds_distribution = bootstrap_emd_distribution_with_smaller_log(
                self.behavior_1,
                self.behavior_2,
                self.cost_fn,
                bootstrapping_dist_size=self.bootstrapping_dist_size,
                resample_size=min(len(self.behavior_1) // 2, len(self.behavior_2)),
                seed=self.seed,
                emd_backend=self.emd_backend,
                show_progress_bar=self.verbose,
            )
        elif self.bootstrapping_style == "sample_smaller_log_size":
            self._reference_emds_distribution = (
                bootstrap_emd_population_resample_split_sampling(
                    self.behavior_1,
                    self.cost_fn,
                    bootstrapping_dist_size=self.bootstrapping_dist_size,
                    resample_size=len(self.behavior_2),
                    seed=self.seed,
                    emd_backend=self.emd_backend,
                    show_progress_bar=self.verbose,
                )
            )

            self._logs_emds_distribution = bootstrap_emd_distribution_with_smaller_log(
                self.behavior_1,
                self.behavior_2,
                self.cost_fn,
                bootstrapping_dist_size=self.bootstrapping_dist_size,
                resample_size=len(self.behavior_2),
                seed=self.seed,
                emd_backend=self.emd_backend,
                show_progress_bar=self.verbose,
            )

        mean_logs_emds = mean(self._logs_emds_distribution)

        num_larger_or_equal_bootstrap_dists = len(
            [d for d in self._reference_emds_distribution if d >= mean_logs_emds]
        )
        self._pval = num_larger_or_equal_bootstrap_dists / self.bootstrapping_dist_size
        return self._pval

    def plot_result(self) -> Figure:
        """Plot the result of the comparison.
        Shows the two distributions of EMDs.

        Returns:
            Figure: The figure with the plot.
        """
        fig, ax = plt.subplots()

        reference_emds_distribution = self.reference_emds_distribution
        logs_emds_distribution = self.logs_emds_distribution

        data: pd.DataFrame = pd.DataFrame(
            {
                "EMD": reference_emds_distribution + logs_emds_distribution,
                "Distribution": ["Reference Distribution"]
                * len(reference_emds_distribution)
                + ["Logs EMD Distribution"] * len(logs_emds_distribution),
            }
        )

        sns.histplot(data, ax=ax, x="EMD", hue="Distribution", common_bins=False)

        logs_emds_mean = np.mean(logs_emds_distribution)
        ax.axvline(
            logs_emds_mean,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Mean Logs Distance",
        )
        return fig


def bootstrap_emd_distribution_with_smaller_log(
    population_1: list[T],
    population_2: list[T],
    cost_fn: Callable[[T, T], float],
    bootstrapping_dist_size: int,
    resample_size: int | None = None,
    seed: int | None = None,
    emd_backend: EMDBackend = "wasserstein",
    show_progress_bar: bool = True,
) -> list[float]:
    """Bootstrap a distribution of distances of log_1 to log_2.
    This is done by sampling |log_2|-sized samples (with replacement)
    from log_1 and comparing them to log_2.
    This is repeated `bootstrapping_dist_size` times.
    Args:

        population_1 (list[T]): The first population. A list of items.
        population_2 (list[T]): The second population. A list of items.
        cost_fn (Callable[[T, T], float]): A function to compute the cost between two items.
        bootstrapping_dist_size (int, optional): The number of EMDs to compute.
            Defaults to 10_000.
        resample_size (int, optional): The size of the samples in bootstrapping. If None,
            defaults to the size of log_2.
        seed (int, optional): The seed to use for sampling in the bootstrapping phase
        emd_backend (EMDBackend, optional): The backend to use to compute the EMD.
            Defaults to "wasserstein" (use the "wasserstein" module).
        show_progress_bar (bool, optional): Whether to show a progress bar for the
            sampling progress. Defaults to True.

    Returns:
        list[float]: The list of computed EMDs.
    """
    gen = np.random.default_rng(seed)
    resample_size = resample_size or len(population_2)

    emds: list[float] = []

    stoch_lang_1 = population_to_stochastic_language(population_1)
    stoch_lang_2 = population_to_stochastic_language(population_2)

    # Precompute all distances
    dists_start = default_timer()
    dists = compute_distance_matrix(
        stoch_lang_1.variants, stoch_lang_2.variants, cost_fn, show_progress_bar
    )
    dists_end = default_timer()

    progress_bar = create_progress_bar(
        show_progress_bar,
        total=bootstrapping_dist_size,
        desc="Bootstrapping EMDs",
    )

    samples_1 = gen.choice(
        dists.shape[0],  # Sample from the indices in log 1
        (
            bootstrapping_dist_size,
            resample_size,
        ),  # Sample of size len(behavior_2) for each sample
        replace=True,
        p=stoch_lang_1.frequencies,
    )

    for idx in range(bootstrapping_dist_size):
        sample_1 = samples_1[idx]

        deduplicated_indices_1, counts_1 = np.unique(sample_1, return_counts=True)

        emds.append(
            emd(
                counts_1 / resample_size,
                stoch_lang_2.frequencies,
                dists[deduplicated_indices_1],
                backend=emd_backend,
            )
        )
        progress_bar.update()
    progress_bar.close()
    emds_end = default_timer()

    _log_bootstrapping_performance(dists_start, dists_end, emds_end)

    return emds


def _split_range(
    high: int, rng: np.random.Generator | None = None
) -> tuple[Numpy1DArray[np.int_], Numpy1DArray[np.int_]]:
    gen = np.random.default_rng(rng)  # Either the passed in generator or a new one
    half_1 = gen.choice(high, high // 2, replace=False)
    half_2 = np.setdiff1d(np.arange(high, dtype=np.int_), half_1)

    return half_1, half_2


def bootstrap_emd_distribution_splitted_resampling(
    population: list[T],
    cost_fn: Callable[[T, T], float],
    bootstrapping_dist_size: int,
    resample_size: int,
    seed: int | None = None,
    emd_backend: EMDBackend = "wasserstein",
    show_progress_bar: bool = True,
) -> list[float]:
    """Bootstrap a distribution of EMDs of a population to itself.
    This is done by first randomly splitting the population into two disjunct halves.
    Then, the a sample of size `resample_size` is drawn from each half with replacement,
        and the EMD is computed.
    This is repeated `bootstrapping_dist_size` times.

    Args:
        population (list[T]): The population. A list of items.
        resample_size (int): The size of the samples in bootstrapping
        cost_fn (Callable[[T, T], float]): A function to compute the cost between two items.
        bootstrapping_dist_size (int, optional): The number of EMDs to compute. Defaults
            to 10_000.
        resample_size (int): The size of the samples in bootstrapping.
        seed (int, optional): The seed to use for sampling in the bootstrapping phase
        emd_backend (EMDBackend, optional): The backend to use to compute the EMD.
            Defaults to "wasserstein" (use the "wasserstein" module).
        show_progress_bar (bool, optional): Whether to show a progress bar for the
            sampling progress. Defaults to True.

    Returns:
        list[float]: The list of computed EMDs.
    """
    gen = np.random.default_rng(seed)

    emds: list[float] = []

    stochastic_lang = population_to_stochastic_language(population)

    # Precompute all distances
    dists_start = default_timer()
    dists = compute_distance_matrix(
        stochastic_lang.variants, stochastic_lang.variants, cost_fn, show_progress_bar
    )
    dists_end = default_timer()

    progress_bar = create_progress_bar(
        show_progress_bar,
        total=bootstrapping_dist_size,
        desc="Bootstrapping EMDs",
    )

    for _ in range(bootstrapping_dist_size):
        half_1, half_2 = _split_range(len(stochastic_lang.variants), gen)

        sample_1 = gen.choice(half_1, resample_size, replace=True)
        sample_2 = gen.choice(half_2, resample_size, replace=True)

        deduplicated_indices_1, counts_1 = np.unique(sample_1, return_counts=True)
        deduplicated_indices_2, counts_2 = np.unique(sample_2, return_counts=True)

        emds.append(
            emd(
                counts_1 / resample_size,
                counts_2 / resample_size,
                dists[deduplicated_indices_1][:, deduplicated_indices_2],
                backend=emd_backend,
            )
        )
        progress_bar.update()
    progress_bar.close()
    emds_end = default_timer()

    _log_bootstrapping_performance(dists_start, dists_end, emds_end)

    return emds
