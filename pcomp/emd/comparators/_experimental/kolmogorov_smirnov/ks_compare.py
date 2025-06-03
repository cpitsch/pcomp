"""
Compare two event logs by first bootstrapping a distribution of distances between Log 1
and samples of itself. Then, bootstrap a distribution of distances between samples of Log 1
and Log 2, sampling half their size in each step, with replacement. Then compute a p-value
from these distributions by applying a two-sample Kolmogorov-Smirnov test.

NOTE: This technique has no foundation in statistics, and is just the result of experimentation.
Also, preliminary experiments showed that this technique _does not work well_, and pretty
much always detects a difference.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from timeit import default_timer
from typing import Callable, Generic, Literal, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.stats import kstest  # type: ignore

from pcomp.emd.comparators.bootstrap.bootstrap_comparator import (
    _log_bootstrapping_performance,
    bootstrap_emd_population_resample_split_sampling,
    bootstrap_emd_population_split_sampling,
)
from pcomp.emd.core import (
    EMDBackend,
    compute_distance_matrix,
    emd,
    population_to_stochastic_language,
)
from pcomp.utils import create_progress_bar
from pcomp.utils.typing import NP_FLOAT, Numpy1DArray

Self_Bootstrapping_Style = Literal["split", "replacement"]

T = TypeVar("T")


@dataclass
class KSTest_ComparisonResult:
    reference_emds_distribution: Numpy1DArray[NP_FLOAT]
    logs_emds_distribution: Numpy1DArray[NP_FLOAT]
    pvalue: float
    runtime: float

    def plot(self) -> Figure:
        """Plot the computed distributions and the mean EMD of the logs emd distribution.

        Returns:
            Figure: The corresponding figure
        """
        fig, ax = plt.subplots()

        N_BINS = 25
        ALPHA = 0.7
        LINEWIDTH = 0.5
        ax.hist(
            self.reference_emds_distribution,
            label=r"$D_{l_1l_1}$",
            bins=N_BINS,
            edgecolor="black",
            alpha=ALPHA,
            linewidth=LINEWIDTH,
        )
        ax.hist(
            self.logs_emds_distribution,
            label=r"$D_{l_1l_2}$",
            bins=N_BINS,
            edgecolor="black",
            alpha=ALPHA,
            linewidth=LINEWIDTH,
        )
        ax.set_xlabel("Earth Mover's Distance")
        ax.set_ylabel("Frequency")

        ax.legend(fontsize=12, loc="upper right")

        return fig


class KolmogorovSmirnovBootstrapComparator(ABC, Generic[T]):
    log_1: pd.DataFrame
    log_2: pd.DataFrame

    bootstrapping_dist_size: int
    verbose: bool
    cleanup_on_del: bool
    emd_backend: EMDBackend
    seed: int | None

    behavior_1: list[T]
    behavior_2: list[T]

    def __init__(
        self,
        log_1: pd.DataFrame,
        log_2: pd.DataFrame,
        bootstrapping_dist_size: int = 10_000,
        verbose: bool = True,
        cleanup_on_del: bool = True,
        self_emds_bootstrapping_style: Self_Bootstrapping_Style = "replacement",
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
            self_emds_bootstrapping_style (Self_Bootstrapping_Style, optional): The
                bootstrapping style to use for the self-emds of log_1. Defaults to
                "replacement". Options are:

                - "replacement": Sample 2 sublogs (size 1/2 of log_1) of log_1 with
                    replacement. Compute their EMD. This is repeated
                    `bootstrapping_dist_size` times.
                - "split": Randomly split log_1 in two halves and compute their EMD.
                    This is repeated `bootstrapping_dist_size` times.

            emd_backend (EMDBackend, optional): The backend to use for EMD computation.
                Defaults to "wasserstein" (use the "wasserstein" module). Alternatively,
                "ot" or "pot" will use the "Python Optimal Transport" package.
            seed: (int, optional): The seed to use for sampling in the bootstrapping phase.
        """
        self.log_1 = log_1
        self.log_2 = log_2
        self.bootstrapping_dist_size = bootstrapping_dist_size
        self.self_emds_bootstrapping_style = self_emds_bootstrapping_style

        self.verbose = verbose
        self.cleanup_on_del = cleanup_on_del
        self.emd_backend = emd_backend

        self.seed = seed

    def __del__(self):
        if self.cleanup_on_del:
            self.cleanup()

    @abstractmethod
    def extract_representations(
        self, log_1: pd.DataFrame, log_2: pd.DataFrame
    ) -> tuple[list[T], list[T]]:
        """Extract the behavior from the event log and do all data processing
        (binning, etc.). This can be a list of traces, or a list of graphs, etc.

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
    def comparison_result(self) -> KSTest_ComparisonResult:
        """
        The object representing the result of the comparison. Computed in `compare`.
        If `compare` has not been called, accessing this will raise a ValueError.
        """
        if not hasattr(self, "_comparison_result"):
            raise ValueError(
                "Must call `compare` before accessing `comparison_result`."
            )
        return self._comparison_result

    @property
    def reference_emds_distribution(self) -> Numpy1DArray[NP_FLOAT]:
        """
        The boostrapped distribution of self-emds of log_1 to itself. Computed in `compare`.
        If `compare` has not been called, accessing this will raise a ValueError.
        """
        return self.comparison_result.reference_emds_distribution

    @property
    def logs_emds_distribution(self) -> Numpy1DArray[NP_FLOAT]:
        """
        The distribution of emds from log_1 to log_2. Computed in `compare`.
        If `compare` has not been called, accessing this will raise a ValueError.
        """
        return self.comparison_result.logs_emds_distribution

    @property
    def pval(self) -> float:
        """
        The p-value from the comparison. Computed in `compare`.
        If `compare` has not been called, accessing this will raise a ValueError.
        """
        return self.comparison_result.pvalue

    @property
    def comparison_runtime(self) -> float:
        """
        The duration of the the `compare` call.
        If accessed before calling `compare`, a ValueError will be raised.
        """
        return self.comparison_result.runtime

    def compare(self) -> KSTest_ComparisonResult:
        """Apply the full pipeline to compare the event logs.

        1. Extract the representations from the event logs using `self.extract_representations`.
        2. Compute the EMD between the two representations.
        3. Bootstrap a Null-distribution of EMDs (EMDs of samples of log_1 to log_1).
        4. Bootstrap a distribution of EMDs from log_1 to log_2.
        4. Compute the p-value.
        Returns:
            float: The computed p-value.
        """
        start_time = default_timer()
        self.behavior_1, self.behavior_2 = self.extract_representations(
            self.log_1, self.log_2
        )

        if self.self_emds_bootstrapping_style == "replacement":
            self._reference_emds_distribution = (
                bootstrap_emd_population_resample_split_sampling(
                    self.behavior_1,
                    self.cost_fn,
                    bootstrapping_dist_size=self.bootstrapping_dist_size,
                    seed=self.seed,
                    emd_backend=self.emd_backend,
                    show_progress_bar=self.verbose,
                )
            )
        elif self.self_emds_bootstrapping_style == "split":
            self._reference_emds_distribution = bootstrap_emd_population_split_sampling(
                self.behavior_1,
                self.cost_fn,
                bootstrapping_dist_size=self.bootstrapping_dist_size,
                seed=self.seed,
                emd_backend=self.emd_backend,
                show_progress_bar=self.verbose,
            )
        else:
            raise ValueError(
                f"Invalid bootstrapping style: {self.self_emds_bootstrapping_style}. Must be 'replacement' or 'split'."
            )

        self._logs_emds_distribution = bootstrap_emd_population_between_logs(
            self.behavior_1,
            self.behavior_2,
            self.cost_fn,
            bootstrapping_dist_size=self.bootstrapping_dist_size,
            seed=self.seed,
            emd_backend=self.emd_backend,
            show_progress_bar=self.verbose,
        )

        # Compare these distributions using a statistical test
        self._pval = kstest(
            self._reference_emds_distribution,
            self._logs_emds_distribution,
            # Null Hypothesis is that the logs are the same, i.e., the
            # EMDs between the logs are <= the self emds of log_1
            # If pvalue low enough, we can reject and say that the distances are
            # larger than the log_1 self distances --> different process
            # UPDATE: Apparently it's the other way around, so we need to use "greater"
            alternative="greater",
        )[1]

        comparison_runtime = default_timer() - start_time

        self._comparison_result = KSTest_ComparisonResult(
            np.array(self._reference_emds_distribution),
            np.array(self._logs_emds_distribution),
            self._pval,
            comparison_runtime,
        )
        return self._comparison_result

    def plot_result(self) -> Figure:
        """Plot the result of the comparison.
        Shows the two distributions of EMDs.

        Returns:
            Figure: The figure with the plot.
        """
        return self.comparison_result.plot()


def bootstrap_emd_population_between_logs(
    population_1: list[T],
    population_2: list[T],
    cost_fn: Callable[[T, T], float],
    bootstrapping_dist_size: int = 10_000,
    seed: int | None = None,
    emd_backend: EMDBackend = "wasserstein",
    show_progress_bar: bool = True,
) -> list[float]:
    """Bootstrap a distribution of EMDs between two populations.
    This is done by sampling a sub-population of each population of half its size and
    computing the EMD between them.
    This is repeated `bootstrapping_dist_size` times.

    Args:
        population_1 (list[T]): The first population. A list of items.
        population_2 (list[T]): The second population. A list of items.
        cost_fn (Callable[[T, T], float]): A function to compute the cost between two items.
        bootstrapping_dist_size (int, optional): The number of EMDs to compute. Defaults
            to 10_000.
        seed (int | None, optional): The seed to use for sampling. Defaults to None.
        emd_backend (EMDBackend, optional): The backend to use for EMD computation.
            Defaults to "wasserstein" (use the `wasserstein` package).
        show_progress_bar (bool, optional): Whether to show a progress bar for the
            sampling progress. Defaults to True.

    Returns:
        list[float]: The list of computed EMDs
    """
    gen = np.random.default_rng(seed)

    emds: list[float] = []

    stochastic_lang_1 = population_to_stochastic_language(population_1)
    stochastic_lang_2 = population_to_stochastic_language(population_2)

    # Precompute all distances since statistically, every pair of traces will be needed at least once
    dists_start = default_timer()
    dists = compute_distance_matrix(
        stochastic_lang_1.variants,
        stochastic_lang_2.variants,
        cost_fn,
        show_progress_bar,
    )
    dists_end = default_timer()

    progress_bar = create_progress_bar(
        show_progress_bar,
        total=bootstrapping_dist_size,
        desc="Bootstrapping EMD Null Distribution",
    )

    SAMPLE_SIZE_1 = len(population_1) // 2
    SAMPLE_SIZE_2 = len(population_2) // 2
    samples_1 = gen.choice(
        dists.shape[0],
        (bootstrapping_dist_size, SAMPLE_SIZE_1),
        replace=True,
        p=stochastic_lang_1.frequencies,
    )
    samples_2 = gen.choice(
        dists.shape[1],
        (bootstrapping_dist_size, SAMPLE_SIZE_2),
        replace=True,
        p=stochastic_lang_2.frequencies,
    )
    for idx in range(bootstrapping_dist_size):
        sample_1 = samples_1[idx]
        sample_2 = samples_2[idx]

        deduplicated_indices_1, counts_1 = np.unique(sample_1, return_counts=True)
        deduplicated_indices_2, counts_2 = np.unique(sample_2, return_counts=True)

        emds.append(
            emd(
                counts_1 / SAMPLE_SIZE_1,
                counts_2 / SAMPLE_SIZE_2,
                dists[deduplicated_indices_1][:, deduplicated_indices_2],
                backend=emd_backend,
            )
        )
        progress_bar.update()
    progress_bar.close()
    emds_end = default_timer()
    _log_bootstrapping_performance(dists_start, dists_end, emds_end)

    return emds
