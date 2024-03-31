from abc import ABC, abstractmethod
from timeit import default_timer
from typing import Callable, Generic, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from tqdm.auto import tqdm

from pcomp.emd.core import (
    EMDBackend,
    _log_bootstrapping_performance,
    compute_distance_matrix,
    compute_emd,
    emd,
    population_to_stochastic_language,
)
from pcomp.utils.typing import Numpy1DArray
from pcomp.utils.utils import ensure_start_timestamp_column

T = TypeVar("T")


class Permutation_Test_Comparator(ABC, Generic[T]):
    log_1: pd.DataFrame
    log_2: pd.DataFrame

    distribution_size: int
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
        distribution_size: int = 10_000,
        verbose: bool = True,
        cleanup_on_del: bool = True,
        emd_backend: EMDBackend = "wasserstein",
        seed: int | None = None,
    ):
        """Create an instance.

        Args:
            log_1 (pd.DataFrame): The first event log in the comparison.
            log_2 (pd.DataFrame): The second event log in the comparison.
            distribution_size (int, optional): The number of samples to compute the
                Self-EMD for. Defaults to 10_000.
            verbose (bool, optional): If True, show progress bars. Defaults to True.
            cleanup_on_del (bool, optional): If True, call `cleanup` upon destruction,
                e.g., when the object goes out of scope. Defaults to True.
            emd_backend (EMDBackend, optional): The backend to use for EMD computation.
                Defaults to "wasserstein" (use the "wasserstein" module). Alternatively, "ot" or "pot" will
            use the "Python Optimal Transport" package.
            seed (int, optional): The seed to use for sampling in the bootstrapping phase.
        """
        self.log_1 = ensure_start_timestamp_column(log_1)
        self.log_2 = ensure_start_timestamp_column(log_2)
        self.distribution_size = distribution_size

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
        """Cleanup function to call after the comparison is done. For instance, clearing caches, etc."""
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
    def permuation_distribution(self) -> Numpy1DArray[np.float_]:
        """
        The distribution of EMDs computed for the permutation test. Computed in `compare`.
        If `compare` has not been called, accessing this will raise a ValueError.
        """
        if not hasattr(self, "_permutation_distribution"):
            raise ValueError(
                "Must call `compare` before accessing `permutation_distribution`."
            )
        return self._permutation_distribution

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

        1. Extract the representations from the event logs using
            `self.extract_representations`.
        2. Compute the EMD between the two representations.
        3. Bootstrap a Null distribution of EMDs (EMDs of samples of log_1 to log_1).
        4. Compute the p-value.
        Returns:
            float: The computed p-value.
        """
        self.behavior_1, self.behavior_2 = self.extract_representations(
            self.log_1, self.log_2
        )

        # TODO: Could precompute the distance matrix for _everything_
        emd = compute_emd(
            population_to_stochastic_language(self.behavior_1),
            population_to_stochastic_language(self.behavior_2),
            self.cost_fn,
            show_progress_bar=self.verbose,
        )

        permutation_test_distribution = compute_permutation_test_distribution(
            self.behavior_1,
            self.behavior_2,
            self.cost_fn,
            self.distribution_size,
            self.seed,
            self.emd_backend,
        )

        # TODO: Video says > not >=, using > for now.
        # Although it will likely never really matter
        num_larger_dists = (permutation_test_distribution > emd).sum()

        self._logs_emd = emd
        self._permutation_distribution = permutation_test_distribution
        self._pval = num_larger_dists / self.distribution_size

        return self._pval

    def plot_result(self) -> Figure:
        """Plot the bootstrapping distribution and the EMD between the two logs.

        Returns:
            plt.figure: The corresponding figure.
        """
        fig, ax = plt.subplots()

        bootstrapping_distribution = self._permutation_distribution
        logs_emd = self.logs_emd

        ax.hist(
            bootstrapping_distribution,
            bins=50,
            edgecolor="black",
            alpha=0.7,
            label=r"$D_{l_1l_1}$",
        )
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


def compute_permutation_test_distribution(
    population_1: list[T],
    population_2: list[T],
    cost_fn: Callable[[T, T], float],
    distribution_size: int = 10_000,
    seed: int | None = None,
    emd_backend: EMDBackend = "wasserstein",
    show_progress_bar: bool = True,
) -> Numpy1DArray[np.float_]:
    """Compute the distribution for a permutation test.

    Args:
        population_1 (list[T]): The first population of behavior.
        population_2 (list[T]): The second population of behavior.
        cost_fn (Callable[[T, T], float]): The function to compute the cost between two
            items.
        distribution_size (int, optional): The size of the distribution to compute.
            Defaults to 10_000.
        seed (int | None, optional): Seed for random sampling. Defaults to None.
        emd_backend (EMDBackend, optional): The backend to compute for the EMD
            computation. Defaults to "wasserstein" (use wasserstein package).
            Alternatively, "pot" uses "Python Optimal Transport" for EMD computation.
        show_progress_bar (bool, optional): Show a progress bar? Defaults to True.

    Returns:
        Numpy1DArray[np.float_]: The distribution of permutation test EMDs
    """
    gen = np.random.default_rng(seed)  # If seed is None, it is random

    combined_population = population_1 + population_2

    stochastic_language_all = population_to_stochastic_language(combined_population)
    all_variants = [item for item, _ in stochastic_language_all]

    dists_start = default_timer()
    dists = compute_distance_matrix(
        all_variants, all_variants, cost_fn, show_progress_bar=show_progress_bar
    )
    dists_end = default_timer()

    unpermuted_matrix = np.tile(
        np.arange(len(combined_population)), (distribution_size, 1)
    )

    # Permute the rows to get our permutation samples
    samples = gen.permuted(unpermuted_matrix, axis=1)

    population_indices_to_variant_indices = np.array(
        [all_variants.index(item) for item in population_1 + population_2]
    )

    emds = np.empty(distribution_size, dtype=np.float_)
    for idx in tqdm(range(distribution_size), "Computing EMDs for Permuation Test"):
        sample_1 = samples[idx][: len(population_1)]
        sample_2 = samples[idx][len(population_1) :]

        # Translate indices in population to variant indices (index in the distance matrix)
        translated_sample_1 = population_indices_to_variant_indices[sample_1]
        translated_sample_2 = population_indices_to_variant_indices[sample_2]

        deduplicated_sample_1, counts_1 = np.unique(
            translated_sample_1, return_counts=True
        )
        deduplicated_sample_2, counts_2 = np.unique(
            translated_sample_2, return_counts=True
        )

        emds[idx] = emd(
            counts_1 / translated_sample_1.shape[0],
            counts_2 / translated_sample_2.shape[0],
            dists[deduplicated_sample_1][:, deduplicated_sample_2],
            backend=emd_backend,
        )

    emds_end = default_timer()

    _log_bootstrapping_performance(dists_start, dists_end, emds_end)
    return emds
