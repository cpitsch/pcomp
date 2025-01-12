import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from timeit import default_timer
from typing import Callable, Generic, TypeVar

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from pcomp.emd.comparators.permutation_test.permutation_test_comparator import (
    compute_symmetric_distance_matrix,
    project_large_distance_matrix,
)
from pcomp.emd.core import (
    EMDBackend,
    StochasticLanguage,
    emd,
    population_to_stochastic_language,
)
from pcomp.utils import create_progress_bar, pretty_format_duration
from pcomp.utils.typing import Numpy1DArray, NumpyMatrix

T = TypeVar("T")


@dataclass
class ClassicBootstrapTestComparisonResult:
    pvalue: float
    logs_emd: float
    bootstrap_distribution: Numpy1DArray[np.float_]
    runtime: float

    def plot(self) -> Figure:
        """Plot the bootstrapping distribution and the EMD between the two logs.

        Returns:
            plt.figure: The corresponding figure.
        """
        fig, ax = plt.subplots()

        bootstrapping_distribution = self.bootstrap_distribution
        logs_emd = self.logs_emd

        ax.hist(
            bootstrapping_distribution,
            bins=50,
            edgecolor="black",
            alpha=0.7,
            label=r"$D$",
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


class ClassicBootstrap_Comparator(ABC, Generic[T]):
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
        emd_backend: EMDBackend = "wasserstein",
        seed: int | None = None,
    ):
        """Create an instance. The classic bootstrap comparator performs a "classic" two-sample
        bootstrap test. This is done by pooling both event logs together and then computing the
        EMD between samples (with replacement) and the pooled observations.

        Args:
            log_1 (pd.DataFrame): The first event log in the comparison.
            log_2 (pd.DataFrame): The second event log in the comparison.
            bootstrapping_dist_size (int, optional): The number of samples to compute
                the Self-EMD for. Defaults to 10_000.
            verbose (bool, optional): If True, show progress bars. Defaults to True.
            cleanup_on_del (bool, optional): If True, call `cleanup` upon destruction,
                e.g., when the object goes out of scope. Defaults to True.
            emd_backend (EMDBackend, optional): The backend to use for EMD computation.
                Defaults to "wasserstein" (use the "wasserstein" module). Alternatively,
                "ot" or "pot" will use the "Python Optimal Transport" package.
            seed (int, optional): The seed to use for sampling in the bootstrapping
                phase.
        """
        self.log_1 = log_1
        self.log_2 = log_2
        self.bootstrapping_dist_size = bootstrapping_dist_size

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
        Cleanup function to call after the comparison is done.
        For instance, clearing caches, etc.
        """
        pass

    @property
    def comparison_result(self) -> ClassicBootstrapTestComparisonResult:
        """
        The object representing the result of the comparison. Computed in `compare`.
        If `compare` has not been called, accessing this will raise a ValueError.
        """
        if not hasattr(self, "_comparison_result"):
            raise ValueError("Must call `compare` before accessing comparison result.")
        return self._comparison_result

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
    def bootstrapping_distribution(self) -> Numpy1DArray[np.float_]:
        """
        The bootstrapping distribution of EMDs of the log to itself. Computed in
        `compare`. If `compare` has not been called, accessing this will raise a
        ValueError.
        """
        return self.comparison_result.bootstrap_distribution

    @property
    def pval(self) -> float:
        """
        The p-value from the comparison. Computed in `compare`.
        If `compare` has not been called, accessing this will raise a ValueError.
        """
        return self.comparison_result.pvalue

    def compare(self) -> ClassicBootstrapTestComparisonResult:
        """Apply the full pipeline to compare the event logs.

        1. Extract the representations from the event logs using
            `self.extract_representations`.
        2. Compute the EMD between the two representations.
        3. Bootstrap a Null distribution of EMDs using bootstrapping.
        4. Compute the p-value.
        Returns:
            ClassicBootstrapTestComparisonResult: The result of the comparison: The pvalue
                and the measures used to compute it.
        """
        start_time = default_timer()
        self.behavior_1, self.behavior_2 = self.extract_representations(
            self.log_1, self.log_2
        )

        stoch_lang_1 = population_to_stochastic_language(self.behavior_1)
        stoch_lang_2 = population_to_stochastic_language(self.behavior_2)
        combined_stoch_lang = population_to_stochastic_language(
            self.behavior_1 + self.behavior_2
        )

        dists_start = default_timer()
        large_distance_matrix = compute_symmetric_distance_matrix(
            combined_stoch_lang.variants, self.cost_fn, self.verbose
        )
        dists_end = default_timer()
        logging.getLogger("@pcomp").info(
            "Computing Complete Distance Matrix took "
            + pretty_format_duration(dists_end - dists_start)
        )

        self._logs_emd = emd(
            stoch_lang_1.frequencies,
            stoch_lang_2.frequencies,
            project_large_distance_matrix(
                large_distance_matrix,
                combined_stoch_lang.variants,
                stoch_lang_1.variants,
                stoch_lang_2.variants,
            ),
            self.emd_backend,
        )

        bootstrap_dist = bootstrap_classic_emd_population_precomputed_distances(
            self.behavior_1,
            self.behavior_2,
            combined_stoch_lang,
            large_distance_matrix,
            bootstrapping_dist_size=self.bootstrapping_dist_size,
            seed=self.seed,
            emd_backend=self.emd_backend,
            show_progress_bar=self.verbose,
        )

        num_larger_or_equal_bootstrap_dists = (bootstrap_dist >= self._logs_emd).sum()

        self._bootstrapping_distribution = bootstrap_dist
        self._pval = num_larger_or_equal_bootstrap_dists / self.bootstrapping_dist_size
        self._comparison_runtime = default_timer() - start_time

        self._comparison_result = ClassicBootstrapTestComparisonResult(
            self._pval,
            self._logs_emd,
            np.array(self._bootstrapping_distribution),
            self._comparison_runtime,
        )

        return self._comparison_result

    def plot_result(self) -> Figure:
        """Plot the bootstrapping distribution and the EMD between the two logs.

        Args:
            bootstrapping_distribution (list[float]): The bootstrapping distribution of
                EMDs of the log to itself.
            logs_emd (float): The EMD between the two logs.

        Returns:
            plt.figure: The corresponding figure.
        """
        return self.comparison_result.plot()


def bootstrap_emd_population_classic(
    behavior_1: list[T],
    behavior_2: list[T],
    cost_fn: Callable[[T, T], float],
    bootstrapping_dist_size: int = 10_000,
    seed: int | None = None,
    emd_backend: EMDBackend = "wasserstein",
    show_progress_bar: bool = True,
) -> Numpy1DArray[np.float_]:
    """Compute a bootstrapping distribution using standard bootstrapping. Computes the
    distances between all behavior and then computes the bootstrapping distribution. If
    some distances between or within `behavior_1` or `behavior_2` are already computed,
    it is likely beneficial to compute all distances at once (for instance, using
    `compute_symmetric_distance_matrix`) and then calling
    `bootstrap_classic_emd_population_precomputed_distances`.

    Args:
        behavior_1 (list[T]): The behavior extracted from the first event log.
            (|Log_1| items).
        behavior_2 (list[T]): The behavior extracted from the second event log.
            (|Log_2| items).
        cost_fn (Callable[[T, T], float]): The cost function to compute the distance
            between two items from `behavior_1` and/or `behavior_2`.
        bootstrapping_dist_size (int, optional): The number of bootstrap samples to
            compute. Defaults to 10_000.
        seed (int | None, optional): The seed to use for sampling. Defaults to None.
        emd_backend (EMDBackend, optional): The backend to use for EMD computation.
            Defaults to "wasserstein" (use the wasserstein package). Alternatively, "pot"
            uses the "Python Optimal Transport" package.
        show_progress_bar (bool, optional): Show a progress bar for the distance and
            bootstrap distribution computation? Defaults to True.

    Returns:
        Numpy1DArray[np.float_]: The computed EMDs.
    """
    stoch_lang_all = population_to_stochastic_language(behavior_1 + behavior_2)
    dists = compute_symmetric_distance_matrix(
        stoch_lang_all.variants, cost_fn, show_progress_bar=show_progress_bar
    )

    return bootstrap_classic_emd_population_precomputed_distances(
        behavior_1,
        behavior_2,
        stoch_lang_all,
        dists,
        bootstrapping_dist_size,
        seed,
        emd_backend,
        show_progress_bar=show_progress_bar,
    )


def bootstrap_classic_emd_population_precomputed_distances(
    behavior_1: list[T],
    behavior_2: list[T],
    distance_matrix_source_stoch_lang: StochasticLanguage,
    distance_matrix: NumpyMatrix[np.float_],
    bootstrapping_dist_size: int = 10_000,
    seed: int | None = None,
    emd_backend: EMDBackend = "wasserstein",
    show_progress_bar: bool = True,
) -> Numpy1DArray[np.float_]:
    """Compute a bootstrapping distribution using standard bootstrapping.
    Leverages precomputed distances in order to save on computation time.

    Args:
        behavior_1 (list[T]): The behavior extracted from the first event log.
            (|Log_1| items).

        behavior_2 (list[T]): The behavior extracted from the second event log.
            (|Log_2| items).
        distance_matrix_source_stoch_lang (StochasticLanguage): The stochastic
            language that was used to compute the distance matrix. Used for the
            frequencies and sampling range.
        distance_matrix (NumpyMatrix[np.float_]): The distance matrix.
        bootstrapping_dist_size (int, optional): The number of samples to compute.
            Defaults to 10_000.
        seed (int | None, optional): The seed to use for sampling. Defaults to None.
        emd_backend (EMDBackend, optional): The backend to use for EMD computation.
            Defaults to "wasserstein" (use the wasserstein package. Alternatively, "pot"
            uses "Python Optimal Transport" to compute the EMD.
        show_progress_bar (bool, optional): Show a progress bar for the computation?
            Defaults to True.

    Returns:
        Numpy1DArray[np.float_]: The computed EMDs
    """
    gen = np.random.default_rng(seed)

    emds_start = default_timer()

    samples = gen.choice(
        len(distance_matrix_source_stoch_lang.variants),
        (bootstrapping_dist_size, len(behavior_1) + len(behavior_2)),
        replace=True,
        p=distance_matrix_source_stoch_lang.frequencies,
    )

    emds = np.empty(bootstrapping_dist_size, dtype=np.float_)
    progress = create_progress_bar(
        show_progress_bar,
        total=bootstrapping_dist_size,
        desc="Computing Bootstrapping Distribution",
    )
    for idx in range(bootstrapping_dist_size):
        sample_1 = samples[idx][: len(behavior_1)]
        sample_2 = samples[idx][len(behavior_1) :]

        deduplicated_sample_1, counts_1 = np.unique(sample_1, return_counts=True)
        deduplicated_sample_2, counts_2 = np.unique(sample_2, return_counts=True)

        emds[idx] = emd(
            counts_1 / sample_1.shape[0],
            counts_2 / sample_2.shape[0],
            distance_matrix[deduplicated_sample_1][:, deduplicated_sample_2],
            backend=emd_backend,
        )
        progress.update()
    progress.close()
    emds_end = default_timer()
    logging.getLogger("@pcomp").info(
        "Computing Bootstrapping Distribution took "
        + pretty_format_duration(emds_end - emds_start)
    )
    return emds
