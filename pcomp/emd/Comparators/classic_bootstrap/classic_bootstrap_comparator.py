import logging
from abc import ABC, abstractmethod
from timeit import default_timer
from typing import Callable, Generic, TypeVar

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from tqdm.auto import tqdm

from pcomp.emd.Comparators.permutation_test.permutation_test_comparator import (
    compute_symmetric_distance_matrix,
    project_large_distance_matrix,
)
from pcomp.emd.core import (
    EMDBackend,
    _log_bootstrapping_performance,
    compute_distance_matrix,
    emd,
    population_to_stochastic_language,
)
from pcomp.utils import ensure_start_timestamp_column
from pcomp.utils.typing import Numpy1DArray, NumpyMatrix
from pcomp.utils.utils import create_progress_bar, pretty_format_duration

T = TypeVar("T")


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
        """Create an instance.

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
        self.log_1 = ensure_start_timestamp_column(log_1)
        self.log_2 = ensure_start_timestamp_column(log_2)
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
        if not hasattr(self, "_bootstrapping_distribution"):
            raise ValueError(
                "Must call `compare` before accessing `bootstrapping_distribution`."
            )
        return self._bootstrapping_distribution

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
        3. Bootstrap a Null distribution of EMDs using bootstrapping.
        4. Compute the p-value.
        Returns:
            float: The computed p-value.
        """
        self.behavior_1, self.behavior_2 = self.extract_representations(
            self.log_1, self.log_2
        )

        stoch_lang_1 = population_to_stochastic_language(self.behavior_1)
        stoch_lang_2 = population_to_stochastic_language(self.behavior_2)
        combined_stoch_lang = population_to_stochastic_language(
            self.behavior_1 + self.behavior_2
        )
        combined_behavior = [item for item, _ in combined_stoch_lang]

        dists_start = default_timer()
        large_distance_matrix = compute_symmetric_distance_matrix(
            combined_behavior, self.cost_fn, self.verbose
        )
        dists_end = default_timer()
        logging.getLogger("@pcomp").info(
            "Computing Complete Distance Matrix took "
            + pretty_format_duration(dists_end - dists_start)
        )

        self._logs_emd = emd(
            np.array([freq for _, freq in stoch_lang_1]),
            np.array([freq for _, freq in stoch_lang_2]),
            project_large_distance_matrix(
                large_distance_matrix,
                combined_behavior,
                [item for item, _ in stoch_lang_1],
                [item for item, _ in stoch_lang_2],
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

        return self._pval

    def plot_result(self) -> Figure:
        """Plot the bootstrapping distribution and the EMD between the two logs.

        Args:
            bootstrapping_distribution (list[float]): The bootstrapping distribution of
                EMDs of the log to itself.
            logs_emd (float): The EMD between the two logs.

        Returns:
            plt.figure: The corresponding figure.
        """
        fig, ax = plt.subplots()

        bootstrapping_distribution = self.bootstrapping_distribution
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


def bootstrap_emd_population_classic(
    population_1: list[T],
    population_2: list[T],
    cost_fn: Callable[[T, T], float],
    bootstrapping_dist_size: int = 10_000,
    seed: int | None = None,
    emd_backend: EMDBackend = "wasserstein",
    show_progress_bar: bool = True,
) -> Numpy1DArray[np.float_]:
    gen = np.random.default_rng(seed)

    stoch_lang_all = population_to_stochastic_language(population_1 + population_2)
    variants_all = [item for item, _ in stoch_lang_all]
    freqs_all = [freq for _, freq in stoch_lang_all]

    dists_start = default_timer()
    dists = compute_distance_matrix(
        variants_all,
        variants_all,
        cost_fn,
        show_progress_bar=show_progress_bar,
    )
    dists_end = default_timer()

    samples_1 = gen.choice(
        len(variants_all),
        (bootstrapping_dist_size, len(population_1)),
        replace=True,
        p=freqs_all,
    )

    samples_2 = gen.choice(
        len(variants_all),
        (bootstrapping_dist_size, len(population_2)),
        replace=True,
        p=freqs_all,
    )

    emds = np.empty(bootstrapping_dist_size, dtype=np.float_)
    for idx in tqdm(range(bootstrapping_dist_size)):
        sample_1 = samples_1[idx]
        sample_2 = samples_2[idx]

        deduplicated_indices_1, counts_1 = np.unique(sample_1, return_counts=True)
        deduplicated_indices_2, counts_2 = np.unique(sample_2, return_counts=True)

        emds[idx] = emd(
            counts_1 / len(population_1),
            counts_2 / len(population_2),
            dists[deduplicated_indices_1][:, deduplicated_indices_2],
            backend=emd_backend,
        )

    emds_end = default_timer()
    _log_bootstrapping_performance(dists_start, dists_end, emds_end)

    return emds


def bootstrap_classic_emd_population_precomputed_distances(
    behavior_1: list[T],
    behavior_2: list[T],
    distance_matrix_source_stoch_lang: list[tuple[T, float]],
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
        distance_matrix_source_stoch_lang (list[tuple[T, float]]): The stochastic
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
        len(distance_matrix_source_stoch_lang),
        (bootstrapping_dist_size, len(behavior_1) + len(behavior_2)),
        replace=True,
        p=[freq for _, freq in distance_matrix_source_stoch_lang],
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
