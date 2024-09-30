import logging
import math
from abc import ABC, abstractmethod
from timeit import default_timer
from typing import Callable, Generic, Literal, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from pcomp.emd.core import (
    EMDBackend,
    compute_distance_matrix,
    compute_emd,
    compute_emd_for_index_sample,
    emd,
    population_to_stochastic_language,
)
from pcomp.utils import (
    create_progress_bar,
    ensure_start_timestamp_column,
    log_len,
    pretty_format_duration,
)
from pcomp.utils.typing import Numpy1DArray, NumpyMatrix

T = TypeVar("T")

# Literal Types
BootstrappingStyle = Literal["replacement sublogs", "split sampling", "resample split"]


class BootstrapComparator(ABC, Generic[T]):
    log_1: pd.DataFrame
    log_2: pd.DataFrame

    bootstrapping_dist_size: int
    resample_size: int
    verbose: bool
    cleanup_on_del: bool
    boostrapping_style: BootstrappingStyle
    emd_backend: EMDBackend
    seed: int | None

    behavior_1: list[T]
    behavior_2: list[T]

    def __init__(
        self,
        log_1: pd.DataFrame,
        log_2: pd.DataFrame,
        bootstrapping_dist_size: int = 10_000,
        resample_size: int | float | None = None,
        verbose: bool = True,
        cleanup_on_del: bool = True,
        bootstrapping_style: BootstrappingStyle = "replacement sublogs",
        emd_backend: EMDBackend = "wasserstein",
        seed: int | None = None,
    ):
        """Create an instance.

        Args:
            log_1 (pd.DataFrame): The first event log in the comparison.
            log_2 (pd.DataFrame): The second event log in the comparison.
            bootstrapping_dist_size (int, optional): The number of samples to compute
                the Self-EMD for. Defaults to 10_000.
            resample_size (int | float | None, optional): The size of each sample for
                the Self-EMDs. If float, it describes the fraction of the size of the
                event log to use. Defaults to None.
            verbose (bool, optional): If True, show progress bars. Defaults to True.
            cleanup_on_del (bool, optional): If True, call `cleanup` upon destruction,
                e.g., when the object goes out of scope. Defaults to True.
            bootstrapping_style ("replacement sublogs" | "split sampling" | "resample split", optional):
                The strategy to use for bootstrapping the null distribution. The
                strategies work as follows:

                - "replacement sublogs": Randomly sample sublogs of `resample_size`
                of log_1, and compute their EMD to log_1. This is done
                `bootstrapping_dist_size` times. This is the approach used by Leemans
                et al. in "Statistical Tests and Association Measures for Business
                Processes"
                - "split sampling": Randomly split the log_1 in two halves and compute
                the EMD between them. This is done `bootstrapping_dist_size` times.
                - "resample split": Randomly sample 2 sublogs of `resample_size` of
                log_1 and compute their EMD. This is done `bootstrapping_dist_size`
                times. This is a kind of mixture of the "replacement sublogs" and
                "split sampling" approaches, taking the sampling with replacement from
                the first, and taking the "split" comparison idea from the latter, as
                opposed to comparing what effectively converges towards a subset.

            emd_backend (EMDBackend, optional): The backend to use for EMD computation.
                Defaults to "wasserstein" (use the "wasserstein" module). Alternatively,
                "ot" or "pot" will use the "Python Optimal Transport" package.
            seed (int, optional): The seed to use for sampling in the bootstrapping
                phase.
        """
        self.log_1 = ensure_start_timestamp_column(log_1)
        self.log_2 = ensure_start_timestamp_column(log_2)
        self.bootstrapping_dist_size = bootstrapping_dist_size

        loglen = log_len(self.log_1)
        if resample_size is None:
            self.resample_size = log_len(self.log_1)
        elif isinstance(resample_size, float):
            self.resample_size = math.floor(resample_size * loglen)
        else:
            self.resample_size = resample_size

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
        """Cleanup function to call after the comparison is done. For instance, clearing
        caches, etc."""
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
        The bootstrapping distribution of EMDs of the log to itself. Computed in
        `compare`.
        If `compare` has not been called, accessing this will raise a ValueError.
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
            show_progress_bar=self.verbose,
        )

        if self.bootstrapping_style == "replacement sublogs":
            self_emds = bootstrap_emd_population(
                self.behavior_1,
                self.cost_fn,
                bootstrapping_dist_size=self.bootstrapping_dist_size,
                resample_size=self.resample_size,
                seed=self.seed,
                emd_backend=self.emd_backend,
                show_progress_bar=self.verbose,
            )
        elif self.bootstrapping_style == "split sampling":
            self_emds = bootstrap_emd_population_split_sampling(
                self.behavior_1,
                self.cost_fn,
                bootstrapping_dist_size=self.bootstrapping_dist_size,
                seed=self.seed,
                emd_backend=self.emd_backend,
                show_progress_bar=self.verbose,
            )
        elif self.bootstrapping_style == "resample split":
            self_emds = bootstrap_emd_population_resample_split_sampling(
                self.behavior_1,
                self.cost_fn,
                bootstrapping_dist_size=self.bootstrapping_dist_size,
                resample_size=self.resample_size,
                seed=self.seed,
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


def bootstrap_emd_population(
    population: list[T],
    cost_fn: Callable[[T, T], float],
    bootstrapping_dist_size: int = 10_000,
    resample_size: int | None = None,
    seed: int | None = None,
    emd_backend: EMDBackend = "wasserstein",
    show_progress_bar: bool = True,
) -> list[float]:
    """Compute a distribution of EMDs from a population to samples of itself.
    Computed by sampling samples of size `resample_size` with replacement from the
    population. Then, an EMD is computed between the population and the sample.
    This is repeated `bootstrapping_dist_size` times.

    Args:
        population (list[T]): The population. A list of items.
        cost_fn (Callable[[T, T], float]): A function to compute the cost between two
            items.
        bootstrapping_dist_size (int, optional): The number of EMDs to compute. Defaults
            to 10_000.
        resample_size (int | None, optional): The size of the samples. Defaults to None.
        seed (int, optional): The seed to use for sampling. Defaults to None.
        emd_backend (EMDBackend, optional): The backend to use to compute the EMD.
            Defaults to "wasserstein" (use the "wasserstein" module).
        show_progress_bar (bool, optional): Whether to show a progress bar for the
            sampling progress. Defaults to True.

    Returns:
        list[float]: The list of computed EMDs.
    """
    gen = np.random.default_rng(seed)

    # Default resample size to log length
    resample_size = resample_size or len(population)

    reference_stoch_lang = population_to_stochastic_language(population)

    dists_start = default_timer()
    # Precompute all distances as statistically, every pair of traces is needed at least once
    dists = compute_distance_matrix(
        reference_stoch_lang.variants,
        reference_stoch_lang.variants,
        cost_fn,
        show_progress_bar,
    )
    dists_end = default_timer()

    with create_progress_bar(
        show_progress_bar,
        total=bootstrapping_dist_size,
        desc="Bootstrapping EMD Null Distribution",
    ) as bootstrapping_progress:

        def _compute_emd_with_pbar(row: Numpy1DArray[np.int_]) -> float:
            res = compute_emd_for_index_sample(
                row, dists, reference_stoch_lang.frequencies, emd_backend
            )
            bootstrapping_progress.update()
            return res

        # Get the samples for the entire bootstrapping stage, respecting the frequencies of the variants
        samples = gen.choice(
            dists.shape[0],
            (bootstrapping_dist_size, resample_size),
            replace=True,
            p=reference_stoch_lang.frequencies,
        )
        emds: Numpy1DArray[np.float_] = np.apply_along_axis(
            _compute_emd_with_pbar,
            1,
            samples,
        )

    emds_end = default_timer()

    _log_bootstrapping_performance(dists_start, dists_end, emds_end)

    return emds.tolist()


def bootstrap_emd_population_split_sampling(
    population: list[T],
    cost_fn: Callable[[T, T], float],
    bootstrapping_dist_size: int = 10_000,
    seed: int | None = None,
    emd_backend: EMDBackend = "wasserstein",
    show_progress_bar: bool = True,
) -> list[float]:
    """Compute a distribution of EMDs from a population to samples of itself.
    Computed by randomly splitting the population in two, and then computing the EMD
    between the two halves.
    This is repeated `bootstrapping_dist_size` times.

    Args:
        population (list[T]): The population. A list of items.
        cost_fn (Callable[[T, T], float]): A function to compute the cost between two
            items.
        bootstrapping_dist_size (int, optional): The number of EMDs to compute. Defaults
            to 10_000.
        seed (int, optional): The seed to use for sampling. Defaults to None.
        emd_backend (EMDBackend, optional): The backend to use to compute the EMD.
            Defaults to "wasserstein".
        show_progress_bar (bool, optional): Whether to show a progress bar for the
            sampling progress. Defaults to True.

    Returns:
        list[float]: The list of computed EMDs.
    """
    gen = np.random.default_rng(seed)

    emds: list[float] = []

    stochastic_lang = population_to_stochastic_language(population)

    dists_start = default_timer()
    # Precompute all distances since statistically, every pair of traces will be needed at least once
    dists = compute_distance_matrix(
        stochastic_lang.variants, stochastic_lang.variants, cost_fn, show_progress_bar
    )
    dists_end = default_timer()

    progress_bar = create_progress_bar(
        show_progress_bar,
        total=bootstrapping_dist_size,
        desc="Bootstrapping EMD Null Distribution",
    )

    population_indices_to_variant_indices = np.array(
        [stochastic_lang.variants.index(item) for item in population]
    )

    for _ in range(bootstrapping_dist_size):
        # Get the samples for the entire bootstrapping stage, respecting the frequencies of the variants
        # Ideally, would do all the sampling at once, but not sure how to do that with replacement off for each row separately

        # First sample from the population so that we respect the frequencies of the variants
        sample = gen.choice(
            len(population),  # Need to sample from the population.
            len(population) // 2,
            replace=False,
        )
        inverted_sample = np.setdiff1d(range(len(population)), sample)

        # Then translate to the index of the variant (index in distance matrix)
        translated_sample = population_indices_to_variant_indices[sample]
        translated_inverted_sample = population_indices_to_variant_indices[
            inverted_sample
        ]

        deduplicated_indices_1, counts_1 = np.unique(
            translated_sample, return_counts=True
        )
        deduplicated_indices_2, counts_2 = np.unique(
            translated_inverted_sample, return_counts=True
        )

        emds.append(
            emd(
                counts_1 / len(translated_sample),
                counts_2 / len(translated_inverted_sample),
                dists[deduplicated_indices_1][:, deduplicated_indices_2],
                backend=emd_backend,
            )
        )

        progress_bar.update()
    progress_bar.close()
    emds_end = default_timer()

    _log_bootstrapping_performance(dists_start, dists_end, emds_end)

    return emds


def bootstrap_emd_population_resample_split_sampling(
    population: list[T],
    cost_fn: Callable[[T, T], float],
    bootstrapping_dist_size: int = 10_000,
    resample_size: int | None = None,
    seed: int | None = None,
    emd_backend: EMDBackend = "wasserstein",
    show_progress_bar: bool = True,
) -> list[float]:
    """Bootstrap a distribution of EMDs of a population to itself.
    Computed by randomly drawing two samples with replacement and computing their EMD.
    This is repeated `bootstrapping_dist_size` times.

    Args:
        population (list[T]): The population. A list of items.
        cost_fn (Callable[[T, T], float]): A function to compute the cost between two
            items.
        bootstrapping_dist_size (int, optional): The number of EMDs to compute. Defaults
            to 10_000.
        resample_size (int | None, optional): The size of the samples. Defaults to None
            (half the population size).
        seed (int, optional): The seed to use for sampling. Defaults to None.
        emd_backend (EMDBackend, optional): The backend to use to compute the EMD.
            Defaults to "wasserstein".
        show_progress_bar (bool, optional): Whether to show a progress bar for the
            sampling progress. Defaults to True.

    Returns:
        list[float]: The list of computed EMDs.
    """
    gen = np.random.default_rng(seed)
    resample_size = resample_size or len(population) // 2

    emds: list[float] = []

    stochastic_lang = population_to_stochastic_language(population)

    dists_start = default_timer()
    # Precompute all distances since statistically, every pair of traces will be needed at least once
    dists = compute_distance_matrix(
        stochastic_lang.variants, stochastic_lang.variants, cost_fn, show_progress_bar
    )
    dists_end = default_timer()

    progress_bar = create_progress_bar(
        show_progress_bar,
        total=bootstrapping_dist_size,
        desc="Bootstrapping EMD Null Distribution",
    )

    samples_1, samples_2 = gen.choice(
        dists.shape[0],
        # Every iteration we need two samples of `resample_size` size
        (2, bootstrapping_dist_size, resample_size),
        replace=True,
        p=stochastic_lang.frequencies,
    )
    for idx in range(bootstrapping_dist_size):
        sample_1 = samples_1[idx]
        sample_2 = samples_2[idx]

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


def _log_bootstrapping_performance(
    dists_start: float, dists_end: float, emds_end: float
):
    """Log performance information about the bootstrapping stage using the logging module.

        Uses the "@pcomp" logger. (`logging.getLogger("@pcomp")`).

    Args:
        dists_start (float): The start time of the distance computation (Output of
            `timeit.default_timer()`).
        dists_end (float): The end time of the distance computation (Output of
            `timeit.default_timer()`).
        emds_end (float): The end time of the EMD computation (Output of
            `timeit.default_timer()`).
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
