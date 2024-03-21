"""
Core functionality pertaining to the standard EMD Bootstrapping comparison.

This includes the EMD_ProcessComparator abstract class which can be implemented
for any data extraction and distance functions.
Apart from that, other important helper functions are also defined. 
"""

import logging
import math
from abc import ABC, abstractmethod
from collections import Counter
from itertools import zip_longest
from timeit import default_timer
from typing import Callable, Generic, Literal, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import ot  # type: ignore
import pandas as pd
import wasserstein  # type: ignore
from matplotlib.figure import Figure

from pcomp.utils import ensure_start_timestamp_column, log_len, pretty_format_duration
from pcomp.utils.typing import Numpy1DArray, NumpyMatrix
from pcomp.utils.utils import create_progress_bar

T = TypeVar("T")

# Literal Types
BootstrappingStyle = Literal["replacement sublogs", "split sampling", "resample split"]
EMDBackend = Literal["wasserstein", "ot", "pot"]


class EMD_ProcessComparator(ABC, Generic[T]):
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
            bootstrapping_dist_size (int, optional): The number of samples to compute the Self-EMD for. Defaults to 10_000.
            resample_size (int | float | None, optional): The size of each sample for the Self-EMDs. If float, it describes the fraction of the size of the event log to use. Defaults to None.
            verbose (bool, optional): If True, show progress bars. Defaults to True.
            cleanup_on_del (bool, optional): If True, call `cleanup` upon destruction, e.g., when the object goes out of scope. Defaults to True.
            bootstrapping_style ("split sampling" | "replacement sublogs", optional): The strategy to use for bootstrapping the null distribution. The strategies work as follows:

              - "replacement sublogs": Randomly sample sublogs of `resample_size` of log_1, and compute their EMD to log_1. This is done `bootstrapping_dist_size` times.
              - "split sampling": Randomly split the log_1 in two, and compute the EMD between the two halves. This is done `bootstrapping_dist_size` times.
              - "resample split": Randomly sample 2 sublogs of `resample_size` of log_1 and compute their EMD. This is done `bootstrapping_dist_size` times.

            emd_backend (EMDBackend, optional): The backend to use for EMD computation. Defaults to "wasserstein" (use the "wasserstein" module). Alternatively, "ot" or "pot" will
            use the "Python Optimal Transport" package.
            seed (int, optional): The seed to use for sampling in the bootstrapping phase.
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
            bootstrapping_distribution (list[float]): The bootstrapping distribution of EMDs of the log to itself.
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


def compute_emd(
    distribution1: list[tuple[T, float]],
    distribution2: list[tuple[T, float]],
    cost_fn: Callable[[T, T], float],
    backend: EMDBackend = "wasserstein",
    show_progress_bar: bool = True,
) -> float:
    """Compute the Earth Mover's Distance between two distributions.

    Args:
        distribution1 (list[tuple[T, float]]): The first distribution. All distinct behavior with their relative frequencies.
        distribution2 (list[tuple[T, float]]): The second distribution.
        cost_fn (Callable[[T, T], float]): A function to compute the transport cost between two items.
        backend (EMDBackend, optional): The backend to use for EMD computation. Defaults to "wasserstein" (use the "wasserstein" module). Alternatively, "ot" or "pot" will
            use the "Python Optimal Transport" package.
        show_progress_bar (bool, optional): Show a progress bar for distance computation. Defaults to True.

    Returns:
        float: The computed Earth Mover's Distance.
    """
    logger = logging.getLogger("@pcomp")

    dists_start = default_timer()

    dists = compute_distance_matrix(
        [trace for trace, _ in distribution1],
        [trace for trace, _ in distribution2],
        cost_fn,
        show_progress_bar=show_progress_bar,
    )

    dists_end = default_timer()

    logs_emd = emd(
        np.array([freq for _, freq in distribution1]),
        np.array([freq for _, freq in distribution2]),
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
    """A wrapper around the EMD computation call.

    Args:
        freqs_1 (Numpy1DArray[np.float_]): 1D histogram of the first distribution. All positive, sums up to 1.
        freqs_2 (Numpy1DArray[np.float_]): 1D histogram of the second distribution. All positive, sums up to 1.
        dists (NumpyMatrix[np.float_]): The cost matrix.
        backend ("wasserstein" | "ot" | "pot"): The backend to use to compute the EMD. Defaults to "wasserstein"
            (use the wasserstein package). Alternatively, "ot"/"pot" refers to the Python Optimal Transport package.
        fall_back (bool, optional): If the wasserstein package is used and an error is thrown, fall back to the ot package. Defaults to True.
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
    """Given a sample of indices of rows in the distance matrix, compute the EMD between the sample and the source population (columns).

    Args:
        indices (Numpy1DArray[np.int_]): The sampled indices (possibly containing duplicates)
        dists (NumpyMatrix[np.float_]): The distance matrix.
        reference_frequencies (Numpy1DArray[np.float_]): 1D Histogram of the source population. Used for EMD computation
        emd_backend (EMDBackend, optional): The backend to use for EMD computation. Defaults to "wasserstein" (Use the "wasserstein" package).

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
    Computed by sampling samples of size `resample_size` with replacement from the population.
    Then, an EMD is computed between the population and the sample.	This is repeated `bootstrapping_dist_size` times.

    Args:
        population (list[T]): The population. A list of items.
        cost_fn (Callable[[T, T], float]): A function to compute the cost between two items.
        bootstrapping_dist_size (int, optional): The number of EMDs to compute. Defaults to 10_000.
        resample_size (int | None, optional): The size of the samples. Defaults to None.
        seed (int, optional): The seed to use for sampling. Defaults to None.
        emd_backend (EMDBackend, optional): The backend to use to compute the EMD. Defaults to "wasserstein" (use the "wasserstein" module).
        show_progress_bar (bool, optional): Whether to show a progress bar for the sampling progress. Defaults to True.

    Returns:
        list[float]: The list of computed EMDs.
    """
    gen = np.random.default_rng(seed) if seed is not None else None

    if seed is not None:
        gen = np.random.default_rng(seed)

    resample_size = resample_size or len(population)

    reference_stochastic_language = population_to_stochastic_language(population)
    reference_freqs = np.array([freq for _, freq in reference_stochastic_language])
    # Essentially "de-duplicated" reference population
    reference_behavior = [item for item, _ in reference_stochastic_language]

    dists_start = default_timer()
    # Precompute all distances since statistically, every pair of traces will be needed at least once
    dists = compute_distance_matrix(
        reference_behavior, reference_behavior, cost_fn, show_progress_bar
    )
    dists_end = default_timer()

    with create_progress_bar(
        show_progress_bar,
        total=bootstrapping_dist_size,
        desc="Bootstrapping EMD Null Distribution",
    ) as bootstrapping_progress:

        def _compute_emd_with_pbar(row: Numpy1DArray[np.int_]) -> float:
            res = compute_emd_for_index_sample(row, dists, reference_freqs, emd_backend)
            bootstrapping_progress.update()
            return res

        # Get the samples for the entire bootstrapping stage, respecting the frequencies of the variants
        samples = (gen or np.random).choice(
            dists.shape[0],
            (bootstrapping_dist_size, resample_size),
            replace=True,
            p=reference_freqs,
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
    Computed by randomly splitting the population in two, and then computing the EMD between
    the two halves.	This is repeated `bootstrapping_dist_size` times.

    Args:
        population (list[T]): The population. A list of items.
        cost_fn (Callable[[T, T], float]): A function to compute the cost between two items.
        bootstrapping_dist_size (int, optional): The number of EMDs to compute. Defaults to 10_000.
        seed (int, optional): The seed to use for sampling. Defaults to None.
        emd_backend (EMDBackend, optional): The backend to use to compute the EMD. Defaults to "wasserstein".
        show_progress_bar (bool, optional): Whether to show a progress bar for the sampling progress. Defaults to True.

    Returns:
        list[float]: The list of computed EMDs.
    """
    gen = np.random.default_rng(seed) if seed is not None else None

    emds: list[float] = []

    stochastic_lang = population_to_stochastic_language(population)
    behavior = [item for item, _ in stochastic_lang]

    dists_start = default_timer()
    # Precompute all distances since statistically, every pair of traces will be needed at least once
    dists = compute_distance_matrix(behavior, behavior, cost_fn, show_progress_bar)
    dists_end = default_timer()

    progress_bar = create_progress_bar(
        show_progress_bar,
        total=bootstrapping_dist_size,
        desc="Bootstrapping EMD Null Distribution",
    )

    population_indices_to_variant_indices = np.array(
        [behavior.index(item) for item in population]
    )

    for _ in range(bootstrapping_dist_size):
        # Get the samples for the entire bootstrapping stage, respecting the frequencies of the variants
        # Ideally, would do all the sampling at once, but not sure how to do that with replacement off for each row separately

        # First sample from the population so that we respect the frequencies of the variants
        sample = (gen or np.random).choice(
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
        cost_fn (Callable[[T, T], float]): A function to compute the cost between two items.
        bootstrapping_dist_size (int, optional): The number of EMDs to compute. Defaults to 10_000.
        resample_size (int | None, optional): The size of the samples. Defaults to None (half the population size).
        seed (int, optional): The seed to use for sampling. Defaults to None.
        emd_backend (EMDBackend, optional): The backend to use to compute the EMD. Defaults to "wasserstein".
        show_progress_bar (bool, optional): Whether to show a progress bar for the sampling progress. Defaults to True.

    Returns:
        list[float]: The list of computed EMDs.
    """
    gen = np.random.default_rng(seed) if seed is not None else None
    resample_size = resample_size or len(population) // 2

    emds: list[float] = []

    stochastic_lang = population_to_stochastic_language(population)
    behavior = [item for item, _ in stochastic_lang]
    freqs = [freq for _, freq in stochastic_lang]

    dists_start = default_timer()
    # Precompute all distances since statistically, every pair of traces will be needed at least once
    dists = compute_distance_matrix(behavior, behavior, cost_fn, show_progress_bar)
    dists_end = default_timer()

    progress_bar = create_progress_bar(
        show_progress_bar,
        total=bootstrapping_dist_size,
        desc="Bootstrapping EMD Null Distribution",
    )

    samples_1, samples_2 = (gen or np.random).choice(
        dists.shape[0],
        # Every iteration we need two samples of `resample_size` size
        (2, bootstrapping_dist_size, resample_size),
        replace=True,
        p=freqs,
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
