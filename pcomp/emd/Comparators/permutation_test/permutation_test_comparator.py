import logging
from abc import ABC, abstractmethod
from timeit import default_timer
from typing import Callable, Generic, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from pathos.multiprocessing import ProcessingPool, cpu_count  # type: ignore

from pcomp.emd.core import EMDBackend, emd, population_to_stochastic_language
from pcomp.utils.typing import Numpy1DArray, NumpyMatrix
from pcomp.utils.utils import (
    create_progress_bar,
    ensure_start_timestamp_column,
    pretty_format_duration,
)

T = TypeVar("T")


class Permutation_Test_Comparator(ABC, Generic[T]):
    log_1: pd.DataFrame
    log_2: pd.DataFrame

    distribution_size: int
    verbose: bool
    cleanup_on_del: bool
    emd_backend: EMDBackend
    seed: int | None
    multiprocess_cores: int

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
        multiprocess_cores: int = 0,
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
            seed (int, optional): The seed to use for sampling in the permutation test
                phase.
            multiprocess_cores (int, optional): Use multiprocessing for distance computation?
                Defaults to 0 (no multiprocessing used).
        """
        self.log_1 = ensure_start_timestamp_column(log_1)
        self.log_2 = ensure_start_timestamp_column(log_2)
        self.distribution_size = distribution_size

        self.verbose = verbose
        self.cleanup_on_del = cleanup_on_del
        self.emd_backend = emd_backend

        self.seed = seed
        self.multiprocess_cores = multiprocess_cores

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
    def permutation_distribution(self) -> Numpy1DArray[np.float_]:
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

    @property
    def comparison_runtime(self) -> float:
        """
        The duration of the the `compare` call.
        If accessed before calling `compare`, a ValueError will be raised.
        """
        if not hasattr(self, "_comparison_runtime"):
            raise ValueError(
                "Must call `compare` before accessing `comparison_runtime`."
            )
        return self._comparison_runtime

    def compare(self) -> float:
        """Apply the full pipeline to compare the event logs.

        1. Extract the representations from the event logs using
            `self.extract_representations`.
        2. Compute the EMD between the two representations.
        3. Compute a distribution of EMDs using a Permutation test.
        4. Compute the p-value. As the fraction of permutation distribution EMDs that
           are larger than the logs EMD computed in step (2).
        Returns:
            float: The computed p-value.
        """

        start_time = default_timer()

        self.behavior_1, self.behavior_2 = self.extract_representations(
            self.log_1, self.log_2
        )

        stoch_lang_1 = population_to_stochastic_language(self.behavior_1)
        stoch_lang_2 = population_to_stochastic_language(self.behavior_2)
        combined_variants = population_to_stochastic_language(
            self.behavior_1 + self.behavior_2
        ).variants

        if self.multiprocess_cores > 0:
            large_distance_matrix = compute_symmetric_distance_matrix_mp(
                combined_variants,
                self.cost_fn,
                self.verbose,
                num_cores=self.multiprocess_cores,
            )
        else:
            large_distance_matrix = compute_symmetric_distance_matrix(
                combined_variants, self.cost_fn, self.verbose
            )

        log_1_log_2_distances = project_large_distance_matrix(
            large_distance_matrix,
            combined_variants,
            stoch_lang_1.variants,
            stoch_lang_2.variants,
        )

        self._logs_emd = emd(
            stoch_lang_1.frequencies,
            stoch_lang_2.frequencies,
            log_1_log_2_distances,
            self.emd_backend,
        )

        if self.multiprocess_cores > 0:
            permutation_test_distribution = (
                compute_permutation_test_distribution_precomputed_distances_mp(
                    self.behavior_1,
                    self.behavior_2,
                    combined_variants,
                    large_distance_matrix,
                    self.distribution_size,
                    seed=self.seed,
                    emd_backend=self.emd_backend,
                    show_progress_bar=self.verbose,
                    num_cores=self.multiprocess_cores,
                )
            )
        else:
            permutation_test_distribution = (
                compute_permutation_test_distribution_precomputed_distances(
                    self.behavior_1,
                    self.behavior_2,
                    combined_variants,
                    large_distance_matrix,
                    self.distribution_size,
                    seed=self.seed,
                    emd_backend=self.emd_backend,
                    show_progress_bar=self.verbose,
                )
            )

        # TODO: Video says > not >=, using > for now.
        # Although it will likely never really matter
        num_larger_dists = (permutation_test_distribution > self._logs_emd).sum()

        self._permutation_distribution = permutation_test_distribution
        self._pval = num_larger_dists / self.distribution_size

        self._comparison_runtime = default_timer() - start_time

        return self._pval

    def plot_result(self) -> Figure:
        """Plot the computed distribution and the EMD between the two logs.

        Returns:
            plt.figure: The corresponding figure.
        """
        fig, ax = plt.subplots()

        ax.hist(
            self._permutation_distribution,
            bins=50,
            edgecolor="black",
            alpha=0.7,
            label=r"$P$",
        )
        ax.set_xlabel("Earth Mover's Distance")
        ax.set_ylabel("Frequency")
        ax.axvline(
            self.logs_emd,
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
    """Compute the distribution for a permutation test. Computes the distance matrix
    for all behavior. If some distances are already computed before this, it is likely
    worth precomputing all distances and calling
    `compute_permutation_test_distribution_precomputed_distances` instead.

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
    combined_variants = population_to_stochastic_language(
        population_1 + population_2
    ).variants

    dists = compute_symmetric_distance_matrix(
        combined_variants, cost_fn, show_progress_bar=show_progress_bar
    )

    return compute_permutation_test_distribution_precomputed_distances(
        population_1,
        population_2,
        combined_variants,
        dists,
        distribution_size,
        seed,
        emd_backend,
    )


def compute_symmetric_distance_matrix_mp(
    population: list[T],
    cost_fn: Callable[[T, T], float],
    show_progress_bar: bool = True,
    num_cores: int | None = None,
) -> NumpyMatrix[np.float_]:
    """Compute the distance matrix from the population to itself, assuming a symmetric
    cost function. Due to this assumption, we can cut the computation workload in half,
    only needing to compute one half of the matrix explicitly.

    Args:
        population (list[T]): The population of behavior to compute the distance_matrix
            for. Assumed to hold unique values.
        cost_fn (Callable[[T, T], float]): The _symmetric_ cost function to use.
        show_progress_bar (bool, optional): Show a progress bar for the computation?
            Defaults to True.

    Returns:
        NumpyMatrix[np.float_]: The distance matrix using the indices of values in
            `population`.
    """
    dists_start = default_timer()
    dists = np.empty((len(population), len(population)), dtype=np.float_)

    def compute_distance_matrix_row(
        args: tuple[int, list[T], Callable[[T, T], float]]
    ) -> Numpy1DArray[np.float_]:
        i, population, cost_fn = args
        row = np.empty(len(population) - i, dtype=np.float_)
        for j, item in enumerate(population[i:]):
            row[j] = cost_fn(population[i], item)

        return row

    logging.getLogger("@pcomp").info(
        f"Computing Complete Distance Matrix ({dists.shape[0]}x{dists.shape[1]})"
    )

    num_cores = num_cores or cpu_count() - 4
    with ProcessingPool(num_cores) as p:
        # args = [(i, population, cost_fn) for i in range(len(population))]
        args = [(i, population, cost_fn) for i in range(len(population))]
        rows = p.map(
            compute_distance_matrix_row,
            args,
            # concatenate_numpy_output=False,
            # progress_bar=show_progress_bar,
            # progress_bar_options={
            #     "desc": f"Computing Distance Matrix ({dists.shape[0]}x{dists.shape[1]})",
            #     "unit": "dists",
            # },
        )

    for i, row in enumerate(rows):
        dists[i, i:] = row
        dists[i:, i] = row

    dists_end = default_timer()
    logging.getLogger("@pcomp").info(
        f"Computing Complete Distance Matrix ({dists.shape[0]}x{dists.shape[1]}) took {pretty_format_duration(dists_end - dists_start)}"
    )
    return dists


def compute_symmetric_distance_matrix(
    population: list[T],
    cost_fn: Callable[[T, T], float],
    show_progress_bar: bool = True,
) -> NumpyMatrix[np.float_]:
    """Compute the distance matrix from the population to itself, assuming a symmetric
    cost function. Due to this assumption, we can cut the computation workload in half,
    only needing to compute one half of the matrix explicitly.

    Args:
        population (list[T]): The population of behavior to compute the distance_matrix
            for. Assumed to hold unique values.
        cost_fn (Callable[[T, T], float]): The _symmetric_ cost function to use.
        show_progress_bar (bool, optional): Show a progress bar for the computation?
            Defaults to True.

    Returns:
        NumpyMatrix[np.float_]: The distance matrix using the indices of values in
            `population`.
    """
    dists_start = default_timer()
    dists = np.empty((len(population), len(population)), dtype=np.float_)
    with create_progress_bar(
        show_progress_bar,
        total=dists.shape[0] * dists.shape[1],
        desc=f"Computing Distance Matrix ({dists.shape[0]}x{dists.shape[1]})",
    ) as progress_bar:
        for i, item_1 in enumerate(population):
            for j, item_2 in enumerate(population[i:], start=i):
                dists[i, j] = cost_fn(item_1, item_2)
                # Assumes symmetric cost function
                dists[j, i] = dists[i, j]
                progress_bar.update(2 if i != j else 1)

    dists_end = default_timer()
    logging.getLogger("@pcomp").info(
        f"Computing Complete Distance Matrix took {pretty_format_duration(dists_end - dists_start)}"
    )
    return dists


def project_large_distance_matrix(
    dist_matrix: NumpyMatrix[np.float_],
    dist_matrix_source_population: list[T],
    population_1: list[T],
    population_2: list[T],
) -> NumpyMatrix[np.float_]:
    """Project a large distance matrix to the distance matrix induced by the two given
    populations.

    Args:
        dist_matrix (NumpyMatrix[np.float_]): The large distance matrix.
        dist_matrix_source_population (list[T]): The population used to generate the
            distance matrix. The distance matrix should have dimension
            |source_population|x|source_population. Used to find the indices in the
            distance matrix to project to.
        population_1 (list[T]): The first population for which we to the projection.
            Defines the rows to project to.
        population_2 (list[T]): The second population for which to do the projection.
            Defines the columns to project to.

    Returns:
        NumpyMatrix[np.float_]: The projected distance matrix
    """
    population_1_matrix_indices = [
        dist_matrix_source_population.index(item) for item in population_1
    ]
    population_2_matrix_indices = [
        dist_matrix_source_population.index(item) for item in population_2
    ]

    return dist_matrix[population_1_matrix_indices, :][:, population_2_matrix_indices]


def get_permutation_sample(
    sample_range: int,
    number_of_samples: int,
    seed: int | None | np.random.Generator = None,
) -> NumpyMatrix[np.float_]:
    """Get a number of permutations of numbers from 0..sample_range.

    Args:
        sample_range (int): The number of distinct possible values. Samples will contain
            numbers from 0 to sample_range - 1
        number_of_samples (int): The number of samples to create. The number of rows in
            the result matrix.
        seed (int | None | np.random.Generator, optional): The seed to use for sampling.

    Returns:
        NumpyMatrix[np.float_]: The permutation samples. Has the diimensions
            sample_range x number_of_samples. Each row contains numbers from the
            interval [0, sample_range) in a random order.
    """
    gen = np.random.default_rng(seed)

    # Create matrix with rows containing 0...sample_range - 1
    unpermuted_sample = np.tile(np.arange(sample_range), (number_of_samples, 1))
    # Permute the rows to get the sample
    return gen.permuted(unpermuted_sample, axis=1)


def compute_permutation_test_distribution_precomputed_distances(
    behavior_1: list[T],
    behavior_2: list[T],
    distance_matrix_source_population: list[T],
    distance_matrix: NumpyMatrix[np.float_],
    distribution_size: int = 10_000,
    seed: int | None = None,
    emd_backend: EMDBackend = "wasserstein",
    show_progress_bar: bool = True,
) -> Numpy1DArray[np.float_]:
    """Compute the distribution for a permutation test.


    Args:
        behavior_1 (list[T]): The behavior extracted from the first event log.
            (|Log_1| items).
        behavior_2 (list[T]): The behavior extracted from the second event log.
            (|Log_2| items).
        distance_matrix_source_population (list[T]): The population used to compute the
            distance matrix. Used to map indices in the behavior lists to indices in the
            distance matrix.
        distance_matrix (NumpyMatrix[np.float_]): The distance matrix.
        distribution_size (int, optional): The number of EMDs to compute. Defaults to
            10_000.
        seed (int | None, optional): The seed to use for sampling. Defaults to None.
        emd_backend (EMDBackend, optional): The backend to use for EMD computation.
            Defaults to "wasserstein" (use the wasserstein package). Alternatively,
            "pot" uses "Python Optimal Transport" to compute the EMD.
        show_progress_bar (bool, optional): Show a progress bar? Defaults to True.

    Returns:
        Numpy1DArray[np.float_]: The computed EMDs.
    """

    gen = np.random.default_rng(seed)

    # Map index in event logs to variant indices
    population_indices_to_variant_indices = np.array(
        [
            distance_matrix_source_population.index(item)
            for item in behavior_1 + behavior_2
        ],
        dtype=np.int_,
    )

    progress_bar = create_progress_bar(
        show_progress_bar,
        total=distribution_size,
        desc="Computing EMDs for Permutation Test",
    )

    sample_size = len(behavior_1) + len(behavior_2)

    emds_start = default_timer()
    emds = np.empty(distribution_size, dtype=np.float_)
    for idx in range(distribution_size):
        sample = gen.permutation(sample_size)

        sample_1 = sample[: len(behavior_1)]
        sample_2 = sample[len(behavior_1) :]

        # Translate indices in population to variant
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
            distance_matrix[deduplicated_sample_1][:, deduplicated_sample_2],
            backend=emd_backend,
        )
        progress_bar.update()
    progress_bar.close()

    emds_end = default_timer()

    logging.getLogger("@pcomp").info(
        f"Computing Permutation Test Distribution took {pretty_format_duration(emds_end - emds_start)}"
    )
    return emds


def compute_permutation_test_distribution_precomputed_distances_mp(
    behavior_1: list[T],
    behavior_2: list[T],
    distance_matrix_source_population: list[T],
    distance_matrix: NumpyMatrix[np.float_],
    distribution_size: int = 10_000,
    seed: int | None = None,
    emd_backend: EMDBackend = "wasserstein",
    show_progress_bar: bool = True,
    num_cores: int | None = None,
) -> Numpy1DArray[np.float_]:
    """Compute the distribution for a permutation test.


    Args:
        behavior_1 (list[T]): The behavior extracted from the first event log.
            (|Log_1| items).
        behavior_2 (list[T]): The behavior extracted from the second event log.
            (|Log_2| items).
        distance_matrix_source_population (list[T]): The population used to compute the
            distance matrix. Used to map indices in the behavior lists to indices in the
            distance matrix.
        distance_matrix (NumpyMatrix[np.float_]): The distance matrix.
        distribution_size (int, optional): The number of EMDs to compute. Defaults to
            10_000.
        seed (int | None, optional): The seed to use for sampling. Defaults to None.
        emd_backend (EMDBackend, optional): The backend to use for EMD computation.
            Defaults to "wasserstein" (use the wasserstein package). Alternatively,
            "pot" uses "Python Optimal Transport" to compute the EMD.
        show_progress_bar (bool, optional): Show a progress bar? Defaults to True.

    Returns:
        Numpy1DArray[np.float_]: The computed EMDs.
    """

    # Matrix of rows of 1...size_of_logs
    samples = get_permutation_sample(
        len(behavior_1) + len(behavior_2), distribution_size, seed
    )

    # Map index in event logs to variant indices
    population_indices_to_variant_indices = np.array(
        [
            distance_matrix_source_population.index(item)
            for item in behavior_1 + behavior_2
        ],
        dtype=np.int_,
    )

    def compute_permutation_emd(
        args: tuple[
            NumpyMatrix[np.float_], Numpy1DArray[np.int_], NumpyMatrix[np.float_], int
        ]
    ):
        samples, sample_translator, dists, separation_index = args

        out = np.empty(samples.shape[0], dtype=np.float_)

        for idx in range(samples.shape[0]):
            sample_1 = samples[idx][:separation_index]
            sample_2 = samples[idx][separation_index:]

            translated_sample_1 = sample_translator[sample_1]
            translated_sample_2 = sample_translator[sample_2]

            deduplicated_sample_1, counts_1 = np.unique(
                translated_sample_1, return_counts=True
            )
            deduplicated_sample_2, counts_2 = np.unique(
                translated_sample_2, return_counts=True
            )

            out[idx] = emd(
                counts_1 / translated_sample_1.shape[0],
                counts_2 / translated_sample_2.shape[0],
                dists[deduplicated_sample_1][:, deduplicated_sample_2],
                backend=emd_backend,
            )

        return out

    emds_start = default_timer()

    num_cores = num_cores or cpu_count() - 4
    with ProcessingPool(num_cores) as p:
        # 100 EMDs per process
        args = [
            (
                samples[i * 100 : (i * 100) + 100],
                population_indices_to_variant_indices,
                distance_matrix,
                len(behavior_1),
            )
            for i in range(samples.shape[0] // 100)
        ]
        emd_results = p.map(compute_permutation_emd, args)

    emds = np.concatenate(emd_results)
    assert emds.shape[0] == distribution_size

    emds_end = default_timer()

    logging.getLogger("@pcomp").info(
        f"Computing Permutation Test Distribution took {pretty_format_duration(emds_end - emds_start)}"
    )
    return emds
