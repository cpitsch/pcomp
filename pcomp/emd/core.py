"""
Core functionality pertaining to EMD computations for various kinds of samples of the data.
"""

import logging
from collections import Counter
from dataclasses import dataclass
from timeit import default_timer
from typing import Callable, Generic, Literal, TypeVar

import numpy as np
import ot  # type: ignore
import wasserstein  # type: ignore

from pcomp.utils import create_progress_bar, pretty_format_duration
from pcomp.utils.typing import NP_FLOAT, Numpy1DArray, NumpyMatrix

T = TypeVar("T")

# Literal Types
EMDBackend = Literal["wasserstein", "ot", "pot"]


@dataclass
class StochasticLanguage(Generic[T]):
    variants: list[T]
    frequencies: Numpy1DArray[NP_FLOAT]


def compute_emd(
    distribution_1: StochasticLanguage[T],
    distribution_2: StochasticLanguage[T],
    cost_fn: Callable[[T, T], float],
    backend: EMDBackend = "wasserstein",
    show_progress_bar: bool = True,
) -> float:
    """Compute the Earth Mover's Distance between two distributions.

    Args:
        distribution_1 (StochasticLanguage[T]): The first distribution. All distinct
            behavior with their relative frequencies.
        distribution_2 (StochasticLanguage[T]): The second distribution.
        cost_fn (Callable[[T, T], float]): A function to compute the transport cost
            between two items.
        backend (EMDBackend, optional): The backend to use for EMD computation. Defaults
            to "wasserstein" (use the "wasserstein" module). Alternatively, "ot" or
            "pot" will use the "Python Optimal Transport" package.
        show_progress_bar (bool, optional): Show a progress bar for distance
            computation. Defaults to True.

    Returns:
        float: The computed Earth Mover's Distance.
    """
    logger = logging.getLogger("@pcomp")

    dists_start = default_timer()

    dists = compute_distance_matrix(
        distribution_1.variants,
        distribution_2.variants,
        cost_fn,
        show_progress_bar=show_progress_bar,
    )

    dists_end = default_timer()

    logs_emd = emd(
        distribution_1.frequencies,
        distribution_2.frequencies,
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
    freqs_1: Numpy1DArray[NP_FLOAT],
    freqs_2: Numpy1DArray[NP_FLOAT],
    dists: NumpyMatrix[NP_FLOAT],
    backend: EMDBackend = "wasserstein",
    fall_back: bool = True,
) -> float:
    """A wrapper around the EMD computation.

    Args:
        freqs_1 (Numpy1DArray[NP_FLOAT]): 1D histogram of the first distribution. All
            positive, sums up to 1.
        freqs_2 (Numpy1DArray[NP_FLOAT]): 1D histogram of the second distribution. All
            positive, sums up to 1.
        dists (NumpyMatrix[NP_FLOAT]): The cost matrix.
        backend ("wasserstein" | "ot" | "pot"): The backend to use to compute the EMD.
            Defaults to "wasserstein" (use the wasserstein package). Alternatively,
            "ot"/"pot" refers to the Python Optimal Transport package.
        fall_back (bool, optional): If the wasserstein package is used and an error is
            thrown, fall back to the ot package. Defaults to True.
    Returns:
        float: The computed Earth Mover's Distance.
    """
    if backend == "wasserstein":
        try:
            solver = wasserstein.EMD()
            return solver(freqs_1, freqs_2, dists)
        except Exception as e:
            logging.getLogger("@pcomp").warning(
                f'Error thrown by wasserstein package: "{e}". {" Falling back to ot package." if fall_back else ""}',
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
) -> StochasticLanguage[T]:
    """Convert a population to a stochastic language object.

    Args:
        population (list[T]): The population to convert.

    Returns:
        StochasticLanguage[T]: The stochastic language.
    """
    pop_len = len(population)
    stochastic_language = [
        (item, freq / pop_len) for item, freq in Counter(population).items()
    ]

    return StochasticLanguage(
        variants=[item for item, _ in stochastic_language],
        frequencies=np.array([freq for _, freq in stochastic_language], dtype=NP_FLOAT),
    )


def compute_emd_for_index_sample(
    indices: Numpy1DArray[np.int_],
    dists: NumpyMatrix[NP_FLOAT],
    reference_frequencies: Numpy1DArray[NP_FLOAT],
    emd_backend: EMDBackend = "wasserstein",
) -> float:
    """Given a sample of indices of rows in the distance matrix, compute the EMD between
    the sample and the source population (columns).

    Args:
        indices (Numpy1DArray[np.int_]): The sampled indices (possibly containing
            duplicates)
        dists (NumpyMatrix[NP_FLOAT]): The distance matrix.
        reference_frequencies (Numpy1DArray[NP_FLOAT]): 1D Histogram of the source
            population. Used for EMD computation
        emd_backend (EMDBackend, optional): The backend to use for EMD computation.
            Defaults to "wasserstein" (Use the "wasserstein" package).

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


def compute_distance_matrix(
    population_1: list[T],
    population_2: list[T],
    cost_fn: Callable[[T, T], float],
    show_progress_bar: bool = True,
) -> NumpyMatrix[NP_FLOAT]:
    """Compute the distance matrix for two populations.

    Args:
        population_1 (list[T]): The first population of items.
        population_2 (list[T]): The second population of items.
        cost_fn (Callable[[T, T], float]): The cost function to compute the distance
            between two items.
        show_progress_bar (bool, optional): Show a progress bar? Defaults to True.

    Returns:
        NumpyMatrix[NP_FLOAT]: The distance matrix. The (i, j)-th element is the
            distance between the i-th element of population_1 and the j-th element of
            population_2.
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
