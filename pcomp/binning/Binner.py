import abc
from collections.abc import Callable
from typing import Generic, TypeVar

import numpy as np

from pcomp.utils.utils import create_progress_bar

T = TypeVar("T")


class Binner(abc.ABC, Generic[T]):
    data: list[T]
    seed: int | None
    rng: np.random.Generator | None
    num_bins: int  # The number of bins

    def __init__(self, data: list[T], seed: int | None = None):
        self.data = data
        self.seed = seed

        if self.seed is not None:
            self.rng = np.random.default_rng(self.seed)
        else:
            self.rng = None

    @abc.abstractmethod
    def bin(self, data: T) -> int:
        """Bin a datapoint, returning the class index"""
        pass


# Binner factory is either the class itself or a function that,
# given a list of datapoints, returns a Binner
# Using this, any kind of binner could be wrapped appropriately
# for use in a BinnerManager
BinnerFactory = type[Binner] | Callable[[list[float]], Binner]


class BinnerManager:
    """Manages a collection of binners, used for different classes"""

    binners: dict[
        str, Binner
    ]  # Maps "class names" (in our case activity labels) to Binners
    num_bins: int  # The largest number of bins of all binners

    def __init__(
        self,
        data: list[tuple[str, float]] | dict[str, list[float]],
        binner_factory: BinnerFactory,
        show_training_progress_bar: bool = True,
        **kwargs,  # Passed on to binner_factory
    ):
        grouped_data: dict[str, list[float]] = dict()
        if isinstance(data, dict):
            grouped_data = data
        else:
            for label, datapoint in data:
                if label not in grouped_data:
                    grouped_data[label] = []
                grouped_data[label].append(datapoint)
        pbar = create_progress_bar(
            show_training_progress_bar,
            total=len(grouped_data),
            desc=(
                "Creating binners" + f" ({binner_factory.__name__})"
                if hasattr(binner_factory, "__name__")
                else ""
            ),
        )
        self.binners = {
            label: binner_factory(datapoints, **kwargs)
            for label, datapoints in grouped_data.items()
            if pbar.update() or True  # Update progress bar each iteration
        }
        pbar.close()

        self.num_bins = max(binner.num_bins for binner in self.binners.values())

    def bin(self, label: str, data: float) -> int:
        return self.binners[label].bin(data)
