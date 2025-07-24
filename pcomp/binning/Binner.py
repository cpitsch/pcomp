import abc
from collections.abc import Callable
from typing import Generic, TypeVar

import numpy as np

from pcomp.utils.utils import create_progress_bar

T = TypeVar("T")


class Binner(abc.ABC, Generic[T]):
    data: list[T]
    seed: int | None
    rng: np.random.Generator
    num_bins: int  # The number of bins

    def __init__(self, data: list[T], seed: int | None = None):
        self.data = data
        self.seed = seed

        self.rng = np.random.default_rng(self.seed)

    @abc.abstractmethod
    def bin(self, data: T) -> int:
        """Bin a datapoint, returning the class index"""
        pass


BinnerFactory = type[Binner] | Callable[[list[float]], Binner]
"""
Binner factory is either the class itself (i.e., calling it calls `__init__`) or a
function that, given a list of datapoints, returns a Binner Using this, any kind of binner
could be wrapped appropriately for use in a BinnerManager
"""


class BinnerManager:
    """
    Manages a collection of binners, used for different classes. Given a label and a value,
    the value is binned using the respective binner trained for that label.
    """

    # Maps "class names" (in our case: activities) to Binners
    binners: dict[str, Binner]
    # The largest number of bins of all binners
    num_bins: int

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
        """Bin a data point using the respective binner trained for the label. If the label
        wasn't seen in the training data, bin 0 is returned.

        Args:
            label (str): The class (activity) of the data point.
            data (float): The data point.

        Returns:
            int: The bin index
        """
        if label in self.binners:
            return self.binners[label].bin(data)
        else:
            # Idea: This activity does not occur in our reference log
            # So: Give it bin number 0 = Don't care about it
            # Then again, this raises the question we had before about if an activity
            # rename should take into account the time or not. However, since we train
            # the binners on both event logs, this case won't happen anyways.
            return 0
