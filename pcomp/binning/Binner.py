import abc
from typing import Generic, TypeVar
import numpy as np

from pcomp.utils.utils import create_progress_bar

T = TypeVar("T")


class Binner(abc.ABC, Generic[T]):
    data: list[T]
    seed: int | None

    def __init__(self, data: list[T], seed: int | None = None):
        self.data = data
        self.seed = seed

        if self.seed is not None:
            np.random.seed(self.seed)

    @abc.abstractmethod
    def bin(self, data: T) -> int:
        """Bin a datapoint, returning the class index"""
        pass


class BinnerManager:
    """Manages a collection of binners, used for different classes"""

    binners: dict[
        str, Binner
    ]  # Maps "class names" (in our case activity labels) to Binners

    def __init__(
        self,
        data: list[tuple[str, float]] | dict[str, list[float]],
        binner_class: type[Binner],
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
            desc=f"Creating binners ({binner_class.__name__})",
        )
        self.binners = {
            label: binner_class(datapoints, **kwargs)
            for label, datapoints in grouped_data.items()
            if pbar.update() or True  # Update progress bar each iteration
        }
        pbar.close()

    def bin(self, label: str, data: float) -> int:
        return self.binners[label].bin(data)
