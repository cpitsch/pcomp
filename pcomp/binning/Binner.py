import abc
from typing import Generic, TypeVar
import numpy as np

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
