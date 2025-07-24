import numpy as np

from .Binner import Binner


class IQR_Binner(Binner):
    """Bins values based on the interquartile range. I.e., the 25th percentile is binned to
    bin 0, the 75th percentile is binned to bin 1, and the rest is binned to bin 3"""

    Q_1: float
    Q_3: float
    num_bins: int = 3

    def __init__(self, data: list[float], seed: int | None = None):
        super().__init__(data, seed)
        self.Q_1 = np.percentile(self.data, 25).astype(float)
        self.Q_3 = np.percentile(self.data, 75).astype(float)

    def bin(self, data: float) -> int:
        if data < self.Q_1:
            # "Value is lower than usual"
            return 0
        elif data < self.Q_3:
            # "Value is a typical value"
            return 1
        else:
            # "Value is higher than usual"
            return 2
