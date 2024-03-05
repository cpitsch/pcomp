import numpy as np

from .Binner import Binner


class IQR_Binner(Binner):
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
