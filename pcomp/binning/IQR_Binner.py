import numpy as np

from .Binner import Binner


class IQR_Binner(Binner):
    Q_1: float
    Q_3: float

    # TODO: Dont need a seed here right
    def __init__(self, data: list[float], seed: int | None = None):
        super().__init__(data, seed)
        self.Q_1 = np.percentile(self.data, 25)
        self.Q_3 = np.percentile(self.data, 75)

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


# IQR_Binner would be OuterPercentileBiner(data, 25)
class OuterPercentileBinner(Binner):
    lower_boundary: float
    upper_boundary: float

    def __init__(self, data: list[float], outer_percent: float = 10):
        super().__init__(data)
        self.lower_boundary = np.percentile(self.data, outer_percent)
        self.upper_boundary = np.percentile(self.data, 100 - outer_percent)

    def bin(self, data: float) -> int:
        if data < self.lower_boundary:
            # Value is lower than usual
            return 0
        elif data < self.upper_boundary:
            # Value is a typical value
            return 1
        else:
            # Value is higher than usual
            return 2


# IQR_Binner could be implemented as follows:
# class IQRBinner2(Binner):
#     inner_binner: OuterPercentileBinner

#     def __init__(self, data: list[float]):
#         super().__init__(data)
#         self.inner_binner = OuterPercentileBinner(data, 25)

#     def bin(self, data: float) -> int:
#         return self.inner_binner.bin(data)
