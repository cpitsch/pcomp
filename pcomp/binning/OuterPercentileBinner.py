import numpy as np

from .Binner import Binner


class OuterPercentileBinner(Binner):
    lower_boundary: float
    upper_boundary: float

    def __init__(self, data: list[float], outer_percent: float = 10):
        super().__init__(data)
        self.lower_boundary = np.percentile(self.data, outer_percent).astype(float)
        self.upper_boundary = np.percentile(self.data, 100 - outer_percent).astype(
            float
        )

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
