import numpy as np

from .Binner import Binner


class OuterPercentileBinner(Binner):
    """Bins data into three bins:
    1. The lower `outer_percent`% of values
    2. The middle
    3. The upper `outer_percent`% of values

    The `IQR_Binner` would be an OuterPercentileBinner with outer_percent = 25
    """

    lower_boundary: float
    upper_boundary: float
    num_bins: int = 3

    def __init__(
        self, data: list[float], outer_percent: float = 10, seed: int | None = None
    ):
        """Create an OuterPercentileBinner which bins data into three bins:
          1. The lower `outer_percent`% of values
          2. The middle
          3. The upper `outer_percent`% of values

        Args:
            data (list[float]): The data to bin
            outer_percent (float, optional): The percent to bin in a separate bin at each end of the distribution. Defaults to 10.
            seed (int | None, optional): The seed used for binning. Not used in OuterPercentileBinner. Defaults to None.
        """
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
