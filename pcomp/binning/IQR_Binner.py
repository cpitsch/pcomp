import numpy as np

from pcomp.utils.typing import Numpy1DArray
from pcomp.utils.utils import create_progress_bar
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


class BinnerManager:
    binners: dict[
        str, Binner
    ]  # Maps "class names" (in our case activity labels) to Binners

    def __init__(
        self,
        data: list[tuple[str, float]] | dict[str, list[float]],
        binner_factory: type[Binner],
        show_training_progress_bar: bool = True,
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
            desc="Creating binners",
        )
        self.binners = {
            label: binner_factory(datapoints)
            for label, datapoints in grouped_data.items()
            if pbar.update() or True  # Update progress bar each iteration
        }
        pbar.close()

    def bin(self, label: str, data: float) -> int:
        return self.binners[label].bin(data)
