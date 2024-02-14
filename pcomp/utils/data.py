from typing import Iterable
from sklearn.preprocessing import KBinsDiscretizer  # type: ignore
import numpy as np
import pydantic


class Binner(pydantic.BaseModel, arbitrary_types_allowed=True):
    num_bins: int
    binner: KBinsDiscretizer

    def transform_one(self, x):
        return self.binner.transform(np.array(x).reshape(-1, 1))[0][0]

    def transform(self, x):
        return self.binner.transform(x)


def create_binner(data: Iterable, num_bins: int) -> Binner:
    data = np.array(data).reshape(-1, 1)
    est = KBinsDiscretizer(
        n_bins=num_bins, encode="ordinal", strategy="uniform", subsample=None
    )
    est.fit(data)
    return Binner(num_bins=num_bins, binner=est)
