import math

import numpy as np
from sklearn.cluster import kmeans_plusplus  # type: ignore

from pcomp.binning.Binner import Binner
from pcomp.utils.typing import Numpy1DArray


class KMeans_Binner(Binner):
    k: int
    centroids: Numpy1DArray[np.float_]

    def __init__(self, data: list[float], k: int, seed: int | None = None):
        super().__init__(data, seed)
        self.num_bins = k

        sample_indices = np.random.choice(
            range(len(self.data)),
            size=math.ceil(0.2 * len(self.data)),
        )

        self.centroids = kmeans_plusplus(
            np.array(self.data)[sample_indices].reshape(-1, 1),
            n_clusters=self.num_bins,
            n_local_trials=10,
            random_state=self.seed,
        )[0]

    def bin(self, data: float) -> int:
        return np.argmin(np.abs(self.centroids - data)).astype(int)
