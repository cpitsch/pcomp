import numpy as np
from sklearn.cluster import kmeans_plusplus  # type: ignore

from pcomp.binning.Binner import Binner
from pcomp.utils.typing import NP_FLOAT, Numpy1DArray


class KMeans_Binner(Binner):
    """Bins values using KMeans++ binning. For this, the resulting centroids are sorted
    ascendingly, and a value is binned to the index of the closest centroid."""

    k: int
    centroids: Numpy1DArray[NP_FLOAT]

    def __init__(self, data: list[float], k: int, seed: int | None = None):
        super().__init__(data, seed)

        self.num_bins = min(k, len(data))
        # sample_indices = self.rng.choice(
        #     range(len(self.data)), size=math.ceil(0.2 * len(self.data)), replace=False
        # )

        self.centroids = kmeans_plusplus(
            # np.array(self.data)[sample_indices].reshape(-1, 1),
            np.array(self.data).reshape(-1, 1),
            n_clusters=self.num_bins,
            n_local_trials=10,
            random_state=self.seed,
        )[0]
        # Sort the centroids ascending. This means that the highest bin corresponds to
        # the highest value, etc.

        self.centroids.sort(axis=0)

    def bin(self, data: float) -> int:
        return np.argmin(np.abs(self.centroids - data)).astype(int)
