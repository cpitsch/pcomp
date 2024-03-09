from typing import TypeVar

import numpy as np

# Numpy types
T_np = TypeVar("T_np", bound=np.generic, covariant=True)
Numpy1DArray = np.ndarray[tuple[int], np.dtype[T_np]]
NumpyMatrix = np.ndarray[tuple[int, int], np.dtype[T_np]]
