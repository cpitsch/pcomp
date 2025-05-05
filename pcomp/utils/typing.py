from typing import TypeVar

import numpy as np

# The numpy float type to be used (what precision)
# (Though the EMD library likely uses f64, so it wouldnt really make a difference to use f32)
NP_FLOAT = np.float64

# Numpy types
T_np = TypeVar("T_np", bound=np.generic, covariant=True)
Numpy1DArray = np.ndarray[tuple[int], np.dtype[T_np]]
NumpyMatrix = np.ndarray[tuple[int, int], np.dtype[T_np]]
