import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        newZ = z - np.max(z)
        expZ = np.exp(newZ)
        return np.round(expZ / np.sum(expZ),4)

