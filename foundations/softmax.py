import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        newZ = z - np.max(z)
        expZ = np.exp(newZ)
        sumZ = np.sum(expZ)
        expZ /= sumZ

        return np.round(expZ,4)

