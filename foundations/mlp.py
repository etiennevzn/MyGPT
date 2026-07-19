import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        hi = x
        for i in range(len(weights)):
            hi = hi @ weights[i] + biases[i]
            if(i != len(weights) - 1):
                hi = np.maximum(0, hi)

        return np.round(hi, 5)
