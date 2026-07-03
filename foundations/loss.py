import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        y_pred += 1e-7
        n = len(y_true)

        sum = 0
        for i in range(n):
            sum += y_true[i] * np.log(y_pred[i]) + (1-y_true[i]) * np.log(1 - y_pred[i])
        return round((-1/n) * sum,4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        n = len(y_true)
        c = len(y_true[0])
        
        sum = 0
        for i in range(n):
            for j in range(c):
                sum += y_true[i][j]*np.log(y_pred[i][j])
        return round((-1/n)*sum, 4)

        
