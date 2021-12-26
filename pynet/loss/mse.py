import numpy as np
from typing import Dict

from pynet.loss.abstract import Loss
from pynet.tensor import Tensor


class MeanSquaredError(Loss):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, y: Tensor) -> Dict[str, float]:
        assert all(
            [s == 1 for s in x.ndarray.shape]
        ), "BinaryCrossEntropy -> x input must be scalar"
        assert all(
            [s == 1 for s in y.ndarray.shape]
        ), "BinaryCrossEntropy -> y input must be scalar"

        x_scalar = np.squeeze(x.ndarray).item()
        y_scalar = np.squeeze(y.ndarray).item()

        self._stored_results["x"] = x_scalar
        self._stored_results["y"] = y_scalar
        self._stored_results["x_shape"] = x.ndarray.shape

        loss = (y_scalar - x_scalar) ** 2.0
        return {"loss": loss}

    def backward(self) -> Tensor:
        x = self._stored_results["x"]
        y = self._stored_results["y"]
        shape = self._stored_results["x_shape"]

        dx = 2.0 * (x - y)
        return Tensor(np.full(shape, dx))
