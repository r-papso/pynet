import numpy as np

from typing import List
from function import Function
from pynet.tensor import Tensor


class Multiply(Function):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: List[Tensor]) -> Tensor:
        assert len(x) == 2, "Multiply function takes 2 parameters"
        assert (
            x[0].ndarray.ndim <= 2 and x[1].ndarray.ndim <= 2
        ), "Forward -> Multiplying tensors with dimensions > 2 are not supported, yet"

        y = np.matmul(x[0].ndarray, x[1].ndarray)

        self._stored_results["a"] = x[0]
        self._stored_results["b"] = x[1]

        return Tensor(y)

    def backward(self, y: Tensor) -> List[Tensor]:
        assert (
            y.ndarray.ndim <= 2
        ), "Backward -> Multiplying tensors with dimensions > 2 are not supported, yet"

        a = self._stored_results["a"]
        b = self._stored_results["b"]

        da = np.matmul(y, b.ndarray.T)
        db = np.matmul(a.ndarray.T, y)

        return [Tensor(da), Tensor(db)]
