from typing import List

import numpy as np
from pynet.functional.function import Function
from pynet.tensor import Tensor


class Max(Function):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: List[Tensor]) -> Tensor:
        assert len(x) == 2, "Max function takes 2 parameters"
        assert x[0].ndarray.shape == x[1].ndarray.shape, "Max -> Invalid input shapes"

        self._stored_results["mask_a"] = x[0].ndarray > x[1].ndarray
        self._stored_results["mask_b"] = x[1].ndarray > x[0].ndarray

        y = np.maximum(x[0].ndarray, x[1].ndarray)
        return Tensor(y)

    def backward(self, y: Tensor) -> List[Tensor]:
        mask_a = self._stored_results["mask_a"]
        mask_b = self._stored_results["mask_b"]

        da = np.multiply(y, mask_a)
        db = np.multiply(y, mask_b)

        return [Tensor(da), Tensor(db)]
