from typing import List

import numpy as np
from pynet.functional.abstract import Function
from pynet.tensor import Tensor


class Sigmoid(Function):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: List[Tensor]) -> Tensor:
        assert len(x) == 1, "Sigmoid function takes 2 parameters"

        sig = 1.0 / (1.0 + np.exp(-x[0].ndarray))
        self._stored_results["sig"] = sig

        return Tensor(sig)

    def backward(self, y: Tensor) -> List[Tensor]:
        sig = self._stored_results["sig"]

        dsig = np.multiply(sig, 1.0 - sig)
        dsig = np.multiply(dsig, y.ndarray)

        return [Tensor(dsig)]
