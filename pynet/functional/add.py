import numpy as np

from typing import List
from function import Function
from pynet.tensor import Tensor


class Add(Function):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: List[Tensor]) -> Tensor:
        assert len(x) == 2, "Add function takes 2 parameters"
        assert x[0].ndarray.shape == x[1].ndarray.shape, "Add -> Invalid input shapes"

        return Tensor(x[0].ndarray + x[1].ndarray)

    def backward(self, y: Tensor) -> List[Tensor]:
        return [Tensor(y.ndarray.copy()), Tensor(y.ndarray.copy())]
