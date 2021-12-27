import numpy as np
from typing import List

from pynet.initializers.abstract import Initializer
from pynet.nn.abstract import Module
from pynet.functional.add import Add
from pynet.functional.matmul import Matmul
from pynet.tensor import Tensor


class Linear(Module):
    def __init__(self, inputs: int, neurons: int, initializer: Initializer) -> None:
        super().__init__()

        self.weights = initializer.initialize(inputs, neurons)
        self.bias = Tensor(np.zeros((neurons, 1)))

        self.__add = Add()
        self.__matmul = Matmul()

    def forward(self, x: Tensor) -> Tensor:
        y = self.__matmul.forward([self.weights, x])
        y = self.__add.forward([y, self.bias])
        return y

    def backward(self, y: Tensor) -> Tensor:
        dy, dbias = self.__add.backward(y)
        dw, dx = self.__matmul.backward(dy)

        self.weights.grad = dw.ndarray
        self.bias.grad = dbias.ndarray

        return dx

    def get_parameters(self) -> List[Tensor]:
        return [self.weights, self.bias]
