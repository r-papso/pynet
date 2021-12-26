import numpy as np
from typing import List

from pynet.nn.abstract import Module
from pynet.nn.initializer import he_normal
from pynet.functional.add import Add
from pynet.functional.multiply import Multiply
from pynet.tensor import Tensor


class Linear(Module):
    def __init__(self, inputs: int, neurons: int) -> None:
        super().__init__()

        self.weights = he_normal(inputs, neurons)
        self.bias = Tensor(np.zeros((neurons, 1)))

        self.__add = Add()
        self.__multiply = Multiply()

    def forward(self, x: Tensor) -> Tensor:
        y = self.__multiply.forward([self.weights, x])
        y = self.__add.forward([y, self.bias])
        return y

    def backward(self, y: Tensor) -> Tensor:
        dy, dbias = self.__add.backward(y)
        dw, dx = self.__multiply.backward(dy)

        self.weights.grad = dw.ndarray
        self.bias.grad = dbias.ndarray

        return dx

    def get_parameters(self) -> List[Tensor]:
        return [self.weights, self.bias]
