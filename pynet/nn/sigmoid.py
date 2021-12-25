import pynet.functional.sigmoid
from pynet.nn.abstract import Module
from pynet.tensor import Tensor


class Sigmoid(Module):
    def __init__(self) -> None:
        super().__init__()

        self.__sigmoid = pynet.functional.sigmoid.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        y = self.__sigmoid.forward([x])
        return y

    def backward(self, y: Tensor) -> Tensor:
        dx = self.__sigmoid.backward(y)[0]
        return dx
