from abc import ABC, abstractmethod

from pynet.tensor import Tensor


class Module(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def backward(self, y: Tensor) -> Tensor:
        pass
