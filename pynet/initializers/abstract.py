from abc import ABC, abstractmethod

from pynet.tensor import Tensor


class Initializer(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def initialize(self, inputs: int, outputs: int) -> Tensor:
        pass
