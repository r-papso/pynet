from abc import ABC, abstractmethod
from typing import List

from pynet.tensor import Tensor


class Function(ABC):
    def __init__(self) -> None:
        super().__init__()

        self._stored_results = dict()

    @abstractmethod
    def forward(self, x: List[Tensor]) -> Tensor:
        pass

    @abstractmethod
    def backward(self, y: Tensor) -> List[Tensor]:
        pass
