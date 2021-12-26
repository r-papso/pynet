from abc import ABC, abstractmethod
from typing import Dict

from pynet.tensor import Tensor


class Loss(ABC):
    def __init__(self) -> None:
        super().__init__()

        self._stored_results = dict()

    @abstractmethod
    def forward(self, x: Tensor, y: Tensor) -> Dict[str, float]:
        pass

    @abstractmethod
    def backward(self) -> Tensor:
        pass
