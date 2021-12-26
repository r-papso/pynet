from abc import ABC, abstractmethod
from typing import List

from pynet.tensor import Tensor


class Optimizer(ABC):
    def __init__(self) -> None:
        super().__init__()

        self.hyperparameters = dict()

    @abstractmethod
    def set_parameters(self, params: List[Tensor]) -> None:
        pass

    @abstractmethod
    def step(self) -> None:
        pass
