from abc import ABC, abstractmethod
from typing import List

from pynet.tensor import Tensor


class Module(ABC):
    """Abstract class representing neural network's module."""

    def __init__(self) -> None:
        """Ctor"""
        super().__init__()

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Performs forward step through the module.

        Args:
            x (Tensor): [description]

        Returns:
            Tensor: [description]
        """
        pass

    @abstractmethod
    def backward(self, y: Tensor) -> Tensor:
        pass

    @abstractmethod
    def get_parameters(self) -> List[Tensor]:
        pass
