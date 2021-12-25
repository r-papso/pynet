from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def step(self) -> None:
        pass
