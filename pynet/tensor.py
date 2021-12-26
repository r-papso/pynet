import numpy as np


class Tensor:
    def __init__(self, ndarray: np.ndarray) -> None:
        self.ndarray = ndarray
        self.grad = None
