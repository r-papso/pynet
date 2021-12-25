import numpy as np
from __future__ import annotations


class Tensor:
    def __init__(self, ndarray: np.ndarray) -> None:
        self.ndarray = ndarray
        self.grad = None
