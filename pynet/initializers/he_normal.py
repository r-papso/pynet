import numpy as np

from pynet.initializers.abstract import Initializer
from pynet.tensor import Tensor


class HeNormal(Initializer):
    def __init__(self) -> None:
        super().__init__()

    def initialize(self, inputs: int, outputs: int) -> Tensor:
        ndarray = np.random.randn(outputs, inputs) * np.sqrt(2.0 / (inputs))
        return Tensor(ndarray)
