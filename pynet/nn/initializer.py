import numpy as np

from pynet.tensor import Tensor


def he_normal(inputs: int, outputs: int) -> Tensor:
    ndarray = np.random.randn(outputs, inputs) * np.sqrt(2.0 / (inputs))
    return Tensor(ndarray)
