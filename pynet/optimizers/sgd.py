import numpy as np
from typing import List

from pynet.optimizers.abstract import Optimizer
from pynet.tensor import Tensor


class SGD(Optimizer):
    def __init__(self, learning_rate: float, momentum: float) -> None:
        super().__init__()

        self.hyperparameters["lr"] = learning_rate
        self.hyperparameters["momentum"] = momentum

        self.__t = 0

    def set_parameters(self, params: List[Tensor]) -> None:
        self.__params = params
        self.__history = self.__create_history(params)

    def step(self) -> None:
        if not self.__params:
            raise ValueError("Parameters not set, call set_params function to set the parameters")

        lr = self.hyperparameters["lr"]
        momentum = self.hyperparameters["momentum"]

        for i, (param, hist) in enumerate(zip(self.__params, self.__history)):
            grad = param.grad

            if momentum > 0.0:
                if self.__t > 0:
                    hist = momentum * hist + (1.0 - momentum) * grad
                else:
                    hist = grad.copy()

                grad = hist
                self.__history[i] = hist

            weights = param.ndarray - lr * grad
            self.__params[i].ndarray = weights
            self.__params[i].grad = None
            self.__t += 1

    def __create_history(self, params: List[Tensor]) -> List[np.ndarray]:
        history = []

        for param in params:
            history.append(np.zeros_like(param.ndarray))

        return history
