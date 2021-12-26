from typing import List

import numpy as np
from pynet.optimizers.abstract import Optimizer
from pynet.tensor import Tensor


class SGD(Optimizer):
    def __init__(self, learning_rate: float, momentum: float) -> None:
        self.__lr = learning_rate
        self.__momentum = momentum
        self.__t = 0

    def set_parameters(self, params: List[Tensor]) -> None:
        self.__params = params
        self.__history = self.__create_history(params)

    def step(self) -> None:
        if not self.__params:
            raise ValueError("Parameters not set, call set_params function to set the parameters")

        for i, (param, hist) in enumerate(zip(self.__params, self.__history)):
            grad = param.grad

            if self.__momentum > 0.0:
                if self.__t > 0:
                    hist = self.__momentum * hist + (1.0 - self.__momentum) * grad
                else:
                    hist = grad.copy()

                grad = hist
                self.__history[i] = hist

            weights = param.ndarray - self.__lr * grad
            self.__params[i].ndarray = weights
            self.__params[i].grad = None
            self.__t += 1

    def __create_history(self, params: List[Tensor]) -> List[np.ndarray]:
        history = []

        for param in params:
            history.append(np.zeros_like(param.ndarray))

        return history
