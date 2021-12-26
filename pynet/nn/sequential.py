from typing import List

from pynet.nn.abstract import Module
from pynet.tensor import Tensor


class Sequential(Module):
    def __init__(self, modules: List[Module]) -> None:
        super().__init__()

        self.__modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.__modules:
            x = module.forward(x)

        return x

    def backward(self, y: Tensor) -> Tensor:
        for module in reversed(self.__modules):
            y = module.backward(y)

        return y

    def get(self, idx: int) -> Module:
        return self.__modules[idx]

    def get_parameters(self) -> List[Tensor]:
        params = []

        for module in self.__modules:
            mparams = module.get_parameters()
            if mparams:
                params.extend(mparams)

        return params
