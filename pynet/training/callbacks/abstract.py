from abc import ABC
from typing import List
from pynet.training.history import History

from pynet.training.stats import Statistics


class Callback(ABC):
    def __init__(self) -> None:
        super().__init__()

    def on_epoch_begin(self) -> None:
        pass

    def on_epoch_end(self, history: History) -> None:
        pass

    def on_train_begin(self) -> None:
        pass

    def on_train_end(self, history: History) -> None:
        pass

    def on_test_begin(self) -> None:
        pass

    def on_test_end(self, history: History) -> None:
        pass
