from typing import Callable

from pynet.optimizers.abstract import Optimizer
from pynet.training.callbacks.abstract import Callback
from pynet.training.history import History


class LrSchedule(Callback):
    def __init__(self, optimizer: Optimizer, schedule: Callable[[int, float], float]) -> None:
        super().__init__()

        self.__optimizer = optimizer
        self.__schedule = schedule

    def on_epoch_end(self, history: History) -> None:
        epoch = history.rows[-1].epoch
        lr = self.__optimizer.hyperparameters["lr"]
        new_lr = self.__schedule(epoch, lr)
        self.__optimizer.hyperparameters["lr"] = new_lr
