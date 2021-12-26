from pynet.training.callbacks.abstract import Callback
from pynet.training.history import History


class PrintCallback(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_epoch_end(self, history: History) -> None:
        self.__print_last_hist_row(history)

    def on_test_end(self, history: History) -> None:
        self.__print_last_hist_row(history)

    def __print_last_hist_row(self, history: History):
        row = history.rows[-1]
        text = f"Epoch {row.epoch:04d} -> "
        text += ", ".join([f"{stat.name}: {stat.mean():.4f}" for stat in row.stats])
        print(text)