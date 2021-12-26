from typing import List

from pynet.training.stats import Statistics


class HistoryRow:
    def __init__(self, epoch: int, stats: List[Statistics]) -> None:
        self.epoch = epoch
        self.stats = stats


class History:
    def __init__(self) -> None:
        self.rows = []

    def add(self, epoch: int, stats: List[Statistics]) -> None:
        self.rows.append(HistoryRow(epoch, stats))
