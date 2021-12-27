import numpy as np
from typing import Dict, List

from pynet.nn.abstract import Module
from pynet.data.abstract import Dataset
from pynet.loss.abstract import Loss
from pynet.optimizers.abstract import Optimizer
from pynet.tensor import Tensor
from pynet.training.callbacks.abstract import Callback
from pynet.training.history import History
from pynet.training.stats import Statistics


class Trainer:
    """Class providing basic training/testing procedure of the neural network."""

    def __init__(self) -> None:
        """Ctor."""
        super().__init__()

    def train(
        self,
        model: Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        loss_f: Loss,
        optimizer: Optimizer,
        epochs: int,
        callbacks: List[Callback] = [],
    ) -> History:
        """Function performing training procedure of the neural network.

        Args:
            model (Module): Neural network to be trained.
            train_dataset (Dataset): Dataset that will be used for network's training.
            val_dataset (Dataset): Dataset that will be used for network's validation (can be None).
            loss_f (Loss): Neural network's loss function.
            optimizer (Optimizer): Optimizer that will be used for network's optimization.
            epochs (int): Number of training epochs.
            callbacks (List[Callback], optional): List of callbacks called during the training. Defaults to [].

        Returns:
            History: Neural network's training history.
        """
        history = History()
        optimizer.register_parameters(model.get_parameters())

        for c in callbacks:
            c.on_train_begin()

        for epoch in range(epochs):
            for c in callbacks:
                c.on_epoch_begin()

            stats = dict()

            for xi, yi in train_dataset:
                self.__run_batch(model, xi, yi, loss_f, stats, "train_")

                dl = loss_f.backward()
                _ = model.backward(dl)
                optimizer.step()

            if val_dataset:
                for xi, yi in val_dataset:
                    self.__run_batch(model, xi, yi, loss_f, stats, "val_")

            history.add(epoch, list(stats.values()))

            for c in callbacks:
                c.on_epoch_end(history)

        for c in callbacks:
            c.on_train_end(history)

        return history

    def test(
        self, model: Module, test_dataset: Dataset, loss_f: Loss, callbacks: List[Callback] = []
    ) -> History:
        """Function performing testing procedure of the neural network.

        Args:
            model (Module): Neural network to be tested.
            test_dataset (Dataset): Dataset that will be used for network's testing.
            loss_f (Loss): Neural network's loss function.
            callbacks (List[Callback], optional): List of callbacks called during the testing. Defaults to [].

        Returns:
            History: Neural network's testing history.
        """
        for c in callbacks:
            c.on_test_begin()

        history = History()
        stats = dict()

        for xi, yi in test_dataset:
            self.__run_batch(model, xi, yi, loss_f, stats, "test_")

        history.add(0, list(stats.values()))

        for c in callbacks:
            c.on_test_end(history)

        return history

    def __run_batch(
        self,
        model: Module,
        xi: np.ndarray,
        yi: np.ndarray,
        loss_f: Loss,
        stats: Dict[str, Statistics],
        key_prefix: str,
    ):
        z = model.forward(Tensor(xi))
        batch_stats = loss_f.forward(z, Tensor(yi))

        for k, v in batch_stats.items():
            kp = f"{key_prefix}{k}"
            if kp not in stats:
                stats[kp] = Statistics(kp)
            stats[kp].add(v)
