import numpy as np
from pynet.data.abstract import Dataset


class InMemoryDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        super().__init__()

        assert len(x) == len(y), "x and y must have same length"

        self.__x = x
        self.__y = y

    def __len__(self):
        return len(self.__x)

    def __getitem__(self, idx):
        xi = self.__x[idx]
        yi = self.__y[idx]

        return xi, yi

    def reset(self):
        super().reset()

        p = np.random.permutation(len(self.__x))
        self.__x, self.__y = self.__x[p], self.__y[p]
