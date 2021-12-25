from abc import ABC, abstractmethod


class Dataset(ABC):
    def __init__(self) -> None:
        super().__init__()

        self.__idx = -1

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        self.__idx += 1

        if self.__idx < len(self):
            return self.__getitem__(self.__idx)

        self.reset()
        raise StopIteration

    def reset(self):
        self.__idx = -1
