import math


class Statistics:
    def __init__(self, name: str) -> None:
        self.name = name

        self.__n = 0
        self.__min = float("inf")
        self.__max = float("-inf")
        self.__sum = 0
        self.__old_mean = 0
        self.__new_mean = 0
        self.__old_square_sum = 0
        self.__new_square_sum = 0

    def n(self) -> int:
        return self.__n

    def mean(self) -> float:
        return self.__new_mean if self.__n > 0 else 0.0

    def min(self) -> float:
        return self.__min

    def max(self) -> float:
        return self.__max

    def variance(self) -> float:
        return self.__new_square_sum / (self.__n - 1.0) if self.__n > 1 else 0.0

    def std(self) -> float:
        return math.sqrt(self.variance())

    def sum(self):
        return self.__sum

    def add(self, x: float) -> None:
        self.__n += 1

        if x < self.__min:
            self.__min = x

        if x > self.__max:
            self.__max = x

        self.__sum += x

        if self.__n == 1:
            self.__old_mean = self.__new_mean = x
            self.__old_square_sum = 0.0
        else:
            self.__new_mean = self.__old_mean + (x - self.__old_mean) / self.__n
            self.__new_square_sum = self.__old_square_sum + (x - self.__old_mean) * (
                x - self.__new_mean
            )

            self.__old_mean = self.__new_mean
            self.__old_square_sum = self.__new_square_sum

    def reset(self) -> None:
        self.__n = 0
        self.__min = float("inf")
        self.__max = float("-inf")
        self.__sum = 0.0
