from abc import ABC, abstractmethod
from time import time

class Budget(ABC):
    @abstractmethod
    def consume(self):
        pass

    @abstractmethod
    def is_exhausted(self) -> bool:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


class StepBudget(Budget):
    def __init__(self, steps):
        self.steps = steps
        self.current_step = steps

    def consume(self, steps):
        if self.is_exhausted():
            raise RuntimeError("can't consume an already exhausted budget")
        if steps <= 0:
            raise ValueError("can only consume a positive value")

        self.current_step -= steps

    def is_exhausted(self):
        return self.current_step < 0

    def __str__(self):
        return f"StepBuget: {self.current_step}/{self.steps} remaining"


class TimeBudget(Budget):
    def __init__(self, seconds):
        self.seconds = seconds
        self.starttime = time()
        self.current_time = None

    def consume(self):
        if self.current_time is not None and self.is_exhausted():
            raise RuntimeError("can't consume an already exhausted budget")
        self.current_time = time()

    def is_exhausted(self):
        if self.current_time is None:
            raise RuntimeError("budget has to be consumed() at the time it should be checked")
        return self.starttime + self.seconds < self.current_time

    def __str__(self):
        return f"TimeBudget: {(self.starttime + self.seconds)-self.current_time}/{self.seconds} remaining"
