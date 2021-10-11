from abc import abstractmethod, ABC

import numpy as np


class BaseDE:

    @abstractmethod
    def calculate_y_deriv(self, x, y):
        pass

    @abstractmethod
    def calculate_y_exact(self, x):
        pass


class BaseDESolver(BaseDE, ABC):
    def __init__(self, x0, y0, N, x1):
        super().__init__()
        self.x = np.linspace(x0, x1, N + 1, dtype=np.float64)  # May be N
        self.N = N + 1
        self.y0 = y0
        self.h = (x1 - x0) / N
        self.y_exact = self.calculate_y_exact(self.x)


class BaseDEMethod(BaseDESolver):

    @abstractmethod
    def compute(self):
        pass


class BaseGTEHelper(BaseDE):
    def __init__(self, n0, N, x0, y0, X):
        super().__init__()
        self.n0 = n0
        self.N = N
        self.x0 = x0
        self.y0 = y0
        self.X = X

        assert n0 < N

    @abstractmethod
    def calculate_gte(self):
        pass
