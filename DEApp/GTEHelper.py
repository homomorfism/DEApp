import numpy as np

from DEApp.BaseClass import BaseGTEHelper
from DEApp.DESolver import DE, ComputeEuler, ComputeImproved, ComputeRunge


class GTEEuler(BaseGTEHelper, DE):
    def __init__(self, n0, N, x0, y0, X):
        super().__init__(n0, N, x0, y0, X)

    def calculate_gte(self):
        gte, steps = [], []
        for i in range(self.n0, self.N + 1):
            solver = ComputeEuler(self.x0, self.y0, i, self.X)

            x_appr, y_appr, _ = solver.compute()

            y_exact = self.calculate_y_exact(x_appr)

            gte.append(np.max(np.abs(y_exact - y_appr)))
            steps.append(i)

        return steps, gte


class GTEImproved(BaseGTEHelper, DE):
    def __init__(self, n0, N, x0, y0, X):
        super().__init__(n0, N, x0, y0, X)

    def calculate_gte(self):
        gte, steps = [], []
        for i in range(self.n0, self.N + 1):
            solver = ComputeImproved(self.x0, self.y0, i, self.X)

            x_appr, y_appr, _ = solver.compute()

            y_exact = self.calculate_y_exact(x_appr)

            gte.append(np.max(np.abs(y_exact - y_appr)))
            steps.append(i)

        return steps, gte


class GTERunge(BaseGTEHelper, DE):
    def __init__(self, n0, N, x0, y0, X):
        super().__init__(n0, N, x0, y0, X)

    def calculate_gte(self):
        gte, steps = [], []
        for i in range(self.n0, self.N + 1):
            solver = ComputeRunge(self.x0, self.y0, i, self.X)

            x_appr, y_appr, _ = solver.compute()

            y_exact = self.calculate_y_exact(x_appr)

            gte.append(np.max(np.abs(y_exact - y_appr)))
            steps.append(i)

        return steps, gte
