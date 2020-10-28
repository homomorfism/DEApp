import numpy as np

from DESolver import DESolver, DE


class GTEHelper:
    def __init__(self, n0, N, x0, y0, X):
        super().__init__()
        self.n0 = n0
        self.N = N
        self.x0 = x0
        self.y0 = y0
        self.X = X

        assert n0 < N

    def calculate_euler(self):
        gte, steps = [], []
        for i in range(self.n0, self.N + 1):
            solver = DESolver(self.x0, self.y0, i, self.X)

            x_appr, y_appr, _ = solver.euler()

            y_exact = DE().y_exact(x_appr)

            gte.append(np.max(np.abs(y_exact - y_appr)))
            steps.append(i)

        return steps, gte

    def calculate_improved(self):
        gte, steps = [], []
        for i in range(self.n0, self.N + 1):
            solver = DESolver(self.x0, self.y0, i, self.X)

            x_appr, y_appr, _ = solver.improved()

            y_exact = DE().y_exact(x_appr)

            gte.append(np.max(np.abs(y_exact - y_appr)))
            steps.append(i)

        return steps, gte

    def calculate_runge(self):
        gte, steps = [], []
        for i in range(self.n0, self.N + 1):
            solver = DESolver(self.x0, self.y0, i, self.X)

            x_appr, y_appr, _ = solver.runge_kutta()

            y_exact = DE().y_exact(x_appr)

            gte.append(np.max(np.abs(y_exact - y_appr)))
            steps.append(i)

        return steps, gte
