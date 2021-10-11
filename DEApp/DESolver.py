import numpy as np
import pyqtgraph as pg

from DEApp.BaseClass import BaseDE, BaseDESolver, BaseDEMethod


class DE(BaseDE):
    def __init__(self):
        pass

    def calculate_y_deriv(self, x, y):
        return np.exp(2 * x) + np.exp(x) + y ** 2 - 2 * y * np.exp(x)

    def calculate_y_exact(self, x):
        return np.exp(x) - 1 / (x + 1)


class DESolver(BaseDESolver, DE):
    def __init__(self, x0, y0, N, x1):
        super().__init__(x0, y0, N, x1)


class ComputeEuler(BaseDEMethod, DESolver):
    def __init__(self, x0, y0, N, x1):
        super().__init__(x0, y0, N, x1)

    def compute(self):
        y_appr = np.empty(shape=[self.N], dtype=np.float64)
        lte = np.zeros(shape=[self.N], dtype=np.float64)
        y_appr[0] = self.y0
        lte[0] = self.y_exact[0]

        for i in range(self.N - 1):
            k1 = self.calculate_y_deriv(self.x[i], y_appr[i])

            y_appr[i + 1] = y_appr[i] + self.h * k1

            k1 = self.calculate_y_deriv(self.x[i], self.y_exact[i])

            lte[i + 1] = self.y_exact[i] + self.h * k1

        lte = np.abs(lte - self.y_exact)

        return self.x, y_appr, lte


class ComputeImproved(BaseDEMethod, DESolver):
    def __init__(self, x0, y0, N, x1):
        super().__init__(x0, y0, N, x1)

    def compute(self):
        y_appr = np.empty(shape=[self.N], dtype=np.float64)
        lte = np.zeros(shape=[self.N], dtype=np.float64)
        y_appr[0] = self.y0
        lte[0] = self.y_exact[0]

        for i in range(self.N - 1):
            k1 = self.calculate_y_deriv(self.x[i], y_appr[i])
            k2 = self.calculate_y_deriv(self.x[i] + self.h, y_appr[i] + self.h * k1)

            y_appr[i + 1] = y_appr[i] + 0.5 * self.h * (k1 + k2)

            k1 = self.calculate_y_deriv(self.x[i], self.y_exact[i])
            k2 = self.calculate_y_deriv(self.x[i] + self.h, self.y_exact[i] + self.h * k1)

            lte[i + 1] = self.y_exact[i] + 0.5 * self.h * (k1 + k2)

        lte = np.abs(lte - self.y_exact)
        return self.x, y_appr, lte


class ComputeRunge(BaseDEMethod, DESolver):
    def __init__(self, x0, y0, N, x1):
        super().__init__(x0, y0, N, x1)

    def compute(self):
        y_appr = np.empty(shape=[self.N], dtype=np.float64)
        lte = np.zeros(shape=[self.N], dtype=np.float64)
        y_appr[0] = self.y0
        lte[0] = self.y_exact[0]

        for i in range(self.N - 1):
            k1 = self.calculate_y_deriv(self.x[i], y_appr[i])
            k2 = self.calculate_y_deriv(self.x[i] + self.h / 2, y_appr[i] + self.h * k1 / 2)
            k3 = self.calculate_y_deriv(self.x[i] + self.h / 2, y_appr[i] + self.h * k2 / 2)
            k4 = self.calculate_y_deriv(self.x[i] + self.h, y_appr[i] + self.h * k3)
            y_appr[i + 1] = y_appr[i] + 1 / 6 * self.h * (k1 + 2 * k2 + 2 * k3 + k4)

            k1 = self.calculate_y_deriv(self.x[i], self.y_exact[i])
            k2 = self.calculate_y_deriv(self.x[i] + self.h / 2, self.y_exact[i] + self.h * k1 / 2)
            k3 = self.calculate_y_deriv(self.x[i] + self.h / 2, self.y_exact[i] + self.h * k2 / 2)
            k4 = self.calculate_y_deriv(self.x[i] + self.h, self.y_exact[i] + self.h * k3)
            lte[i + 1] = self.y_exact[i] + 1 / 6 * self.h * (k1 + 2 * k2 + 2 * k3 + k4)

        lte = np.abs(lte - self.y_exact)

        return self.x, y_appr, lte


class Graph:
    def __init__(self, title):
        self.graph = pg.PlotWidget()
        self.graph.setBackground('w')
        self.graph.setTitle(title)
        self.graph.addLegend()

    def set_x_range(self, start, end):
        self.graph.setXRange(start, end)

    def plot(self, x, y, name, color):
        self.graph.plot(x, y, name=name, pen=pg.mkPen(color, width=5))

    def clear(self):
        self.graph.clear()
