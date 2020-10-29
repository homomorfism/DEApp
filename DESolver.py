import numpy as np
import pyqtgraph as pg


#
# def y_exact(x):
#     return np.exp(x) - 1 / (x + 1)
#
#
# def y_deriv(x, y):
#     return np.exp(2 * x) + np.exp(x) + y ** 2 - 2 * y * np.exp(x)

class DE:
    def __init__(self):
        pass

    def calculate_y_deriv(self, x, y):
        return np.exp(2 * x) + np.exp(x) + y ** 2 - 2 * y * np.exp(x)

    def calculate_y_exact(self, x):
        return np.exp(x) - 1 / (x + 1)


class DESolver(DE):
    def __init__(self, x0, y0, N, x1):
        super().__init__()
        self.x = np.linspace(x0, x1, N + 1, dtype=np.float64)  # May be N
        self.N = N + 1
        self.y0 = y0
        self.h = (x1 - x0) / N
        self.y_exact = self.calculate_y_exact(self.x)

    def euler(self):
        y_appr = np.empty(shape=[self.N], dtype=np.float64)
        lte = np.zeros(shape=[self.N], dtype=np.float64)
        y_appr[0] = self.y0
        lte[0] = self.y_exact[0]
        # print('y0:', self.y0, 'y_exact[0]:', self.y_exact[0])

        for i in range(self.N - 1):
            k1 = self.calculate_y_deriv(self.x[i], y_appr[i])

            y_appr[i + 1] = y_appr[i] + self.h * k1

            k1 = self.calculate_y_deriv(self.x[i], self.y_exact[i])

            lte[i + 1] = self.y_exact[i] + self.h * k1

        # print("shape lte:", lte.shape)
        # print("y_exact_shape:", self.y_exact.shape)

        lte = np.abs(lte - self.y_exact)
        # print("Euler")
        # print('lte:', lte)
        # print('x:', self.x)
        # print('y(Euler):', y_appr)

        return self.x, y_appr, lte

    def runge_kutta(self):
        y_appr = np.empty(shape=[self.N], dtype=np.float64)
        lte = np.zeros(shape=[self.N], dtype=np.float64)
        y_appr[0] = self.y0
        lte[0] = self.y_exact[0]
        # print('y0:', self.y0, 'y_exact[0]:', self.y_exact[0])

        for i in range(self.N - 1):
            k1 = self.calculate_y_deriv(self.x[i], y_appr[i])
            k2 = self.calculate_y_deriv(self.x[i] + self.h / 2, y_appr[i] + self.h * k1 / 2)
            k3 = self.calculate_y_deriv(self.x[i] + self.h / 2, y_appr[i] + self.h * k2 / 2)
            k4 = self.calculate_y_deriv(self.x[i] + self.h, y_appr[i] + self.h * k3)
            # print(k1)
            y_appr[i + 1] = y_appr[i] + 1 / 6 * self.h * (k1 + 2 * k2 + 2 * k3 + k4)

            k1 = self.calculate_y_deriv(self.x[i], self.y_exact[i])
            k2 = self.calculate_y_deriv(self.x[i] + self.h / 2, self.y_exact[i] + self.h * k1 / 2)
            k3 = self.calculate_y_deriv(self.x[i] + self.h / 2, self.y_exact[i] + self.h * k2 / 2)
            k4 = self.calculate_y_deriv(self.x[i] + self.h, self.y_exact[i] + self.h * k3)
            # print(k1)
            lte[i + 1] = self.y_exact[i] + 1 / 6 * self.h * (k1 + 2 * k2 + 2 * k3 + k4)

        lte = np.abs(lte - self.y_exact)
        # print("Runge-kutta")
        # print('lte:', lte)
        # print('x:', self.x)
        # print('y(Runge):', y_appr)

        return self.x, y_appr, lte

    def improved(self):
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
        # print("Improved Euler")
        # print('lte:', lte)
        # print('x:', self.x)
        # print('y(Improved):', y_appr)

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
