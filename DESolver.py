import numpy as np

#
# def y_exact(x):
#     return np.exp(x) - 1 / (x + 1)
#
#
# def y_deriv(x, y):
#     return np.exp(2 * x) + np.exp(x) + y ** 2 - 2 * y * np.exp(x)


def y_deriv(x, y):
    return (y ** 2 + x * y - x ** 2) / (x ** 2)


def y_exact(x):
    return x * (1 + (x ** 2) / 3) / (1 - (x ** 2) / 3)


class DESolver:
    def __init__(self, x0, y0, N, interval):
        self.x = np.linspace(x0, x0 + interval, N + 1)  # May be N
        self.N = N + 1
        self.y0 = y0
        self.h = interval / N
        self.y_exact = y_exact(self.x)
        # print("h:", self.h)

    def euler(self):
        y_appr = np.empty(shape=[self.N], dtype=np.float)
        lte = np.zeros(shape=[self.N], dtype=np.float)
        y_appr[0] = self.y0
        lte[0] = self.y_exact[0]
        # print('y0:', self.y0, 'y_exact[0]:', self.y_exact[0])

        for i in range(self.N - 1):
            k1 = y_deriv(self.x[i], y_appr[i])

            y_appr[i + 1] = y_appr[i] + self.h * k1

            k1 = y_deriv(self.x[i], self.y_exact[i])

            lte[i + 1] = self.y_exact[i] + self.h * k1

        lte = np.abs(lte - self.y_exact)
        # print("Euler")
        # print('lte:', lte)
        # print('x:', self.x)
        # print('y(Euler):', y_appr)

        return self.x, y_appr, lte

    def runge_kutta(self):
        y_appr = np.empty(shape=[self.N], dtype=np.float)
        lte = np.zeros(shape=[self.N], dtype=np.float)
        y_appr[0] = self.y0
        lte[0] = self.y_exact[0]
        # print('y0:', self.y0, 'y_exact[0]:', self.y_exact[0])

        for i in range(self.N - 1):
            k1 = y_deriv(self.x[i], y_appr[i])
            k2 = y_deriv(self.x[i] + self.h / 2, y_appr[i] + self.h * k1 / 2)
            k3 = y_deriv(self.x[i] + self.h / 2, y_appr[i] + self.h * k2 / 2)
            k4 = y_deriv(self.x[i] + self.h, y_appr[i] + self.h * k3)
            # print(k1)
            y_appr[i + 1] = y_appr[i] + 1 / 6 * self.h * (k1 + 2 * k2 + 2 * k3 + k4)


            k1 = y_deriv(self.x[i], self.y_exact[i])
            k2 = y_deriv(self.x[i] + self.h / 2, self.y_exact[i] + self.h * k1 / 2)
            k3 = y_deriv(self.x[i] + self.h / 2, self.y_exact[i] + self.h * k2 / 2)
            k4 = y_deriv(self.x[i] + self.h, self.y_exact[i] + self.h * k3)
            # print(k1)
            lte[i + 1] = self.y_exact[i] + 1 / 6 * self.h * (k1 + 2 * k2 + 2 * k3 + k4)

        lte = np.abs(lte - self.y_exact)
        # print("Runge-kutta")
        # print('lte:', lte)
        # print('x:', self.x)
        # print('y(Runge):', y_appr)

        return self.x, y_appr, lte

    def improved(self):
        y_appr = np.empty(shape=[self.N], dtype=np.float)
        lte = np.zeros(shape=[self.N], dtype=np.float)
        y_appr[0] = self.y0
        lte[0] = self.y_exact[0]
        # print('y0:', self.y0, 'y_exact[0]:', self.y_exact[0])

        for i in range(self.N - 1):
            k1 = y_deriv(self.x[i], y_appr[i])
            k2 = y_deriv(self.x[i] + self.h, y_appr[i] + self.h * k1)

            y_appr[i + 1] = y_appr[i] + 0.5 * self.h * (k1 + k2)

            k1 = y_deriv(self.x[i], self.y_exact[i])
            k2 = y_deriv(self.x[i] + self.h, self.y_exact[i] + self.h * k1)

            lte[i + 1] = self.y_exact[i] + 0.5 * self.h * (k1 + k2)

        lte = np.abs(lte - self.y_exact)
        # print("Improved Euler")
        # print('lte:', lte)
        # print('x:', self.x)
        # print('y(Improved):', y_appr)

        return self.x, y_appr, lte


