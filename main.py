from PyQt5 import QtWidgets
from QtApp import Ui_MainWindow
import sys
import pyqtgraph as pg
import numpy as np
from DESolver import DESolver, DE, Graph


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


class ExampleApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(ExampleApp, self).__init__()
        self.setupUi(self)

        self.graph1 = Graph(title="Graph plots")
        self.graph2 = Graph(title="Local truncation error")

        self.graphError = Graph(title="Max local truncation error")

        self.set_solution_tab()
        self.set_error_tab()

        # Debug
        self.lineEdit_x0.setText('1.0')
        self.lineEdit_y0.setText('2.0')
        self.lineEdit_interval.setText('0.5')
        self.lineEdit_num_steps.setText('5')

    def set_solution_tab(self):

        self.graphsLayout.addWidget(self.graph1.graph)
        self.graphsLayout.addWidget(self.graph2.graph)

        self.tabWidget.setTabText(0, "Solutions")

        self.pushButton.clicked.connect(self.recalculate_button_handler)

    def set_error_tab(self):
        self.graphErrorLayout.addWidget(self.graphError.graph)
        self.tabWidget.setTabText(1, "Error")

        self.pushErrorButton.clicked.connect(self.recalculate_button_error_handler)

    def recalculate_button_error_handler(self):
        x0, y0, N, interval = self.read_input()

        n0 = int(self.lineEdit_n0.text())
        N = int(self.lineEdit_N.text())

        self.graphError.clear()
        self.graphError.set_x_range(n0, N)

        gteHelper = GTEHelper(n0, N, x0, y0, interval)

        gte, steps = gteHelper.calculate_euler()
        self.graphError.plot(gte, steps, name="Euler", color='b')

        gte, steps = gteHelper.calculate_improved()
        self.graphError.plot(gte, steps, name="Improved", color='r')

        gte, steps = gteHelper.calculate_runge()
        self.graphError.plot(gte, steps, name="Runge_Kutta", color='g')

    def recalculate_button_handler(self):
        x0, y0, N, interval = self.read_input()

        solver = DESolver(x0, y0, N, interval)

        self.graph1.clear()
        self.graph2.clear()
        self.graph1.set_x_range(x0, x0 + interval)
        self.graph2.set_x_range(x0, x0 + interval)

        if self.checkBox_exact.isChecked():
            x_ex = np.linspace(x0, x0 + interval)
            y_ex = DE().y_exact(x_ex)

            self.graph1.plot(x_ex, y_ex, name="Exact", color='b')

        if self.checkBox_euler.isChecked():
            x_appr, y_appr, lte = solver.euler()

            self.graph1.plot(x_appr, y_appr, name="Euler", color='r')
            self.graph2.plot(x_appr, lte, name="Euler", color='r')

        if self.checkBox_runge.isChecked():
            x_appr, y_appr, lte = solver.runge_kutta()

            self.graph1.plot(x_appr, y_appr, name="Runge-Kutta", color='c')
            self.graph2.plot(x_appr, lte, name="Runge-Kutta", color='c')

        if self.checkBox_improved.isChecked():
            x_appr, y_appr, lte = solver.improved()

            self.graph1.plot(x_appr, y_appr, name="Improved Euler", color='g')
            self.graph2.plot(x_appr, lte, name="Improved Euler", color='g')

    def read_input(self):
        x0 = float(self.lineEdit_x0.text())
        y0 = float(self.lineEdit_y0.text())
        N = int(self.lineEdit_num_steps.text())
        interval = float(self.lineEdit_interval.text())

        return x0, y0, N, interval


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = ExampleApp()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
