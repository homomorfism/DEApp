import numpy as np
from PyQt5 import QtWidgets

from DEApp.DESolver import DE, Graph, ComputeRunge, ComputeImproved, ComputeEuler
from DEApp.GTEHelper import GTEEuler, GTERunge, GTEImproved
from DEApp.QtApp import Ui_MainWindow


class ExampleApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(ExampleApp, self).__init__()
        self.setupUi(self)

        self.graph1 = Graph(title="Graph plots")
        self.graph2 = Graph(title="Local truncation error")

        self.graphError = Graph(title="Max global truncation error")

        self.set_solution_tab()
        self.set_error_tab()

        # Debug
        self.lineEdit_x0.setText('0.0')
        self.lineEdit_y0.setText('0.0')
        self.lineEdit_x1.setText('5.0')
        self.lineEdit_num_steps.setText('20')
        self.lineEdit_n0.setText('20')
        self.lineEdit_N.setText('100')

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
        x0, y0, N, x1 = self.read_input()

        n0 = int(self.lineEdit_n0.text())
        N = int(self.lineEdit_N.text())

        self.graphError.clear()
        self.graphError.set_x_range(n0, N)

        gte, steps = GTEEuler(n0, N, x0, y0, x1).calculate_gte()
        self.graphError.plot(gte, steps, name="Euler", color='b')

        gte, steps = GTEImproved(n0, N, x0, y0, x1).calculate_gte()
        self.graphError.plot(gte, steps, name="Improved", color='r')

        gte, steps = GTERunge(n0, N, x0, y0, x1).calculate_gte()
        self.graphError.plot(gte, steps, name="Runge_Kutta", color='g')

    def recalculate_button_handler(self):
        x0, y0, N, x1 = self.read_input()

        self.graph1.clear()
        self.graph2.clear()
        self.graph1.set_x_range(x0, x1)
        self.graph2.set_x_range(x0, x1)

        if self.checkBox_exact.isChecked():
            x_ex = np.linspace(x0, x1)
            y_ex = DE().calculate_y_exact(x_ex)

            self.graph1.plot(x_ex, y_ex, name="Exact", color='b')

        if self.checkBox_euler.isChecked():
            x_appr, y_appr, lte = ComputeEuler(x0, y0, N, x1).compute()

            self.graph1.plot(x_appr, y_appr, name="Euler", color='r')
            self.graph2.plot(x_appr, lte, name="Euler", color='r')

        if self.checkBox_runge.isChecked():
            x_appr, y_appr, lte = ComputeRunge(x0, y0, N, x1).compute()

            self.graph1.plot(x_appr, y_appr, name="Runge-Kutta", color='c')
            self.graph2.plot(x_appr, lte, name="Runge-Kutta", color='c')

        if self.checkBox_improved.isChecked():
            x_appr, y_appr, lte = ComputeImproved(x0, y0, N, x1).compute()

            self.graph1.plot(x_appr, y_appr, name="Improved Euler", color='g')
            self.graph2.plot(x_appr, lte, name="Improved Euler", color='g')

    def read_input(self):
        x0 = float(self.lineEdit_x0.text())
        y0 = float(self.lineEdit_y0.text())
        N = int(self.lineEdit_num_steps.text())
        x1 = float(self.lineEdit_x1.text())

        return x0, y0, N, x1
