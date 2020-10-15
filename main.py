from PyQt5 import QtWidgets
from QtApp import Ui_MainWindow
import sys
import pyqtgraph as pg
import numpy as np
from DESolver import DESolver, y_exact


class ExampleApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(ExampleApp, self).__init__()
        self.setupUi(self)

        self.set_solution_tab()
        self.set_error_tab()

    def set_solution_tab(self):
        self.graph1 = pg.PlotWidget()
        self.graph2 = pg.PlotWidget()
        self.graph1.setBackground('w')
        self.graph2.setBackground('w')
        self.graph1.setTitle("Graph plots")
        self.graph2.setTitle("Local truncation error")
        self.graphsLayout.addWidget(self.graph1)
        self.graphsLayout.addWidget(self.graph2)

        self.tabWidget.setTabText(0, "Solutions")

        self.pushButton.clicked.connect(self.recalculate_button_handler)

    def set_error_tab(self):
        self.tabWidget.setTabText(1, "Error")

    def recalculate_button_handler(self):
        x0 = float(self.lineEdit_x0.text())
        y0 = float(self.lineEdit_y0.text())
        N = int(self.lineEdit_num_steps.text())
        interval = float(self.lineEdit_interval.text())

        solver = DESolver(x0, y0, N, interval)

        self.graph1.clear()
        self.graph2.clear()
        self.graph1.setXRange(x0, x0 + interval)
        self.graph2.setXRange(x0, x0 + interval)
        self.graph1.addLegend()
        self.graph2.addLegend()

        if self.checkBox_exact.isChecked():
            x_ex = np.linspace(x0, x0 + interval)
            y_ex = y_exact(x_ex)

            self.graph1.plot(x_ex, y_ex, "Exact solution", 'r')

        if self.checkBox_euler.isChecked():
            x_appr, y_appr, lte = solver.euler()

            self.graph1.plot(x_appr, y_appr, "Euler", 'b')
            self.graph2.plot(x_appr, lte, "Euler", 'b')

        if self.checkBox_runge.isChecked():
            x_appr, y_appr, lte = solver.runge_kutta()

            self.graph1.plot(x_appr, y_appr, "Runge-Kutta", 'c')
            self.graph2.plot(x_appr, lte, "Runge-Kutta", 'c')

        if self.checkBox_improved.isChecked():
            x_appr, y_appr, lte = solver.improved()

            self.graph1.plot(x_appr, y_appr, "Improved Euler", 'g')
            self.graph2.plot(x_appr, lte, "Improved Euler", 'g')


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = ExampleApp()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
