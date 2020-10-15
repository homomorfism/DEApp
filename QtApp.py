# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'QtApp.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(884, 659)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(-10, -10, 851, 611))
        self.tabWidget.setObjectName("tabWidget")
        self.SolutionsTab = QtWidgets.QWidget()
        self.SolutionsTab.setMaximumSize(QtCore.QSize(803, 16777215))
        self.SolutionsTab.setObjectName("SolutionsTab")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.SolutionsTab)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 10, 831, 551))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.inputLayout = QtWidgets.QVBoxLayout()
        self.inputLayout.setObjectName("inputLayout")
        self.label_x0 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.label_x0.setObjectName("label_x0")
        self.inputLayout.addWidget(self.label_x0)
        self.lineEdit_x0 = QtWidgets.QLineEdit(self.horizontalLayoutWidget)
        self.lineEdit_x0.setText("")
        self.lineEdit_x0.setObjectName("lineEdit_x0")
        self.inputLayout.addWidget(self.lineEdit_x0)
        self.label_y0 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.label_y0.setObjectName("label_y0")
        self.inputLayout.addWidget(self.label_y0)
        self.lineEdit_y0 = QtWidgets.QLineEdit(self.horizontalLayoutWidget)
        self.lineEdit_y0.setEnabled(True)
        self.lineEdit_y0.setObjectName("lineEdit_y0")
        self.inputLayout.addWidget(self.lineEdit_y0)
        self.label_interval = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.label_interval.setObjectName("label_interval")
        self.inputLayout.addWidget(self.label_interval)
        self.lineEdit_interval = QtWidgets.QLineEdit(self.horizontalLayoutWidget)
        self.lineEdit_interval.setObjectName("lineEdit_interval")
        self.inputLayout.addWidget(self.lineEdit_interval)
        self.label = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.label.setObjectName("label")
        self.inputLayout.addWidget(self.label)
        self.lineEdit_num_steps = QtWidgets.QLineEdit(self.horizontalLayoutWidget)
        self.lineEdit_num_steps.setObjectName("lineEdit_num_steps")
        self.inputLayout.addWidget(self.lineEdit_num_steps)
        self.checkBox_exact = QtWidgets.QCheckBox(self.horizontalLayoutWidget)
        self.checkBox_exact.setObjectName("checkBox_exact")
        self.inputLayout.addWidget(self.checkBox_exact)
        self.checkBox_euler = QtWidgets.QCheckBox(self.horizontalLayoutWidget)
        self.checkBox_euler.setObjectName("checkBox_euler")
        self.inputLayout.addWidget(self.checkBox_euler)
        self.checkBox_improved = QtWidgets.QCheckBox(self.horizontalLayoutWidget)
        self.checkBox_improved.setObjectName("checkBox_improved")
        self.inputLayout.addWidget(self.checkBox_improved)
        self.checkBox_runge = QtWidgets.QCheckBox(self.horizontalLayoutWidget)
        self.checkBox_runge.setObjectName("checkBox_runge")
        self.inputLayout.addWidget(self.checkBox_runge)
        self.pushButton = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton.setObjectName("pushButton")
        self.inputLayout.addWidget(self.pushButton)
        self.horizontalLayout_2.addLayout(self.inputLayout)
        self.graphsLayout = QtWidgets.QVBoxLayout()
        self.graphsLayout.setObjectName("graphsLayout")
        self.horizontalLayout_2.addLayout(self.graphsLayout)
        self.tabWidget.addTab(self.SolutionsTab, "")
        self.ErrorTab = QtWidgets.QWidget()
        self.ErrorTab.setObjectName("ErrorTab")
        self.tabWidget.addTab(self.ErrorTab, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 884, 30))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_x0.setText(_translate("MainWindow", "x0"))
        self.label_y0.setText(_translate("MainWindow", "y0"))
        self.label_interval.setText(_translate("MainWindow", "Interval"))
        self.label.setText(_translate("MainWindow", "Number of steps"))
        self.checkBox_exact.setText(_translate("MainWindow", "Exact Solution"))
        self.checkBox_euler.setText(_translate("MainWindow", "Euler"))
        self.checkBox_improved.setText(_translate("MainWindow", "Improved Euler"))
        self.checkBox_runge.setText(_translate("MainWindow", "Runge-Kutta"))
        self.pushButton.setText(_translate("MainWindow", "Recalculate"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.SolutionsTab), _translate("MainWindow", "Tab 1"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.ErrorTab), _translate("MainWindow", "Tab 2"))
