import sys

from PyQt5 import QtWidgets

from DEApp.desktop_application import ExampleApp


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = ExampleApp()
    window.show()
    app.exec_()


if __name__ == '__main__':
    print("Let's get it started")
    main()
