from project_ui import Ui_MainWindow
from PyQt5.Qt import *
import sys

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow,self).__init__()
        self.setupUi(self)
        self.pushButton.text()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
    