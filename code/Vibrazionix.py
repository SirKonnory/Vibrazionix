from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSize
from PySide6.QtGui import QIcon

import vibro_view

if __name__ == "__main__":
    app = QApplication.instance()
    if not app:
        app = QApplication()
        app.setStyle("Fusion")

    run = vibro_view.MainWindow()

    icon1 = QIcon("../images/iconka.png")
    run.setWindowIcon(icon1)

    icon_size = QSize(200, 200)
    run.setIconSize(icon_size)

    run.showMaximized()

    run.setWindowTitle('Vibrazionix')
    run.show()

    app.exec()
