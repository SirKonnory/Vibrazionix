from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSize
from PySide6.QtGui import QIcon

import VIBRO_VIEW

if __name__ == "__main__":
    app = QApplication.instance()
    if not app:
        app = QApplication()
        app.setStyle("Fusion")

    run = VIBRO_VIEW.MainWindow()

    icon1 = QIcon("iconka.png")
    run.setWindowIcon(icon1)

    icon_size = QSize(200, 200)
    run.setIconSize(icon_size)
    run.resize(1400, 800)
    run.setWindowTitle('Vibrazionix')
    run.show()

    app.exec()
