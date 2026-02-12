import pyqtgraph as pg
import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QFont
from gui.ui import MainWindow


def main() -> None:
    pg.setConfigOptions(antialias=True)
    app = QApplication(sys.argv)
    
    # Set global default font size to 14px
    default_font = QFont()
    default_font.setPointSize(14)
    app.setFont(default_font)
    
    # Clear any stored Qt window geometry to enforce a fresh window size
    settings = QSettings("evo", "evo_gui")
    settings.remove("geometry")  # Clear window geometry
    settings.remove("windowState")  # Clear window state
    settings.sync()  # Sync changes to disk immediately
    
    w = MainWindow()
    # Ensure window size is correct before show()
    w.show()
    # After show(), adjust size again to make sure it takes effect
    screen = QApplication.primaryScreen()
    if screen is not None:
        available_rect = screen.availableGeometry()
        max_height = max(600, int(available_rect.height() * 0.8))
        current_height = w.height()
        if current_height > max_height:
            w.resize(w.width(), max_height)
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


