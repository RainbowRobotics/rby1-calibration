"""Text plus an endless progress bar, for work in a worker whose length is not known in advance.

Shared by the Camera tab and the wizard intrinsics slide. Hidden until show() is called.
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar

from core.language import tr


class BusyIndicator(QWidget):
    def __init__(self, text_key, parent=None):
        super().__init__(parent)
        self.text_key = text_key
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(4)
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setWordWrap(True)
        self.label.setStyleSheet("color: #1e88e5; font-weight: bold; font-size: 13px;")
        layout.addWidget(self.label)
        self.bar = QProgressBar()
        # Range (0, 0) makes Qt animate the bar without a percentage.
        self.bar.setRange(0, 0)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(10)
        self.bar.setStyleSheet("QProgressBar { background-color: #2a2a2a; border: none; border-radius: 4px; }"
                               "QProgressBar::chunk { background-color: #1e88e5; border-radius: 4px; }")
        layout.addWidget(self.bar)
        self.retranslate()
        self.hide()

    def retranslate(self):
        self.label.setText(tr(self.text_key))
