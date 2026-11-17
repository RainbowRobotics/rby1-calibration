"""Wizard progress line: numbered circles strung on one line (2026-11-12, user).

A state per step: "done" (green tick), "now" (blue), "todo" (grey), "failed" (red), "skipped" (dim).
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QFrame


class StepProgress(QWidget):
    COLORS = {"done": "#00e676", "now": "#1e88e5", "todo": "#9e9e9e", "failed": "#ff1744", "skipped": "#616161"}

    def __init__(self, count, dot=22, font_px=14, parent=None):
        super().__init__(parent)
        self.dot, self.font_px = dot, font_px
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.dots, self.labels, self.lines = [], [], []
        for index in range(count):
            if index:
                line = QFrame()
                line.setFixedHeight(2)
                line.setMinimumWidth(16)
                layout.addWidget(line, 1)
                self.lines.append(line)
            dot_label = QLabel()
            dot_label.setFixedSize(dot, dot)
            dot_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(dot_label)
            self.dots.append(dot_label)
            name = QLabel()
            layout.addWidget(name)
            self.labels.append(name)
        self.states = ["todo"] * count

    def update_steps(self, names, states, tooltips=None):
        self.states = list(states)
        for index, (name, state) in enumerate(zip(names, states)):
            color = self.COLORS[state]
            filled = state in ("done", "now", "failed")
            self.dots[index].setText({"done": "✓", "failed": "!"}.get(state, str(index + 1)))
            self.dots[index].setStyleSheet(
                f"border: 2px solid {color}; border-radius: {self.dot // 2}px; font-weight: bold;"
                f" font-size: {self.font_px - 2}px; color: {'#ffffff' if filled else color};"
                f" background-color: {color if filled else 'transparent'};")
            self.labels[index].setText(name)
            weight = "bold" if state in ("now", "failed") else "normal"
            text_color = "#dddddd" if state == "done" else color
            self.labels[index].setStyleSheet(f"font-size: {self.font_px}px; font-weight: {weight}; color: {text_color};")
            tip = tooltips[index] if tooltips else ""
            self.dots[index].setToolTip(tip)
            self.labels[index].setToolTip(tip)
            if index:
                reached = state != "todo"
                self.lines[index - 1].setStyleSheet(f"background-color: {self.COLORS['done'] if reached else '#444444'};")
