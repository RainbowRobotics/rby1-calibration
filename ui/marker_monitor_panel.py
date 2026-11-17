"""Marker recognition/jitter panel, shared by the wizard exposure slide and Camera > Brightness.

Shows the core marker monitor's summary only (no detection or maths here). Each marker is remembered
as confirmed once it was stable at the current exposure, so the arms can be shown to the camera one
at a time; the confirmation is cleared whenever the exposure changes.
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QGroupBox, QVBoxLayout, QLabel, QPushButton

from core.language import tr
from core.calibration.sequences.marker_monitor import MONITOR_WINDOW

TEXT_KEY = "wizard.slides.slide_exposure."


class MarkerMonitorPanel(QGroupBox):
    SIDES = ("right", "left")
    IDLE_STYLE = "color: #9e9e9e; font-size: 14px; font-weight: bold;"
    OK_STYLE = "color: #4caf50; font-size: 14px; font-weight: bold;"
    WARN_STYLE = "color: #ff9800; font-size: 14px; font-weight: bold;"
    BAD_STYLE = "color: #f44336; font-size: 14px; font-weight: bold;"

    restart_requested = Signal()

    def __init__(self, parent=None, sides=SIDES):
        super().__init__(parent)
        # The wizard checks the right marker only (its test pose shows one arm); Camera > Brightness both.
        self.SIDES = tuple(sides)
        self.setStyleSheet("QGroupBox::title { color: #00e5ff; font-weight: bold; font-size: 15px;}")
        self.last_info = None
        self.confirmed = {side: False for side in self.SIDES}
        self.min_window = 0
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        self.labels = {}
        for side in self.SIDES:
            lbl = QLabel()
            lbl.setWordWrap(True)
            self.labels[side] = lbl
            layout.addWidget(lbl)
        self.lbl_overall = QLabel()
        self.lbl_overall.setWordWrap(True)
        layout.addWidget(self.lbl_overall)
        self.lbl_note = QLabel()
        self.lbl_note.setStyleSheet("color: #9e9e9e; font-size: 12px;")
        self.lbl_note.setWordWrap(True)
        layout.addWidget(self.lbl_note)
        # A calibration sequence, a robot move or a camera reconnect stops the monitor.
        self.btn_restart = QPushButton()
        self.btn_restart.setStyleSheet("background-color: #546e7a; color: white; font-weight: bold; font-size: 13px; border-radius: 6px; padding: 4px 12px;")
        self.btn_restart.clicked.connect(self.restart_requested)
        layout.addWidget(self.btn_restart, alignment=Qt.AlignRight)
        self.retranslate()

    @property
    def all_confirmed(self):
        return all(self.confirmed.values())

    def retranslate(self):
        self.setTitle(tr(TEXT_KEY + "monitor_title"))
        self.lbl_note.setText(tr(TEXT_KEY + "monitor_note", window=MONITOR_WINDOW))
        self.btn_restart.setText(tr(TEXT_KEY + "monitor_restart"))
        self.update_monitor(self.last_info)

    def reset_confirmation(self, window=0):
        """The exposure changed: what was confirmed at the old value no longer counts. `window` is
        the monitor window that started with the new exposure (CalibrationCore.restart_marker_monitor_window)."""
        self.confirmed = {side: False for side in self.SIDES}
        self.min_window = window
        # last_info was measured at the old exposure: showing it again would confirm the markers again.
        self.update_monitor(None)

    def update_monitor(self, info):
        """Show the latest core marker monitor summary (None/running=False = not monitoring)."""
        self.last_info = info
        running = bool(info and info.get("running"))
        sides = info.get("sides", {}) if running else {}
        limit = f"{info['max_jitter_mm']:.2f}" if running and info.get("max_jitter_mm") is not None else "-"
        # A summary already on its way when the exposure changed still describes the old exposure.
        current = running and info.get("window", self.min_window) >= self.min_window
        for side, lbl in self.labels.items():
            name = tr(TEXT_KEY + f"monitor_{side}")
            state = sides.get(side)
            if current and state is not None and state.get("stable"):
                self.confirmed[side] = True
            # Green = recognized and RMS jitter within the limit (now, or earlier at this exposure),
            # orange = recognized but jumping, red = not recognized, grey = not monitoring.
            if state is None or not state.get("visible"):
                if self.confirmed[side]:
                    lbl.setText(tr(TEXT_KEY + "monitor_confirmed", marker=name))
                    lbl.setStyleSheet(self.OK_STYLE)
                    continue
                if state is None:
                    lbl.setText(tr(TEXT_KEY + "monitor_idle", marker=name))
                    lbl.setStyleSheet(self.IDLE_STYLE)
                    continue
            jitter = state.get("jitter")
            jitter_text = (tr(TEXT_KEY + "monitor_jitter", rms=f"{jitter['rms_mm']:.2f}", max=f"{jitter['max_mm']:.2f}")
                           if jitter else "-")
            if not state.get("visible"):
                key, style = "monitor_hidden", self.BAD_STYLE
            elif state.get("stable"):
                key, style = "monitor_stable", self.OK_STYLE
            else:
                key, style = "monitor_unstable", self.WARN_STYLE
            lbl.setText(tr(TEXT_KEY + key, marker=name, limit=limit,
                           rate=f"{state.get('rate', 0.0) * 100:.0f}", jitter=jitter_text))
            lbl.setStyleSheet(style)
        self.lbl_overall.setVisible(running or self.all_confirmed)
        if self.all_confirmed:
            self.lbl_overall.setText(tr(TEXT_KEY + "monitor_all_ok") if len(self.SIDES) > 1 else
                                     tr(TEXT_KEY + "monitor_side_ok", marker=tr(TEXT_KEY + f"monitor_{self.SIDES[0]}")))
            self.lbl_overall.setStyleSheet(self.OK_STYLE)
        elif running:
            self.lbl_overall.setText(tr(TEXT_KEY + "monitor_not_ok", limit=limit))
            self.lbl_overall.setStyleSheet(self.WARN_STYLE)
        self.btn_restart.setVisible(not running)
