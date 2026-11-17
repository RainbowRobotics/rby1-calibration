"""Live marker recognition and jitter while the operator sets the camera exposure.

Runs in its own thread outside the sequence lock, so it only ever runs while no calibration
sequence does: CalibrationCore stops it before any sequence starts and whenever the camera
changes. Detection is raw (no low-pass filter) so the jitter shown is what the detector gives.
"""
import threading
from collections import deque

import numpy as np

from ..data import load_max_marker_jitter_mm, marker_monitor_summary
from .result import SequenceCancelled

# One sample per marker every MONITOR_PERIOD_S; the summary covers the last MONITOR_WINDOW
# samples per marker (about 3 s).
MONITOR_PERIOD_S = 0.1
MONITOR_WINDOW = 30
STOP_TIMEOUT_S = 3.0
SIDES = ("right", "left")


class MarkerMonitor:
    def __init__(self, core):
        self.core = core
        self._stop = threading.Event()
        self._thread = None
        self._window = 0

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def restart_window(self):
        """The exposure changed: drop the samples taken so far. Returns the new window number;
        every summary carries the number of the window its samples belong to."""
        self._window += 1
        return self._window

    def start(self):
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="marker-monitor", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=STOP_TIMEOUT_S)
            if thread.is_alive():
                raise RuntimeError("Marker monitor did not stop")

    def _sample(self, side):
        result = self.core.observer.get_marker_transform(sampling_time=0, side=side, use_filter=False)
        if isinstance(result, list) and result:
            return np.asarray(result[0], dtype=np.float64).reshape(4, 4)[:3, 3].copy()
        return None

    def _run(self):
        history = {side: deque(maxlen=MONITOR_WINDOW) for side in SIDES}
        window = self._window
        try:
            max_jitter, from_config = load_max_marker_jitter_mm()
            if not from_config:
                self.core.log_msg(f"[WARN] exposure_check.max_marker_jitter_rms_mm is missing from setting.yaml; "
                                  f"using the default {max_jitter} mm.")
            while not self._stop.is_set():
                if window != self._window:
                    window = self._window
                    for samples in history.values():
                        samples.clear()
                for side in SIDES:
                    history[side].append(self._sample(side))
                if self._stop.is_set():
                    break
                sides = marker_monitor_summary(history, max_jitter)
                self.core.emit("marker_monitor", {
                    "running": True, "window": window, "sides": sides, "max_jitter_mm": max_jitter,
                    "all_stable": all(sides[side]["stable"] for side in SIDES)})
                self._stop.wait(MONITOR_PERIOD_S)
        except SequenceCancelled:
            pass
        except Exception as error:
            self.core.log_msg(f"[WARN] Marker monitor stopped: {error}")
        finally:
            self.core.emit("marker_monitor", {"running": False})
