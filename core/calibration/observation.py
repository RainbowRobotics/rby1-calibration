"""Serialize calls to the existing marker engine; camera acquisition is independent."""
import threading
import time
from copy import deepcopy

from .sequences.result import SequenceCancelled


class _FrameReader:
    def __init__(self, camera, source):
        self.device = camera
        self.source = source
        self.last_frame_id = -1

    def __getattr__(self, name):
        return getattr(self.device, name)

    def get_color_image(self):
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            self.source.check_cancelled()
            snapshot = self.device.get_monitor_snapshot()
            if snapshot.get("error"):
                raise RuntimeError(f"Camera acquisition failed: {snapshot['error']}")
            if not snapshot["connected"]:
                raise RuntimeError("Camera disconnected during observation")
            if snapshot["frame_id"] > self.last_frame_id and snapshot["image"] is not None:
                self.last_frame_id = snapshot["frame_id"]
                self.source.last_frame = {k: v for k, v in snapshot.items() if k != "image"}
                return snapshot["image"]
            time.sleep(0.005)
        raise RuntimeError("No fresh camera frame within 2 seconds")


class ObservationSource:
    def __init__(self, engine, stop_event):
        self.engine = engine
        self.stop_event = stop_event
        self.lock = threading.RLock()
        self.last_frame = {}
        self.last_observation = None
        self.on_observation = None
        camera = engine.camera
        if camera is not None:
            try:
                camera.start_capture()
            except Exception:
                camera.stream_off()
                raise
            engine.camera = _FrameReader(camera, self)

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def check_cancelled(self):
        if self.stop_event.is_set():
            raise SequenceCancelled()

    def get_marker_transform(self, *args, **kwargs):
        with self.lock:
            self.check_cancelled()
            result = self.engine.get_marker_transform(*args, **kwargs)
            self.last_observation = deepcopy(result)
            if self.on_observation:
                self.on_observation(result, self.last_frame.copy())
            self.check_cancelled()
            return result

    def close(self):
        if self.engine.camera is not None:
            self.engine.camera.stream_off()
