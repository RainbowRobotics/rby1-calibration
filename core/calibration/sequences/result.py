from copy import deepcopy
from dataclasses import dataclass, field
from threading import Event
from time import time


def require_converged_joint(result, context, label="joint"):
    context.checkpoint(label, result)
    context.check_cancelled()
    if not isinstance(result, dict) or not result.get("converged", False):
        raise RuntimeError(f"{label}: joint calibration did not converge")
    return result


class SequenceCancelled(Exception):
    """Cooperative cancellation, carrying any completed observations/iterations."""
    def __init__(self, partial=None):
        super().__init__("Stopped by user")
        self.partial = deepcopy(partial or {})


@dataclass
class SequenceResult:
    name: str
    status: str = "running"
    completed: dict = field(default_factory=dict)
    partial: dict = field(default_factory=dict)
    error: str = ""
    started_at: float = field(default_factory=time)
    finished_at: float | None = None

    @property
    def success(self):
        return self.status == "completed"

    def finish(self, status, error=""):
        self.status, self.error, self.finished_at = status, error, time()
        return self

    def snapshot(self):
        return deepcopy(self)


class SequenceContext:
    def __init__(self, name, stop_event=None, on_progress=None):
        self.result = SequenceResult(name)
        self.stop_event = stop_event if stop_event is not None else Event()
        self.on_progress = on_progress

    def check_cancelled(self):
        if self.stop_event.is_set():
            raise SequenceCancelled(self.result.partial)

    def checkpoint(self, key, value):
        self.result.partial[key] = deepcopy(value)
        if self.on_progress:
            self.on_progress(self.result.snapshot())

    def complete(self, key, value):
        self.result.completed[key] = deepcopy(value)
        if self.on_progress:
            self.on_progress(self.result.snapshot())
