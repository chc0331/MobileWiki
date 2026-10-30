"""Persist elapsed compute even when a local experiment fails or is interrupted."""

from __future__ import annotations

import json
import signal
import time
from pathlib import Path


class WallBudget:
    def __init__(self, path: Path, hours: float, prior_seconds: float = 0):
        self.path = path
        self.hours = hours
        self.prior_seconds = prior_seconds
        self.previous = json.loads(path.read_text())["elapsed_seconds"] if path.exists() else 0.0

    @property
    def elapsed_seconds(self) -> float:
        return self.previous + time.monotonic() - self.started

    def __enter__(self):
        remaining = self.hours * 3600 - self.prior_seconds - self.previous
        if remaining <= 0:
            raise TimeoutError("The combined experiment compute budget is exhausted")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.previous_handler = signal.getsignal(signal.SIGALRM)
        def expired(signum, frame):
            raise TimeoutError("The combined experiment compute budget is exhausted")
        signal.signal(signal.SIGALRM, expired)
        self.previous_timer = signal.setitimer(signal.ITIMER_REAL, remaining)
        return self

    def __exit__(self, error_type, error, traceback):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self.previous_handler)
        if self.previous_timer[0]:
            signal.setitimer(signal.ITIMER_REAL, *self.previous_timer)
        self.path.write_text(json.dumps({"elapsed_seconds": self.elapsed_seconds,
                                        "status": "completed" if error_type is None else "interrupted_or_failed"}, indent=2) + "\n")
        return False
