"""Background thread around AgeBandPipeline, same shape as ChatBox ``_DetectionWorker``:
latest-frame handoff (stale frames dropped), non-blocking ``get_results``. Construct it only in
museum mode; when museum mode is off nothing here is imported or run."""

from __future__ import annotations

import threading
from typing import List, Optional

from .appearance import AgeBandPipeline


class AgeBandWorker(threading.Thread):
    def __init__(self, pipeline: AgeBandPipeline):
        super().__init__(daemon=True)
        self._pipe = pipeline
        self._lock = threading.Lock()
        self._frame = None
        self._results: List[dict] = []
        self._wake = threading.Event()
        self._stop_evt = threading.Event()   # not `_stop`: would shadow Thread._stop

    def submit(self, frame) -> None:
        with self._lock:
            self._frame = frame
        self._wake.set()

    def get_results(self) -> List[dict]:
        with self._lock:
            return list(self._results)

    def stop(self) -> None:
        self._stop_evt.set()
        self._wake.set()

    def run(self) -> None:
        while not self._stop_evt.is_set():
            self._wake.wait()
            self._wake.clear()
            with self._lock:
                frame, self._frame = self._frame, None
            if frame is None or self._stop_evt.is_set():
                continue
            res = self._pipe.process(frame)
            with self._lock:
                self._results = res
