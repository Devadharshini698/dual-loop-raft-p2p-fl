"""Wall-clock scheduler implementing the clock interface used by RaftNode."""

from __future__ import annotations

import heapq
import math
import threading
import time
from collections.abc import Callable


class MonotonicClock:
    """Serialize scheduled callbacks on one monotonic scheduler thread."""

    def __init__(self, on_error: Callable[[Exception], None] | None = None) -> None:
        self._origin_ns = time.monotonic_ns()
        self._queue: list[tuple[float, int, Callable[[], None]]] = []
        self._sequence = 0
        self._condition = threading.Condition()
        self._closed = False
        self._on_error = on_error
        self._thread = threading.Thread(target=self._run, name="raft-clock", daemon=True)
        self._thread.start()

    @property
    def now_ms(self) -> float:
        return (time.monotonic_ns() - self._origin_ns) / 1_000_000.0

    def call_at(self, timestamp_ms: float, callback: Callable[[], None]) -> None:
        if not math.isfinite(timestamp_ms) or not callable(callback):
            raise ValueError("timestamp must be finite and callback must be callable")
        with self._condition:
            if self._closed:
                return
            self._sequence += 1
            heapq.heappush(self._queue, (float(timestamp_ms), self._sequence, callback))
            self._condition.notify()

    def call_later(self, delay_ms: float, callback: Callable[[], None]) -> None:
        if not math.isfinite(delay_ms) or delay_ms < 0:
            raise ValueError("delay must be non-negative")
        self.call_at(self.now_ms + delay_ms, callback)

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while True:
            callback = self._next_callback()
            if callback is None:
                return
            try:
                callback()
            except Exception as error:  # scheduler must remain alive for evidence collection
                if self._on_error is not None:
                    try:
                        self._on_error(error)
                    except Exception:
                        pass

    def _next_callback(self) -> Callable[[], None] | None:
        with self._condition:
            while not self._closed:
                if not self._queue:
                    self._condition.wait()
                    continue
                delay = (self._queue[0][0] - self.now_ms) / 1_000.0
                if delay > 0:
                    self._condition.wait(delay)
                    continue
                return heapq.heappop(self._queue)[2]
            return None
