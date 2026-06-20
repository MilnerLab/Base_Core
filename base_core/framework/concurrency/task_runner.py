from __future__ import annotations

import logging
import queue
import threading
from functools import wraps
from typing import Callable, Iterable, TypeVar

from base_core.framework.concurrency.models import StreamHandle

log = logging.getLogger(__name__)

T = TypeVar("T")

_STOP = object()  # sentinel — tells _loop to exit


def _wrap_error(
    fn: Callable[[], None],
    on_error: Callable[[BaseException], None],
) -> Callable[[], None]:
    def wrapped() -> None:
        try:
            fn()
        except BaseException as e:
            on_error(e)
    return wrapped


class TaskRunner:
    """
    Serial task queue backed by a single daemon thread.

    run(fn)      — submit a callable; runs in arrival order on the runner's thread
    stream(...)  — lossless producer/consumer: producer runs on this runner's thread,
                   consumer runs on a dedicated thread; returns StreamHandle
    shutdown()   — drain remaining tasks then stop the thread
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._queue: queue.Queue[Callable[[], None] | object] = queue.Queue()
        self._thread = threading.Thread(target=self._loop, daemon=True, name=name)
        self._thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        fn: Callable[[], None],
        *,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        """Submit a callable to run serially on this runner's thread."""
        self._queue.put(fn if on_error is None else _wrap_error(fn, on_error))

    def stream(
        self,
        producer: Callable[[threading.Event], Iterable[T]],
        *,
        on_item: Callable[[T], None],
        on_error: Callable[[BaseException], None] | None = None,
        on_complete: Callable[[], None] | None = None,
    ) -> StreamHandle:
        """
        Lossless producer/consumer stream.

        The producer generator runs on this runner's thread (so it cannot overlap with
        other run() tasks on the same runner). A dedicated consumer thread drains the
        item queue and calls on_item for every item in arrival order.

        Returns a StreamHandle whose stop_event signals the producer to exit and whose
        done_event is set once the consumer has drained all remaining items (including
        any item that was in-flight when stop was signalled). Callers that stop the
        stream and need to reclaim resources (e.g. hardware or shared-memory slots)
        should wait on done_event before doing so.
        """
        stop_event = threading.Event()
        done_event = threading.Event()
        item_queue: queue.Queue[object] = queue.Queue()
        _DONE_ITEM = object()

        def consumer_loop() -> None:
            try:
                while True:
                    item = item_queue.get()
                    if item is _DONE_ITEM:
                        if on_complete is not None:
                            on_complete()
                        return
                    on_item(item)  # type: ignore[arg-type]
            finally:
                # Set after all items (including the one in-flight at stop time)
                # have been processed. This guarantees that shared resources
                # touched by on_item are released before handle.wait() returns.
                done_event.set()

        consumer_thread = threading.Thread(
            target=consumer_loop, daemon=True, name=f"{self._name}.consumer"
        )

        def producer_loop() -> None:
            consumer_thread.start()
            try:
                for item in producer(stop_event):
                    item_queue.put(item)  # enqueue before checking stop
                    if stop_event.is_set():
                        break
            except BaseException as exc:
                if on_error is not None:
                    on_error(exc)
                else:
                    log.exception("TaskRunner stream producer error in %s", self._name)
            finally:
                item_queue.put(_DONE_ITEM)

        self._queue.put(producer_loop)
        return StreamHandle(stop_event=stop_event, done_event=done_event)

    def shutdown(self, wait: bool = True, timeout: float = 5.0) -> None:
        """Signal the runner to stop. With wait=True, blocks until the thread exits."""
        self._queue.put(_STOP)
        if wait:
            self._thread.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while True:
            fn = self._queue.get()
            if fn is _STOP:
                return
            try:
                fn()  # type: ignore[operator]
            except BaseException:
                log.exception("TaskRunner._loop: unhandled error in %s", self._name)
