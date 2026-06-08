from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Callable, Hashable, Iterable, Optional, TypeVar
from concurrent.futures import Future, ThreadPoolExecutor

from base_core.framework.concurrency.models import StreamHandle


# Sentinel distinct from any possible item, so a producer that legitimately
# yields None is never mistaken for "no value".
_NOTHING = object()
T = TypeVar("T")


@dataclass
class _Entry:
    token: int
    future: Future
    stop_event: Optional[threading.Event] = None


class TaskRunner:
    """
    Background execution helper:
    - run(): one-shot
    - stream(): producer yields many items
        * coalesce=True  -> "latest wins": intermediate items may be dropped,
                            only the most recent is delivered (good for UI frames)
        * coalesce=False -> lossless: every item delivered in order via an
                            unbounded queue drained on a dedicated consumer
                            thread (good for control/message streams)
    - key/cancel_previous/drop_outdated: "latest wins" task-supersession.
    """

    def __init__(self, executor: ThreadPoolExecutor) -> None:
        self._executor = executor
        self._lock = threading.RLock()
        self._entries: dict[Hashable, _Entry] = {}

    def _next_token_and_cancel_prev(
        self,
        key: Hashable | None,
        *,
        cancel_previous: bool,
        new_stop_event: Optional[threading.Event],
    ) -> int:
        if key is None:
            return 0

        prev = self._entries.get(key)
        if prev is None:
            return 1

        token = prev.token + 1

        if cancel_previous:
            prev.future.cancel()
            if prev.stop_event is not None:
                prev.stop_event.set()

        return token

    def _set_entry(
        self,
        key: Hashable | None,
        token: int,
        future: Future,
        stop_event: Optional[threading.Event],
    ) -> None:
        if key is None:
            return
        self._entries[key] = _Entry(token=token, future=future, stop_event=stop_event)

    def _is_latest(self, key: Hashable | None, token: int, *, drop_outdated: bool) -> bool:
        if key is None or not drop_outdated:
            return True
        cur = self._entries.get(key)
        return cur is not None and cur.token == token

    def run(
        self,
        fn: Callable[[], T],
        *,
        on_success: Optional[Callable[[T], None]] = None,
        on_error: Optional[Callable[[BaseException], None]] = None,
        key: Hashable | None = None,
        cancel_previous: bool = False,
        drop_outdated: bool = True,
    ) -> Future[T]:
        with self._lock:
            token = self._next_token_and_cancel_prev(
                key, cancel_previous=cancel_previous, new_stop_event=None
            )

        fut: Future[T] = self._executor.submit(fn)

        with self._lock:
            self._set_entry(key, token, fut, stop_event=None)

        if on_success is None and on_error is None:
            return fut

        def _done(f: Future[T]) -> None:
            with self._lock:
                if not self._is_latest(key, token, drop_outdated=drop_outdated):
                    return
            try:
                res = f.result()
            except BaseException as e:
                if on_error is not None:
                    on_error(e)
                return
            if on_success is not None:
                on_success(res)

        fut.add_done_callback(_done)
        return fut

    def stream(
        self,
        producer: Callable[[threading.Event], Iterable[T]],
        *,
        on_item: Callable[[T], None],
        on_error: Optional[Callable[[BaseException], None]] = None,
        on_complete: Optional[Callable[[], None]] = None,
        key: Hashable | None = None,
        cancel_previous: bool = False,
        drop_outdated: bool = True,
        coalesce: bool = True,
    ) -> StreamHandle:
        stop_event = threading.Event()

        with self._lock:
            token = self._next_token_and_cancel_prev(
                key, cancel_previous=cancel_previous, new_stop_event=stop_event
            )

        if coalesce:
            loop = self._make_coalescing_loop(
                producer, on_item, on_error, on_complete,
                key, token, stop_event, drop_outdated,
            )
        else:
            loop = self._make_lossless_loop(
                producer, on_item, on_error, on_complete,
                key, token, stop_event, drop_outdated,
            )

        fut: Future[None] = self._executor.submit(loop)

        with self._lock:
            self._set_entry(key, token, fut, stop_event=stop_event)

        return StreamHandle(stop_event=stop_event, future=fut)

    # ---------- stream strategies ----------

    def _make_coalescing_loop(
        self, producer, on_item, on_error, on_complete,
        key, token, stop_event, drop_outdated,
    ):
        latest_lock = threading.Lock()
        latest = _NOTHING
        scheduled = False

        def flush_latest() -> None:
            nonlocal scheduled, latest
            with self._lock:
                if not self._is_latest(key, token, drop_outdated=drop_outdated):
                    with latest_lock:
                        scheduled = False
                    return
            with latest_lock:
                value = latest
                latest = _NOTHING
                scheduled = False
            if value is not _NOTHING:
                on_item(value)

        def publish(item) -> None:
            nonlocal latest, scheduled
            with latest_lock:
                latest = item
                if scheduled:
                    return
                scheduled = True
            flush_latest()

        def loop() -> None:
            try:
                for item in producer(stop_event):
                    if stop_event.is_set():
                        break
                    with self._lock:
                        if not self._is_latest(key, token, drop_outdated=drop_outdated):
                            break
                    publish(item)
            except BaseException as e:
                if on_error is not None:
                    on_error(e)
            finally:
                if on_complete is not None:
                    on_complete()

        return loop

    def _make_lossless_loop(
        self, producer, on_item, on_error, on_complete,
        key, token, stop_event, drop_outdated,
    ):
        # Producer thread fills the queue; a consumer thread drains it and
        # calls on_item for every item in order. Keeping on_item off the
        # producer thread means a slow handler never blocks stdout reading.
        q: queue.Queue = queue.Queue()
        _DONE = object()

        def consumer() -> None:
            while True:
                item = q.get()
                if item is _DONE:
                    return
                with self._lock:
                    if not self._is_latest(key, token, drop_outdated=drop_outdated):
                        return
                on_item(item)

        def loop() -> None:
            consumer_thread = threading.Thread(target=consumer, daemon=True)
            consumer_thread.start()
            try:
                for item in producer(stop_event):
                    if stop_event.is_set():
                        break
                    with self._lock:
                        if not self._is_latest(key, token, drop_outdated=drop_outdated):
                            break
                    q.put(item)
            except BaseException as e:
                if on_error is not None:
                    on_error(e)
            finally:
                q.put(_DONE)
                consumer_thread.join()
                if on_complete is not None:
                    on_complete()

        return loop

    def cancel(self, key: Hashable) -> bool:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return False
            if entry.stop_event is not None:
                entry.stop_event.set()
            cancelled = entry.future.cancel()
        # A running stream can't be future.cancel()'d, but the stop_event will
        # halt it; report success when we have a stop_event to signal with.
        return cancelled or entry.stop_event is not None

    def cancel_all(self) -> None:
        with self._lock:
            keys = list(self._entries.keys())
        for k in keys:
            self.cancel(k)