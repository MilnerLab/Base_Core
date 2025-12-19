from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Hashable, Iterable, Optional

from concurrent.futures import Future, ThreadPoolExecutor

from base_lib.framework.app.dispatcher import UiPost
from base_lib.framework.concurrency.interfaces import T, ITaskRunner


@dataclass
class _Entry:
    token: int
    future: Future
    stop_event: Optional[threading.Event] = None


@dataclass(frozen=True)
class StreamHandle:
    stop_event: threading.Event
    future: Future[None]

    def stop(self) -> None:
        self.stop_event.set()


class TaskRunner(ITaskRunner):
    """
    One place for background execution:
    - run(): one-shot (exactly one result)
    - stream(): producer yields many items (UI gets always the latest item)
    - key/cancel_previous/drop_outdated: "latest wins" semantics
    - ui_post: ensures callbacks run on UI thread if provided
    """

    def __init__(self, executor: ThreadPoolExecutor, *, ui_post: UiPost | None = None) -> None:
        self._executor = executor
        self._ui_post = ui_post
        self._lock = threading.RLock()
        self._entries: dict[Hashable, _Entry] = {}

    # ----------------- internal helpers -----------------

    def _dispatch(self, cb: Callable[[], None]) -> None:
        if self._ui_post is None:
            cb()
        else:
            self._ui_post(cb)

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
            # Best-effort cancel (only works if not started) + cooperative stop for streams
            prev.future.cancel()
            if prev.stop_event is not None:
                prev.stop_event.set()

        # We don't set the new entry here; done after submit.
        return token

    def _set_entry(self, key: Hashable | None, token: int, future: Future, stop_event: Optional[threading.Event]) -> None:
        if key is None:
            return
        self._entries[key] = _Entry(token=token, future=future, stop_event=stop_event)

    def _is_latest(self, key: Hashable | None, token: int, *, drop_outdated: bool) -> bool:
        if key is None or not drop_outdated:
            return True
        cur = self._entries.get(key)
        return cur is not None and cur.token == token

    # ----------------- public API -----------------

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
                key,
                cancel_previous=cancel_previous,
                new_stop_event=None,
            )

        fut: Future[T] = self._executor.submit(fn)

        with self._lock:
            self._set_entry(key, token, fut, stop_event=None)

        def _done(f: Future[T]) -> None:
            with self._lock:
                if not self._is_latest(key, token, drop_outdated=drop_outdated):
                    return

            try:
                res = f.result()
            except BaseException as e:
                if on_error is not None:
                    self._dispatch(lambda: on_error(e))
                return

            if on_success is not None:
                self._dispatch(lambda: on_success(res))

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
    ) -> StreamHandle:
        stop_event = threading.Event()

        with self._lock:
            token = self._next_token_and_cancel_prev(
                key,
                cancel_previous=cancel_previous,
                new_stop_event=stop_event,
            )

        # "latest-only" delivery: UI never gets a backlog, only the newest item.
        latest_lock = threading.Lock()
        latest: Optional[T] = None
        scheduled = False

        def schedule_flush() -> None:
            nonlocal scheduled

            def flush() -> None:
                nonlocal scheduled
                # If stream is outdated, ignore UI updates.
                with self._lock:
                    if not self._is_latest(key, token, drop_outdated=drop_outdated):
                        scheduled = False
                        return

                with latest_lock:
                    value = latest
                    scheduled = False

                if value is not None:
                    on_item(value)

            self._dispatch(flush)

        def publish(item: T) -> None:
            nonlocal latest, scheduled
            with latest_lock:
                latest = item
                if scheduled:
                    return
                scheduled = True
            schedule_flush()

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
                    self._dispatch(lambda: on_error(e))
            finally:
                if on_complete is not None:
                    self._dispatch(on_complete)

        fut: Future[None] = self._executor.submit(loop)

        with self._lock:
            self._set_entry(key, token, fut, stop_event=stop_event)

        return StreamHandle(stop_event=stop_event, future=fut)

    def cancel(self, key: Hashable) -> bool:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return False
            if entry.stop_event is not None:
                entry.stop_event.set()
            return entry.future.cancel()

    def cancel_all(self) -> None:
        with self._lock:
            keys = list(self._entries.keys())
        for k in keys:
            self.cancel(k)