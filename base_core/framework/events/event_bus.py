from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import Callable, DefaultDict, Optional, TypeVar

log = logging.getLogger(__name__)

TEvent = TypeVar("TEvent")


class EventBus:
    """
    Synchronous, type-dispatched, in-process event bus.

    Handlers are invoked on the thread that calls publish(). Keep them fast and
    non-blocking; for slow work, hand off to an executor inside the handler.

    Dispatch is by the event's concrete type (exact match). Optionally a
    subscription can be scoped to a `source`, so that with a single bus shared
    by multiple devices, a handler only sees events whose `.source` matches.
    Events without a `source` attribute are delivered only to unscoped handlers.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # event_type -> list of (handler, source_or_None)
        self._subs: DefaultDict[
            type, list[tuple[Callable[..., None], Optional[str]]]
        ] = defaultdict(list)

    def subscribe(
        self,
        event_type: type[TEvent],
        handler: Callable[[TEvent], None],
        *,
        source: Optional[str] = None,
    ) -> Callable[[], None]:
        """
        Register `handler` for events of exactly `event_type`.

        If `source` is given, the handler only fires for events whose `.source`
        equals it. If `source` is None, the handler fires for every event of
        that type regardless of source.
        """
        entry = (handler, source)
        with self._lock:
            self._subs[event_type].append(entry)

        def unsubscribe() -> None:
            with self._lock:
                handlers = self._subs.get(event_type)
                if handlers and entry in handlers:
                    handlers.remove(entry)
                    if not handlers:
                        del self._subs[event_type]

        return unsubscribe

    def publish(self, event: object) -> None:
        with self._lock:
            entries = list(self._subs.get(type(event), []))

        event_source = getattr(event, "source", None)

        for handler, want_source in entries:
            if want_source is not None and want_source != event_source:
                continue
            try:
                handler(event)
            except Exception:
                log.exception(
                    "EventBus handler %r failed for event %s",
                    getattr(handler, "__qualname__", handler),
                    type(event).__name__,
                )