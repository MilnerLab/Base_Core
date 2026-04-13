from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, DefaultDict, Generic, TypeVar
import threading

TEvent = TypeVar("TEvent")


@dataclass()
class EventBus:
    def __post_init__(self) -> None:
        self._lock = threading.RLock()
        self._subs: DefaultDict[type, list[Callable[[object], None]]] = defaultdict(list)

    def subscribe(
        self,
        event_type: type[TEvent],
        handler: Callable[[TEvent], None],
    ) -> Callable[[], None]:
        with self._lock:
            handlers = self._subs[event_type]
            handlers.append(handler)  # type: ignore[arg-type]

        def unsubscribe() -> None:
            with self._lock:
                handlers = self._subs.get(event_type, [])
                if handler in handlers:
                    handlers.remove(handler)  # type: ignore[arg-type]

        return unsubscribe

    def publish(self, event: object) -> None:
        event_type = type(event)

        with self._lock:
            handlers = list(self._subs.get(event_type, []))

        for handler in handlers:
            try:
                handler(event)
            except Exception:
                pass