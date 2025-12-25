from __future__ import annotations

from concurrent.futures import Future
import threading
from typing import Callable, Hashable, Iterable, Optional, Protocol, TypeVar

from base_core.framework.concurrency.models import StreamHandle


T = TypeVar("T")



class ITaskRunner(Protocol):
    """
    Qt-free concurrency interface.

    Threading note:
    - Callbacks are invoked from worker threads.
      In Qt apps: emit signals; don't touch widgets.
    """

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
        ...

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
        ...

    def cancel(self, key: Hashable) -> bool:
        ...

    def cancel_all(self) -> None:
        ...