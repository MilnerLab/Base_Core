from __future__ import annotations

from concurrent.futures import Future
from typing import Callable, Hashable, Optional, Protocol, TypeVar

T = TypeVar("T")



class ITaskRunner(Protocol):
    """
    Runs functions in background and executes callbacks (optionally) on the UI thread.

    Cancel semantics:
    - Future.cancel() only works if the task has not started yet.
    - For running tasks, use drop_outdated=True to ignore stale results.
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

    def cancel(self, key: Hashable) -> bool:
        ...

    def cancel_all(self) -> None:
        ...