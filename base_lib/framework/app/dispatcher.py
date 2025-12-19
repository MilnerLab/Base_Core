from __future__ import annotations

from typing import Callable, Protocol


UiPost = Callable[[Callable[[], None]], None]

class UiDispatcher(Protocol):
    """
    UI/main-thread dispatcher abstraction.
    Implement this in your GUI app (Qt/Tk/etc.).
    """

    def post(self, fn: Callable[[], None]) -> None:
        """Schedule fn to run on the UI/main thread."""
