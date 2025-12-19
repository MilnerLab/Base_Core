from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional
import logging


@dataclass
class Lifecycle:
    """
    Collect shutdown hooks and run them in reverse order.
    """

    _shutdown_hooks: List[Callable[[], None]] = field(default_factory=list)

    def add_shutdown_hook(self, fn: Callable[[], None]) -> None:
        self._shutdown_hooks.append(fn)

    def shutdown(self, log: Optional[logging.Logger] = None) -> None:
        for fn in reversed(self._shutdown_hooks):
            try:
                fn()
            except Exception:
                if log:
                    log.exception("Error during shutdown hook")
