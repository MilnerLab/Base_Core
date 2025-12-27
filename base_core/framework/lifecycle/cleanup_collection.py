from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List

@dataclass
class CleanupCollection:
    _actions: List[Callable[[], None]] = field(default_factory=list)

    def add(self, fn: Callable[[], None]) -> None:
        self._actions.append(fn)

    def clear(self) -> None:
        actions, self._actions = self._actions, []
        for fn in reversed(actions):
            try:
                fn()
            except Exception:
                pass
