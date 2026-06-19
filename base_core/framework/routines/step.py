from __future__ import annotations

from abc import ABC, abstractmethod


class Step(ABC):
    """
    A discrete phase of a BaseRoutine.

    Each step declares its own slot number, which determines its position in the
    routine's execution order regardless of the order add_step() is called.

    All lifecycle methods are called on the routine's own thread.
    """

    def __init__(self, slot: int) -> None:
        self._slot = slot

    @property
    def slot(self) -> int:
        return self._slot

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def start(self) -> None:
        """Enter this step."""

    @abstractmethod
    def stop(self) -> None:
        """Leave this step (advance or revert)."""

    @abstractmethod
    def reset(self) -> None:
        """Reset this step to its initial state without leaving it."""
