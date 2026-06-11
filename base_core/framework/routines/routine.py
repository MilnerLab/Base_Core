from abc import ABC, abstractmethod

from base_core.framework.concurrency.task_runner import TaskRunner
from base_core.framework.events.event_bus import EventBus


class Routine(ABC):
    def __init__(self, bus: EventBus, io: TaskRunner) -> None:
        self._bus = bus
        self._io = io
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    @abstractmethod
    def start(self) -> None:
        """Begin the routine from its first step. Idempotent."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Abort the routine regardless of which step is active. Idempotent."""
        ...
