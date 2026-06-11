from abc import ABC, abstractmethod

from base_core.framework.concurrency.task_runner import TaskRunner
from base_core.framework.events.event_bus import EventBus


class Step(ABC):
    def __init__(self, bus: EventBus, io: TaskRunner) -> None:
        self._bus = bus
        self._io = io

    @abstractmethod
    def stop(self) -> None:
        """Stop this step. Safe to call even if the step is not currently running."""
        ...
