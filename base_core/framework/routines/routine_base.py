from __future__ import annotations

import logging
from abc import ABC
from functools import wraps
from typing import Callable, TYPE_CHECKING

from base_core.framework.concurrency.task_runner import TaskRunner
from base_core.framework.routines.step import Step

if TYPE_CHECKING:
    from base_core.framework.events.event_bus import EventBus

log = logging.getLogger(__name__)


def routine_thread(fn: Callable) -> Callable:
    """Dispatch a BaseRoutine method onto the routine's own serial TaskRunner.

    Mirrors @worker_thread: the calling thread (EventBus publisher, often the IPC
    reader) returns immediately; the method body runs on the routine's thread.
    """
    @wraps(fn)
    def wrapper(self: BaseRoutine, *args, **kwargs) -> None:
        self._dispatch(lambda: fn(self, *args, **kwargs))
    return wrapper


class BaseRoutine(ABC):
    """
    Main-process routine. Factory-registered in DI; each container.get() produces a
    fresh instance that immediately starts its own serial task queue + thread.

    Step sequencing
    ---------------
    Register steps via add_step(). Each Step carries its own slot number; the routine
    sorts by slot so execution order is defined by the step, not by call order.

        add_step(GatherSpectraStep(slot=0, handle=h))
        add_step(FitPhaseStep(slot=1, config=cfg))

    - advance_step()  — stop current step, start the next
    - revert_step()   — stop current step, restart the previous
    - reset_step()    — reset current step in place (no transition)
    - current_step    — the active Step object, or None if none registered

    Subclass pattern
    ----------------
    - __init__: accept bus + handles/config from DI; call super().__init__(bus), then add_step()
    - _setup(): subscribe to events; store each unsub in self._unsubs
    - handlers: decorate with @routine_thread so they run on this routine's thread

    DI registration example
    -----------------------
        c.register_factory(MyRoutine, lambda c: MyRoutine(
            bus=ctx.event_bus,
            handle=c.get(ScopeHandle),
        ))

    Usage
    -----
        routine = c.get(MyRoutine)   # creates instance + starts thread
        ...
        routine.stop()               # unsubscribes + shuts down thread
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._unsubs: list[Callable[[], None]] = []
        self._runner = TaskRunner(name=type(self).__name__.lower())
        self._step_list: list[Step] = []   # sorted by step.slot
        self._step_index: int = 0
        self._setup()

    # ------------------------------------------------------------------
    # Step registration
    # ------------------------------------------------------------------

    def add_step(self, step: Step) -> None:
        """Register a step. Steps are sorted by slot number, not insertion order."""
        self._step_list.append(step)
        self._step_list.sort(key=lambda s: s.slot)

    # ------------------------------------------------------------------
    # Step state (read from any thread; mutations dispatched to runner)
    # ------------------------------------------------------------------

    @property
    def current_step(self) -> Step | None:
        if not self._step_list:
            return None
        return self._step_list[self._step_index]

    @property
    def step_index(self) -> int:
        return self._step_index

    # ------------------------------------------------------------------
    # Step transitions
    # ------------------------------------------------------------------

    def advance_step(self) -> None:
        """Stop the current step and start the next. No-op at the last step."""
        self._dispatch(self._do_advance)

    def revert_step(self) -> None:
        """Stop the current step and restart the previous. No-op at the first step."""
        self._dispatch(self._do_revert)

    def reset_step(self) -> None:
        """Reset the current step in place."""
        self._dispatch(self._do_reset)

    def _do_advance(self) -> None:
        if not self._step_list or self._step_index >= len(self._step_list) - 1:
            return
        self._step_list[self._step_index].stop()
        self._step_index += 1
        self._step_list[self._step_index].start()

    def _do_revert(self) -> None:
        if not self._step_list or self._step_index <= 0:
            return
        self._step_list[self._step_index].stop()
        self._step_index -= 1
        self._step_list[self._step_index].start()

    def _do_reset(self) -> None:
        step = self.current_step
        if step is not None:
            step.reset()

    # ------------------------------------------------------------------
    # Infrastructure
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        """Override to subscribe to events. Store each unsub in self._unsubs."""

    def _dispatch(self, fn: Callable[[], None]) -> None:
        self._runner.run(
            fn,
            on_error=lambda _: log.exception("Unhandled error in %s", type(self).__name__),
        )

    def stop(self) -> None:
        """Unsubscribe all events and shut down the runner thread."""
        for unsub in reversed(self._unsubs):
            unsub()
        self._runner.shutdown()
