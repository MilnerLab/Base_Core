from __future__ import annotations

import logging
from abc import ABC
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Iterator, TYPE_CHECKING

from base_core.framework.concurrency.task_runner import TaskRunner
from base_core.framework.routines.run_control import RunControl, ScanAborted

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

    What a routine gets from here
    -----------------------------
    - **A thread.** One serial TaskRunner. Dispatch onto it with @routine_thread, or
      _dispatch() directly. A whole run is normally ONE dispatched method.
    - **Run control.** pause() / resume() / abort() / step(), callable from any thread,
      plus checkpoint() to yield to them from the routine thread. See RunControl — the
      threading discipline there is load-bearing and explained in full.
    - **A run lifecycle.** run_lifecycle() publishes a failure event on any raise,
      always runs your cleanup, and always marks the run stopped.

    Deliberately NOT a state machine. A routine expresses its sequence in ordinary
    Python — for, try/finally, early return — because loops, nesting and early exit are
    exactly what a slot-indexed step list cannot express and what a measurement routine
    is made of. What Python does not give you is the threading discipline and the
    lifecycle guarantee above, so that is what lives here.

    Subclass pattern
    ----------------
    - __init__: accept bus + handles/config from DI; call super().__init__(bus)
    - _setup(): subscribe to events; store each unsub in self._unsubs
    - handlers: decorate with @routine_thread so they run on this routine's thread
    - the run itself: control.begin() on the caller's thread, then one @routine_thread
      method wrapping its body in run_lifecycle()

    DI registration example
    -----------------------
        c.register_factory(MyRoutine, lambda c: MyRoutine(
            bus=ctx.event_bus,
            handle=c.get(ScopeHandle),
        ))

    Usage
    -----
        routine = c.get(MyRoutine)   # creates instance + starts thread
        routine.pause() / routine.resume() / routine.abort()
        routine.dispose()            # unsubscribes + shuts down thread
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._unsubs: list[Callable[[], None]] = []
        self._runner = TaskRunner(name=type(self).__name__.lower())
        self._control = RunControl()
        self._setup()

    # ------------------------------------------------------------------
    # Run control — callable from any thread, never dispatched
    # ------------------------------------------------------------------

    @property
    def control(self) -> RunControl:
        return self._control

    @property
    def is_running(self) -> bool:
        return self._control.is_running

    @property
    def is_paused(self) -> bool:
        return self._control.is_paused

    @property
    def is_step_mode(self) -> bool:
        return self._control.is_step_mode

    def pause(self) -> None:
        """Hold the run at its next checkpoint. Devices keep position."""
        self._control.pause()

    def resume(self) -> None:
        """Continue from where a pause parked."""
        self._control.resume()

    def abort(self) -> None:
        """Stop the run at its next checkpoint, keeping whatever is already complete."""
        self._control.abort()

    def set_step_mode(self, enabled: bool) -> None:
        """Turn operator-advanced stepping on or off."""
        self._control.set_step_mode(enabled)

    def step(self, n: int = 1) -> None:
        """Permit ``n`` more advances while in step mode."""
        self._control.step(n)

    # ------------------------------------------------------------------
    # Routine-thread helpers
    # ------------------------------------------------------------------

    def checkpoint(self, *, on_hold: Callable[[], None] | None = None) -> None:
        """Yield to the operator: block while paused, raise ScanAborted if stopping.

        Call at every point the run can safely be interrupted, and nowhere else.
        ``on_hold`` fires once if this call actually parks — for suspending anything that
        would otherwise keep accumulating against a rig that is now standing still.
        """
        self._control.checkpoint(on_hold=on_hold)

    @contextmanager
    def run_lifecycle(
        self,
        failed: Callable[[BaseException], Any] | None = None,
        *,
        cleanup: Callable[[], None] | None = None,
    ) -> Iterator[None]:
        """Own the end of a run, however it ends.

        On any exception: log it and publish ``failed(exc)``. On :class:`ScanAborted`
        that escapes this far: log it and finish quietly — an abort is the operator
        getting what they asked for, not a failure. Always: run ``cleanup``, then mark
        the run stopped.

        The guarantee is the point. A run that raises somewhere its author did not
        anticipate must still announce *something*, or every waiter — a headless harness,
        a progress panel, an operator watching a status line — hangs forever on a run
        that is already dead, while the subprocesses it started stay up holding the
        hardware.

        ``failed`` is optional only because a routine nobody waits on has no event worth
        publishing, and inventing one so this parameter can be filled would be worse than
        leaving it out. If anything outside the routine can observe the run, pass it.
        """
        try:
            yield
        except ScanAborted:
            log.info("%s: run aborted", type(self).__name__)
        except Exception as exc:  # noqa: BLE001 — deliberately everything
            log.exception("%s: run failed", type(self).__name__)
            if failed is not None:
                self._bus.publish(failed(exc))
        finally:
            if cleanup is not None:
                try:
                    cleanup()
                except Exception:
                    log.exception("%s: cleanup failed", type(self).__name__)
            self._control.end()

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

    def dispose(self) -> None:
        """Retire this routine: unsubscribe all events and shut the runner thread down.

        This is *disposal*, not a way to stop a run — ``TaskRunner``'s stop sentinel
        queues behind whatever is already running, so calling this mid-run waits for the
        run to finish rather than ending it. To stop a run, call :meth:`abort`, which the
        routine thread notices at its next checkpoint; dispose afterwards.
        """
        for unsub in reversed(self._unsubs):
            unsub()
        self._runner.shutdown()
