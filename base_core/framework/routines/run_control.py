"""``RunControl`` — start/pause/resume/stop for a routine that is busy driving hardware.

Why this is not simply ``BaseRoutine.stop()``
---------------------------------------------
``BaseRoutine.stop()`` shuts the ``TaskRunner`` down, and ``TaskRunner``'s ``_STOP``
sentinel goes to the *back* of the queue (``task_runner.py``). A routine whose whole run
is one long dispatched method is therefore unstoppable by that route: ``stop()`` returns
five seconds later having achieved nothing while a daemon thread keeps commanding stages.

So run control cannot be a message on the routine's own queue. Every operator-facing
method here is called on the **caller's** thread and sets a flag; the routine thread
notices at the next checkpoint it reaches. That is the only ordering that works, and it
is why none of these methods is decorated with ``@routine_thread``.

The cost is granularity: a pause or an abort lands at the next checkpoint, never
mid-move. On hardware that is not merely acceptable but required — a controller that
holds its lock across a blocking wait-for-motion cannot be interrupted from another
thread anyway, so a request to stop *has* to be a request the moving code picks up when
it is next between operations.

Two primitives for stepping, not one
------------------------------------
:attr:`_step_mode` is the MODE and :attr:`_step_permits` is the QUEUE OF PRESSES. They
answer different questions — "are we stepping?" and "how many advances are owed?" — and
a press made before the routine reaches the gate must not be lost, which a bare Event
cannot express.
"""
from __future__ import annotations

import logging
import threading

from typing import Callable

log = logging.getLogger(__name__)

#: How often a parked routine wakes to re-check its flags. Short enough that an abort
#: raised while parked is noticed promptly, long enough to cost nothing.
DEFAULT_POLL_S = 0.05


class ScanAborted(RuntimeError):
    """The operator asked the run to stop, and a checkpoint noticed.

    Raised on the routine thread, so a routine unwinds through its own ``try/finally``
    and gets its flush-and-park for free. Catch it where partial results are still worth
    keeping — typically around an inner loop, to record what was collected before the
    stop — and let it propagate everywhere else.
    """


class RunControl:
    """The pause/abort/step state of one run. Safe to touch from any thread."""

    def __init__(self, poll_s: float = DEFAULT_POLL_S) -> None:
        self._poll_s = poll_s
        # Set from the caller's thread, read on the routine thread. NOT dispatched — a
        # dispatched abort would queue behind the very loop it is meant to stop.
        self._abort = threading.Event()
        # Pause gate, same threading discipline as _abort: set = free to run, cleared =
        # paused. The routine thread blocks on it at each checkpoint, so a pause (like an
        # abort) takes effect at the next checkpoint, never mid-move. Starts set so a
        # fresh run is never born paused.
        self._resume = threading.Event()
        self._resume.set()
        self._step_mode = threading.Event()
        self._step_permits = threading.Semaphore(0)
        self._running = threading.Event()

    # -- state ------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    @property
    def is_paused(self) -> bool:
        return self._running.is_set() and not self._resume.is_set()

    @property
    def is_aborting(self) -> bool:
        return self._abort.is_set()

    @property
    def is_step_mode(self) -> bool:
        return self._step_mode.is_set()

    # -- run lifecycle ----------------------------------------------------

    def begin(self) -> None:
        """Arm for a fresh run. Call on the caller's thread, before dispatching the run.

        Clearing the abort and re-setting the pause gate here rather than at the end of
        the previous run means a run that was paused or aborted cannot leave the next one
        born parked. Stale permits are drained for the same reason: a press that arrived
        after the last setpoint would otherwise be spent silently on the first gate of
        this run, advancing past a position the operator never looked at.
        """
        self._abort.clear()
        self._resume.set()
        while self._step_permits.acquire(blocking=False):
            pass
        self._running.set()

    def end(self) -> None:
        """The run is over, however it ended."""
        self._running.clear()

    # -- operator API (any thread; never dispatched) ----------------------

    def abort(self) -> None:
        """Request an orderly stop. Takes effect at the next checkpoint.

        An in-flight move or acquisition is never interrupted. Whatever the routine has
        already completed is kept — the point of stopping at a checkpoint rather than
        killing a thread is that the run ends somewhere it can describe.
        """
        if not self._running.is_set():
            return
        log.info("RunControl: abort requested — will stop at the next checkpoint")
        self._abort.set()
        # If a pause is in force the routine thread is parked in checkpoint(); release it
        # so it wakes, sees the abort and unwinds. Without this an abort requested while
        # paused would hang until someone resumed.
        self._resume.set()

    def pause(self) -> None:
        """Request a pause. Takes effect at the next checkpoint, like an abort.

        Devices hold position while paused; nothing is commanded until :meth:`resume`.
        """
        if not self._running.is_set():
            return
        log.info("RunControl: pause requested — will hold at the next checkpoint")
        self._resume.clear()

    def resume(self) -> None:
        """Lift a pause; the run continues from where it parked."""
        if not self._resume.is_set():
            log.info("RunControl: resuming")
        self._resume.set()

    def set_step_mode(self, enabled: bool) -> None:
        """Turn operator-advanced stepping on or off.

        Takes effect at the next step gate, so work already in progress always finishes —
        arming this mid-operation never strands a half-written result. Turning it *off*
        frees a routine already parked at a gate within one poll interval, so it runs on
        without waiting for a press it no longer needs.
        """
        if enabled == self._step_mode.is_set():
            return
        if enabled:
            log.info("RunControl: step mode ON — each gate waits for step()")
            self._step_mode.set()
            return
        log.info("RunControl: step mode OFF — free running")
        self._step_mode.clear()

    def step(self, n: int = 1) -> None:
        """Permit ``n`` more advances. No-op unless running.

        Safe to call ahead of the routine reaching a gate: permits accumulate, so the
        count is what the operator pressed, not what happened to be timed right.
        """
        if not self._running.is_set():
            return
        for _ in range(max(1, n)):
            self._step_permits.release()

    # -- routine-thread API -----------------------------------------------

    def checkpoint(self, *, on_hold: Callable[[], None] | None = None) -> None:
        """Yield to the operator: block while paused, raise if stopping.

        Call this at every point where the run can safely be interrupted — between
        acquisitions, before a move — and nowhere else. ``on_hold`` fires once if this
        call actually parks, which is how a caller suspends something for the duration
        (a data stream that would otherwise pile up against one stationary point) without
        this class needing to know such a thing exists.

        Polls rather than a bare ``wait()`` so an abort raised while parked is noticed
        promptly. :meth:`abort` also sets the resume gate, but polling keeps the contract
        robust to either order.

        Raises :class:`ScanAborted` if an abort is pending, including one raised *while*
        this call was parked.
        """
        if self._abort.is_set():
            raise ScanAborted("run aborted")
        if self._resume.is_set():
            return
        if on_hold is not None:
            on_hold()
        while not self._resume.wait(self._poll_s):
            if self._abort.is_set():
                raise ScanAborted("run aborted while paused")
        if self._abort.is_set():
            raise ScanAborted("run aborted while paused")

    def wait_for_step(self, *, on_hold: Callable[[], None] | None = None) -> None:
        """Block until the operator permits one advance. No-op unless in step mode.

        Consumes exactly one permit. Returns normally — letting the run continue — if
        step mode is switched off while parked. Raises :class:`ScanAborted` on abort,
        the same as :meth:`checkpoint`, so a routine parked at a gate can still be
        stopped.
        """
        if not self._step_mode.is_set():
            return
        if self._abort.is_set():
            raise ScanAborted("run aborted")
        if on_hold is not None:
            on_hold()
        while not self._step_permits.acquire(timeout=self._poll_s):
            if self._abort.is_set():
                raise ScanAborted("run aborted while stepping")
            if not self._step_mode.is_set():
                return
