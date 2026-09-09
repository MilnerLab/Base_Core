"""Subprocess-side connection policy for workers that front real hardware."""
from __future__ import annotations

import logging

from base_core.ipc.connection_mode import ConnectionMode
from base_core.ipc.worker_messages import StartWorker

log = logging.getLogger(__name__)


class DeviceWorkerMixin:
    """Try the hardware, fall back to the mock, and remember which one you got.

    A mixin rather than a base class, because the ten device workers sit on three
    unrelated chains — ``WriterWorker`` (oscilloscope, spectrometer), ``ThreadedWorker``
    (picomotor, servo shutter) and ``MotorizedWorker`` (the five motorized devices).
    A common ancestor would either force hardware semantics onto the phase-control
    workers, which front nothing, or fork into three parallel base classes. Mix it in
    **first**, so its ``_begin_start`` wins the MRO::

        class PicomotorWorker(DeviceWorkerMixin, ThreadedWorker): ...

    It deliberately declares no ``__init__``: the three bases take incompatible
    constructor signatures, so the state below stays as class-level defaults that
    become instance attributes on first write.

    Subclasses implement ``_connect`` and ``_connect_mock``, and call
    ``_open_device()`` from their ``_start()`` in place of building a driver.
    """

    #: Set from the StartWorker request, before _start() runs.
    _requested_mode: ConnectionMode = ConnectionMode.DEVICE
    _demotion_reason: str = ""

    # -- the seam into BaseWorker's start path -----------------------------

    def _begin_start(self, msg: StartWorker) -> None:
        self._requested_mode = ConnectionMode(msg.mode)
        self._demotion_reason = ""
        # A start that never reaches _open_device (an early return, say) must not
        # report a stale mode from the previous run.
        self._connection_mode = ConnectionMode.NONE
        self._start()

    def _start_reason(self) -> str:
        return self._demotion_reason

    # -- the policy --------------------------------------------------------

    def _open_device(self):
        """Return an opened driver, and record what it actually is.

        A MOCK request never touches the hardware. A DEVICE request tries it, and on
        any failure demotes rather than raising: an unreachable instrument should cost
        the operator a warning banner, not the whole panel.

        A failing *mock* is left to raise. That is a bug in the mock, not a fact about
        the lab, and it surfaces as an ErrorReply from the start path.
        """
        if self._requested_mode == ConnectionMode.MOCK:
            self._connection_mode = ConnectionMode.MOCK
            return self._connect_mock()
        try:
            driver = self._connect()
        except Exception as exc:
            self._demotion_reason = f"{type(exc).__name__}: {exc}"
            log.error("%s: hardware unavailable (%s) — FALLING BACK TO THE MOCK",
                      getattr(self, "_worker_id", type(self).__name__),
                      self._demotion_reason)
            self._connection_mode = ConnectionMode.MOCK
            return self._connect_mock()
        self._connection_mode = ConnectionMode.DEVICE
        return driver

    # -- hooks -------------------------------------------------------------

    def _connect(self):
        """Build AND open the real driver. Raise if the hardware cannot be reached.

        Opening belongs in here, not in the caller: a driver that constructs happily
        and fails on ``open()`` is the ordinary failure for PyVISA, pyserial and TCP,
        and it has to demote like any other.
        """
        raise NotImplementedError

    def _connect_mock(self):
        """Build and open the mock. Must not raise."""
        raise NotImplementedError
