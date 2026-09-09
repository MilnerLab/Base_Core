"""Main-process side of the device connection mode: record it, and say it out loud."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from base_core.framework.app.app_message import AppMessage, MessageLevel
from base_core.ipc.connection_mode import ConnectionMode

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceConnectionModeChanged:
    """A device worker reported what it connected to.

    Published on the main-process bus only; it never crosses IPC, so it is a plain
    dataclass and is not registered with the codec. ``requested`` alongside ``mode``
    is what lets a subscriber tell a deliberate mock from a demoted one.
    """

    worker_id: str
    mode: ConnectionMode
    requested: ConnectionMode
    reason: str = ""

    @property
    def demoted(self) -> bool:
        """True when hardware was asked for and a mock came back."""
        return self.mode == ConnectionMode.MOCK and self.requested == ConnectionMode.DEVICE


class DeviceHandleMixin:
    """Ask for hardware by default, and announce whatever actually answered.

    A mixin for the same reason as ``DeviceWorkerMixin``: the device handles do not
    share one parent. Most extend ``BaseWorkerHandle``, the five motorized ones go
    through ``MotorizedStageHandle``, and the spectrometer rides ``WriterWorkerHandle``.
    Mix it in **first**::

        class PicomotorHandle(DeviceHandleMixin, BaseWorkerHandle): ...

    Silence is the failure mode this exists to prevent. A rig quietly running on a
    fake looks exactly like a rig that works, right up until someone files the data.
    """

    #: What a device handle asks for unless told otherwise. Subclasses may override.
    DEFAULT_MODE: ConnectionMode = ConnectionMode.DEVICE

    def _default_start_mode(self) -> ConnectionMode:
        return self.DEFAULT_MODE

    @property
    def requested_mode(self) -> ConnectionMode:
        """What the next ``start()`` will ask for. Assignable before starting.

        A property over the single stored value rather than a second attribute: two
        knobs for one piece of state is how a handle ends up asking for one thing and
        judging the answer against another.
        """
        return self._requested_mode

    @requested_mode.setter
    def requested_mode(self, mode: ConnectionMode) -> None:
        self._requested_mode = ConnectionMode(mode)

    def _on_mode_reported(self, mode: ConnectionMode, reason: str) -> None:
        super()._on_mode_reported(mode, reason)
        if mode != ConnectionMode.MOCK:
            return

        requested = self._requested_start_mode()
        self._bus.publish(DeviceConnectionModeChanged(
            worker_id=self._worker_id, mode=mode, requested=requested, reason=reason,
        ))

        if requested == ConnectionMode.DEVICE:
            detail = f" ({reason})" if reason else ""
            text = f"{self._worker_id}: hardware unavailable — RUNNING ON A MOCK{detail}"
            log.error(text)
        else:
            text = f"{self._worker_id}: running on a mock, as requested"
        # WARNING rather than INFO on purpose: the status area retains a warning and
        # lets the operator navigate back to it, while an INFO clears itself after a
        # few seconds. This state lasts the whole session, so the notice should too.
        self._bus.publish(AppMessage(text, MessageLevel.WARNING))
