from __future__ import annotations

from dataclasses import dataclass

from base_core.ipc.codec import register
from base_core.ipc.connection_mode import ConnectionMode
from base_core.ipc.message import OKReply, Request


@register
@dataclass(frozen=True)
class WorkerStartedReply(OKReply):
    """Reply to StartWorker, carrying what the worker actually connected to.

    A subclass of ``OKReply`` rather than a new reply type, so every existing
    ``_on_start_reply(reply: OKReply)`` keeps working untouched. ``mode`` is the
    outcome, which is not always what was asked for: a DEVICE request whose hardware
    refused comes back MOCK, and that is how the demotion reaches the main process.

    ``reason`` carries the controller's own failure text on a demotion, empty
    otherwise. The operator needs to know *why* the rig is on a fake, and by the time
    the warning reaches the screen the subprocess log is somewhere else entirely.
    """

    mode: ConnectionMode = ConnectionMode.NONE
    reason: str = ""


@register
@dataclass(frozen=True)
class StartWorker(Request[WorkerStartedReply]):
    """Start a worker.

    ``mode`` says what the caller wants on the other end. It defaults to ``NONE`` so
    every worker that is not a device worker keeps starting exactly as before, without
    having to know this field exists; only device handles send DEVICE or MOCK.
    """

    worker_id: str = ""
    mode: ConnectionMode = ConnectionMode.NONE


@register
@dataclass(frozen=True)
class PauseWorker(Request[OKReply]):
    worker_id: str = ""


@register
@dataclass(frozen=True)
class ResumeWorker(Request[OKReply]):
    worker_id: str = ""


@register
@dataclass(frozen=True)
class StopWorker(Request[OKReply]):
    worker_id: str = ""
