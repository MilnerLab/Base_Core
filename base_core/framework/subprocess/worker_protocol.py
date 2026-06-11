from __future__ import annotations

from dataclasses import dataclass

from base_core.framework.subprocess.messages import Kind, Message


# ---------------------------------------------------------------------------
# Worker lifecycle messages (app-level: no target field, handled by SubprocessApp)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StartWorker(Message):
    """Command: start a named worker thread in the subprocess."""
    NAME = "start_worker"
    KIND = Kind.COMMAND
    worker_name: str = ""


@dataclass(frozen=True)
class StopWorker(Message):
    """Command: stop a named worker thread in the subprocess."""
    NAME = "stop_worker"
    KIND = Kind.COMMAND
    worker_name: str = ""


@dataclass(frozen=True)
class WorkerError(Message):
    """Event emitted by SubprocessApp when a worker thread crashes."""
    NAME = "worker_error"
    KIND = Kind.EVENT
    worker_name: str = ""
    error: str = ""
