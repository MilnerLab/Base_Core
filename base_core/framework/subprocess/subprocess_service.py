from __future__ import annotations

from base_core.framework.subprocess.device_service import DeviceService
from base_core.framework.subprocess.worker_handle import WorkerHandle


class SubprocessService(DeviceService):
    """
    DeviceService for subprocesses built with SubprocessApp.

    Adds worker() to obtain a WorkerHandle for a named worker.
    All transport / lifecycle / EventBus wiring is inherited from DeviceService,
    so SubprocessService is a drop-in replacement for any DeviceService use.
    """

    def worker(self, name: str) -> WorkerHandle:
        """Return a handle for sending commands to a specific worker by name."""
        return WorkerHandle(service=self, name=name)
