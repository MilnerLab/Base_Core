from __future__ import annotations

import dataclasses
from concurrent.futures import Future
from typing import TYPE_CHECKING, Callable, Optional

from base_core.framework.subprocess.messages import ErrorMessage, Message
from base_core.framework.subprocess.worker_protocol import StartWorker, StopWorker

if TYPE_CHECKING:
    from base_core.framework.subprocess.subprocess_service import SubprocessService


class WorkerHandle:
    """
    Main-process command proxy for a named worker within a SubprocessService.

    Responsibilities: send typed commands, start/stop the worker thread.
    Buffer configuration, slot grants, ItemWritten/ItemAck routing, and
    consumer registration are all owned by SubprocessService — not here.
    """

    def __init__(
        self,
        service: SubprocessService,
        name: str,
    ) -> None:
        self._service = service
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    # ------------------------------------------------------------------
    # Worker thread lifecycle
    # ------------------------------------------------------------------

    def start_async(
        self,
        *,
        post_start: Optional[Callable[[], None]] = None,
        key: Optional[str] = None,
        on_success: Optional[Callable] = None,
        on_error: Optional[Callable[[BaseException], None]] = None,
    ) -> Future:
        return self._service.run_async(
            lambda: self._do_start(post_start=post_start),
            key=key or f"worker.{self._name}.start",
            on_success=on_success,
            on_error=on_error,
        )

    def stop(self) -> None:
        reply = self._service.request_typed(StopWorker(worker_name=self._name))
        if isinstance(reply, ErrorMessage):
            raise RuntimeError(f"StopWorker failed for {self._name!r}: {reply.error}")

    def restart_async(
        self,
        *,
        post_start: Optional[Callable[[], None]] = None,
        key: Optional[str] = None,
        on_success: Optional[Callable] = None,
        on_error: Optional[Callable[[BaseException], None]] = None,
    ) -> Future:
        self.stop()
        return self.start_async(
            post_start=post_start,
            key=key,
            on_success=on_success,
            on_error=on_error,
        )

    # ------------------------------------------------------------------
    # Command API
    # ------------------------------------------------------------------

    def send(self, cmd: Message) -> None:
        self._service.send(self._stamp(cmd))

    def request_async(
        self,
        cmd: Message,
        *,
        timeout_s: float = 2.0,
        key: Optional[str] = None,
        cancel_previous: bool = False,
        drop_outdated: bool = True,
        on_success: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
    ) -> Future:
        return self._service.request_async(
            self._stamp(cmd),
            timeout_s=timeout_s,
            key=key or f"worker.{self._name}.request",
            cancel_previous=cancel_previous,
            drop_outdated=drop_outdated,
            on_success=on_success,
            on_error=on_error,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _do_start(self, post_start: Optional[Callable[[], None]] = None) -> None:
        reply = self._service.request_typed(StartWorker(worker_name=self._name))
        if isinstance(reply, ErrorMessage):
            raise RuntimeError(f"StartWorker failed for {self._name!r}: {reply.error}")
        if post_start is not None:
            post_start()

    def _stamp(self, cmd: Message) -> Message:
        return dataclasses.replace(cmd, consumer_id=self._name)
