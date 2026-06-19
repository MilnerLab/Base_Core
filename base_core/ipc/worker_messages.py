from __future__ import annotations

from dataclasses import dataclass

from base_core.ipc.codec import register
from base_core.ipc.message import OKReply, Request


@register
@dataclass(frozen=True)
class StartWorker(Request[OKReply]):
    worker_id: str = ""


@register
@dataclass(frozen=True)
class PauseWorker(Request[OKReply]):
    worker_id: str = ""


@register
@dataclass(frozen=True)
class ResetWorker(Request[OKReply]):
    worker_id: str = ""
