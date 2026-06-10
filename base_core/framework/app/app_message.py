from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MessageLevel(Enum):
    INFO    = "info"
    WARNING = "warning"
    ERROR   = "error"


@dataclass(frozen=True)
class AppMessage:
    text: str
    level: MessageLevel = MessageLevel.INFO
