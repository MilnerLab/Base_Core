from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ServiceStatus:
    """Published by a service when its running state or detail changes."""
    name: str
    running: bool
    detail: str = ""
