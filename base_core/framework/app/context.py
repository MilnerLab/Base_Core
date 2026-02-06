from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from operator import imod
from typing import Optional
import logging

from base_core.framework.lifecycle.cleanup_collection import CleanupCollection
from base_core.framework.events.event_bus import EventBus
from base_core.framework.app.enums import AppStatus



@dataclass(frozen=True)
class AppContext:
    """
    Cross-cutting app resources (Qt-free):

    - config: runtime configuration/settings
    - log: application logger
    - events: pub/sub event bus
    - lifecycle: shutdown hooks
    """

    config: dict
    status: AppStatus
    log: logging.Logger
    event_bus: EventBus
    lifecycle: CleanupCollection
