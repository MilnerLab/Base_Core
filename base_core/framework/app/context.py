from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import logging

from base_core.framework.app.dispatcher import UiDispatcher
from base_core.framework.app.lifecycle import Lifecycle
from base_core.framework.events.event_bus import EventBus



@dataclass(frozen=True)
class AppContext:
    """
    Cross-cutting app resources (Qt-free):

    - config: runtime configuration/settings
    - log: application logger
    - events: pub/sub event bus
    - executor: background thread pool for blocking work
    - lifecycle: shutdown hooks
    - ui: UI dispatcher adapter (set by GUI app)
    """

    config: dict
    log: logging.Logger
    events: EventBus
    executor: ThreadPoolExecutor
    lifecycle: Lifecycle
    ui: Optional[UiDispatcher] = None
