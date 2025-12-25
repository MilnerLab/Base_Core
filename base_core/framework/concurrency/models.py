from concurrent.futures import Future
from dataclasses import dataclass
import threading
from typing import Optional

@dataclass(frozen=True)
class StreamHandle:
    stop_event: threading.Event
    future: Future[None]

    def stop(self) -> None:
        self.stop_event.set()
