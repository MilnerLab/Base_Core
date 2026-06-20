from dataclasses import dataclass
import threading


@dataclass(frozen=True)
class StreamHandle:
    stop_event: threading.Event
    done_event: threading.Event  # set after consumer drains all in-flight items

    def stop(self) -> None:
        self.stop_event.set()

    def wait(self, timeout: float = 5.0) -> None:
        """Block until the consumer has finished all in-flight items (use before releasing hardware or shared-memory slots)."""
        self.done_event.wait(timeout=timeout)
