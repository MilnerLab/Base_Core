from dataclasses import dataclass
import threading


@dataclass(frozen=True)
class StreamHandle:
    stop_event: threading.Event
    done_event: threading.Event  # set when producer loop exits (including finally)

    def stop(self) -> None:
        self.stop_event.set()

    def wait(self, timeout: float = 5.0) -> None:
        """Block until the producer has finished (use before closing hardware)."""
        self.done_event.wait(timeout=timeout)
