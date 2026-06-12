import dataclasses
import uuid
from typing import Generic, TypeVar

TReply = TypeVar("TReply")


@dataclasses.dataclass(frozen=True)
class Message:
    id: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))


@dataclasses.dataclass(frozen=True)
class Request(Message, Generic[TReply]):
    """Message that expects a reply of type TReply. Register on_reply via ServicePipelineConnector.send()."""
    pass


@dataclasses.dataclass(frozen=True)
class Reply(Message):
    """Base for all reply messages. Set request_id to the originating Request.id."""
    request_id: str = ""


@dataclasses.dataclass(frozen=True)
class OKReply(Reply):
    pass


@dataclasses.dataclass(frozen=True)
class ErrorReply(Reply):
    error: str = ""
