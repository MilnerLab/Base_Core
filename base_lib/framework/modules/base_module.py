from abc import ABC, abstractmethod
from typing import Sequence

from base_lib.framework.app.context import AppContext
from base_lib.framework.di.container import Container


class BaseModule(ABC):
    """
    Module contract.

    register(): register providers/factories (no heavy work)
    start(): optional, after ALL modules registered (safe to resolve deps)
    stop(): optional, during shutdown
    """

    name: str = "unnamed"
    requires: Sequence[str] = ()

    @abstractmethod
    def register(self, c: Container, ctx: AppContext) -> None:
        raise NotImplementedError

    def start(self, c: Container, ctx: AppContext) -> None:
        return None

    def stop(self, c: Container, ctx: AppContext) -> None:
        return None
