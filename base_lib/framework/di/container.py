from __future__ import annotations

from typing import Any, Callable, Dict, Optional, TypeVar

T = TypeVar("T")
Provider = Callable[["Container"], Any]


class Container:
    """
    Minimal DI container with two lifetimes:
    - singleton: created once (lazy) and cached
    - factory: created on every get()

    Keys can be types (recommended), strings, enums, etc.
    """

    def __init__(self) -> None:
        self._singleton_providers: Dict[Any, Provider] = {}
        self._singletons: Dict[Any, Any] = {}
        self._factory_providers: Dict[Any, Provider] = {}

    # --- registration -----------------------------------------------------

    def register_singleton(self, key: Any, provider: Callable[["Container"], T]) -> None:
        if key in self._factory_providers:
            raise KeyError(f"{key!r} already registered as factory.")
        self._singleton_providers[key] = provider

    def register_instance(self, key: Any, instance: T) -> None:
        if key in self._factory_providers:
            raise KeyError(f"{key!r} already registered as factory.")
        self._singletons[key] = instance

    def register_factory(self, key: Any, factory: Callable[["Container"], T]) -> None:
        if key in self._singleton_providers or key in self._singletons:
            raise KeyError(f"{key!r} already registered as singleton.")
        self._factory_providers[key] = factory

    # --- resolution -------------------------------------------------------

    def get(self, key: Any) -> Any:
        if key in self._singletons:
            return self._singletons[key]

        if key in self._singleton_providers:
            instance = self._singleton_providers[key](self)
            self._singletons[key] = instance
            return instance

        if key in self._factory_providers:
            return self._factory_providers[key](self)

        raise KeyError(f"No provider registered for {key!r}.")

    def try_get(self, key: Any) -> Optional[Any]:
        try:
            return self.get(key)
        except KeyError:
            return None

    def is_registered(self, key: Any) -> bool:
        return (
            key in self._singletons
            or key in self._singleton_providers
            or key in self._factory_providers
        )
