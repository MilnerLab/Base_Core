from dataclasses import dataclass
from typing import Mapping, Sequence

from base_core.framework.app.context import AppContext
from base_core.framework.di.container import Container
from base_core.framework.modules.base_module import BaseModule
from base_core.framework.modules.error import ModuleError


@dataclass
class ModuleManager:
    """
    - orders modules by `requires`
    - calls register() for all
    - then start() for all
    - hooks stop() into ctx.lifecycle (reverse order)
    """
    modules: Sequence[BaseModule]

    def _index(self) -> Mapping[str, BaseModule]:
        by_name: dict[str, BaseModule] = {}
        for m in self.modules:
            if m.name in by_name:
                raise ModuleError(f"Duplicate module name: {m.name!r}")
            by_name[m.name] = m
        return by_name

    def _order(self) -> list[BaseModule]:
        by_name = self._index()

        for m in self.modules:
            for dep in m.requires:
                if dep not in by_name:
                    raise ModuleError(f"Module {m.name!r} requires missing module {dep!r}")

        incoming: dict[str, int] = {m.name: 0 for m in self.modules}
        outgoing: dict[str, list[str]] = {m.name: [] for m in self.modules}

        for m in self.modules:
            for dep in m.requires:
                incoming[m.name] += 1
                outgoing[dep].append(m.name)

        queue = [name for name, n in incoming.items() if n == 0]
        ordered_names: list[str] = []

        while queue:
            n = queue.pop()
            ordered_names.append(n)
            for nxt in outgoing[n]:
                incoming[nxt] -= 1
                if incoming[nxt] == 0:
                    queue.append(nxt)

        if len(ordered_names) != len(self.modules):
            remaining = [name for name, n in incoming.items() if n > 0]
            raise ModuleError(f"Dependency cycle among: {remaining}")

        return [by_name[name] for name in ordered_names]

    def bootstrap(self, c: Container, ctx: AppContext) -> None:
        ordered = self._order()

        for m in ordered:
            ctx.log.info("Registering module: %s", m.name)
            m.register(c, ctx)

        for m in ordered:
            ctx.log.info("Starting module: %s", m.name)
            m.start(c, ctx)

        def _shutdown() -> None:
            for m in reversed(ordered):
                try:
                    ctx.log.info("Stopping module: %s", m.name)
                    m.stop(c, ctx)
                except Exception:
                    ctx.log.exception("Error stopping module: %s", m.name)

        ctx.lifecycle.add_shutdown_hook(_shutdown)