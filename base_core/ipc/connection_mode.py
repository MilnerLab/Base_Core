"""What a device worker is talking to: the hardware, or a mock standing in for it."""
from __future__ import annotations

from enum import Enum


class ConnectionMode(str, Enum):
    """Both the request ("what should I connect to?") and the outcome ("what did I get?").

    One enum for both directions on purpose: the interesting case is precisely when
    they differ. A ``DEVICE`` request that comes back ``MOCK`` is a demotion, and that
    gap is the thing worth telling the operator about.

    The ``str`` mixin is load-bearing, not decoration. The IPC codec serializes via
    ``to_primitive``, which accepts str/int/float/bool and raises ``TypeError`` on a
    plain ``Enum``; and its ``_reconstruct`` has no enum branch, so the value arrives
    on the far side as a bare string. Mixing in ``str`` makes both directions work,
    because the decoded string compares and hashes equal to the member.
    """

    #: Try the hardware; fall back to the mock if it cannot be reached.
    DEVICE = "device"
    #: Never touch the hardware, even if it is present and working.
    MOCK = "mock"
    #: Not a device worker. Start normally; there is no mock and nothing to fall back to.
    NONE = "none"
