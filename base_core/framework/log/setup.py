from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


_CONSOLE_MARKER = "_base_core_console_logging"


def enable_console_logging(level: int = logging.INFO) -> Optional[logging.Handler]:
    """Attach a StreamHandler to the ROOT logger so *every* module logger
    (those created with ``logging.getLogger(__name__)``) prints to the terminal.

    Why this is needed: ``setup_logging()`` configures only a single *named*
    logger and sets ``propagate=False`` on it. Module-level loggers under
    ``app_apps`` / ``base_core`` / ``control_readout`` are not children of that
    named logger, so they have no handler and their INFO/DEBUG records are
    silently dropped by logging's last-resort handler (WARNING+ only). Nothing
    those modules log with ``log.info(...)`` ever reaches the terminal.

    Call this once per process — the main process AND every subprocess, since
    subprocesses are separate Python interpreters with their own logging config
    (subprocesses inherit the parent's stdout/stderr, so their records show up
    interleaved in the same terminal). ``%(process)d`` in the format lets you
    tell the three processes (main / control_readout / phase_control) apart.

    Idempotent: a second call in the same process is a no-op.
    """
    root = logging.getLogger()
    for handler in root.handlers:
        if getattr(handler, _CONSOLE_MARKER, False):
            return handler  # already enabled in this process

    root.setLevel(level)
    fmt = logging.Formatter(
        fmt="%(asctime)s | pid=%(process)d | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)
    setattr(sh, _CONSOLE_MARKER, True)
    root.addHandler(sh)
    return sh


def setup_logging(
    name: str,
    *,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """
    Configure and return a logger with stream handler and optional rotating file handler.
    Idempotent: does nothing if the logger already has handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            log_file,
            maxBytes=2_000_000,
            backupCount=5,
            encoding="utf-8",
        )
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
