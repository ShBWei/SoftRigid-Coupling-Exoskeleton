"""Centralised logging configuration.

Usage
-----
    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("training started")
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_dir: str | None = None,
    filename: str | None = None,
) -> logging.Logger:
    """Return a logger with a StreamHandler (and optional FileHandler).

    Parameters
    ----------
    name     : logger name (typically ``__name__``)
    level    : logging level (default INFO)
    log_dir  : if provided, also write logs to a file in this directory
    filename : override log filename (default: ``<name>_<timestamp>.log``)
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s – %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # Optional file handler
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = name.replace(".", "_")
            filename = f"{safe_name}_{ts}.log"
        fh = logging.FileHandler(os.path.join(log_dir, filename))
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger
