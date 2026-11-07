"""Centralized logging for the project.

All modules should use:
    from logger import get_logger
    logger = get_logger(__name__)
"""

import logging

LOGGER_NAME = "BoundarySearch"

_initialized = False


def setup_logging(verbose: bool = False, log_file: str = None) -> None:
    """Initialize the root project logger. Call once from main.py."""
    global _initialized
    if _initialized:
        return
    _initialized = True

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    # Prevent messages from bubbling up to the root logger — some imported
    # library may call logging.basicConfig() and add a default StreamHandler
    # to root, which would re-emit every message in default format.
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file:
        from pathlib import Path
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def add_file_handler(log_file: str) -> None:
    """Add a file handler after setup_logging() has been called."""
    from pathlib import Path
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    formatter = logging.Formatter(
        "%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def get_logger(name: str = None) -> logging.Logger:
    """Get a child logger under the project root logger.

    Args:
        name: Logger name suffix (typically __name__).
              If None, returns the root project logger.
    """
    if name:
        return logging.getLogger(f"{LOGGER_NAME}.{name}")
    return logging.getLogger(LOGGER_NAME)
