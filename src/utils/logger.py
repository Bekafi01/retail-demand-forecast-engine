"""Structured logging configuration."""

import logging
import sys
from typing import Optional


def get_logger(name: str = "retail_forecast", level: Optional[int] = None) -> logging.Logger:
    """Get a configured logger instance with formatted console output."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level or logging.INFO)
    return logger
