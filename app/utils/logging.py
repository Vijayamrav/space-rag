"""
Centralised logging configuration.
Call `setup_logging()` once at app startup.
"""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
