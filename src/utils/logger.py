"""
Logging utilities
"""

import logging
import sys
from typing import Optional


def setup_logger(name: str = 'scalegnn', level: int = logging.INFO,
                rank: Optional[int] = None) -> logging.Logger:
    """
    Set up logger with consistent formatting.

    Args:
        name: Logger name
        level: Logging level
        rank: Process rank (for distributed training)

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Format
    if rank is not None:
        fmt = f'[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
    else:
        fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    formatter = logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger
