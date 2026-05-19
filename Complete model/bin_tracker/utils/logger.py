"""
Centralized logging setup.
"""

import logging
from pathlib import Path


def setup_logger(name: str = "bin_tracker", config: dict | None = None) -> logging.Logger:
    """Create a logger with console + optional file output."""
    log_cfg = config.get("logging", {}) if config else {}
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
                            datefmt="%H:%M:%S")
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler (optional)
    log_file = log_cfg.get("log_file")
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
