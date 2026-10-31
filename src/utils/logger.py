"""logger configuration"""

import copy
import datetime
import logging
from pathlib import Path

################################################################
# PATH CONSTANTS
################################################################
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# timestamp (date + time)
TIMESTAMP: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# log directory
LOG_DIR: Path = PROJECT_ROOT / "log" / TIMESTAMP
PROJECT_LOGGER_NAME = "project"


################################################################
# LOGGER
################################################################
class ColoredStreamHandler(logging.StreamHandler):
    """colored stream handler (only colorize when output is a TTY)"""

    cmap = {
        "TRACE": "[TRACE]",
        "DEBUG": "\x1b[0;36mDEBUG\x1b[0m",
        "INFO": "\x1b[0;32mINFO\x1b[0m",
        "WARNING": "\x1b[0;33mWARN\x1b[0m",
        "WARN": "\x1b[0;33mWARN\x1b[0m",
        "ERROR": "\x1b[0;31mERROR\x1b[0m",
        "ALERT": "\x1b[0;37;41mALERT\x1b[0m",
        "CRITICAL": "\x1b[0;37;41mCRITICAL\x1b[0m",
    }

    cmap_plain = {
        "TRACE": "TRACE",
        "DEBUG": "DEBUG",
        "INFO": "INFO",
        "WARNING": "WARN",
        "WARN": "WARN",
        "ERROR": "ERROR",
        "ALERT": "ALERT",
        "CRITICAL": "CRITICAL",
    }

    def emit(self, record: logging.LogRecord) -> None:
        """output log record"""
        record = copy.deepcopy(record)
        # Only use colors if output is a TTY (terminal)
        if self.stream.isatty():
            record.levelname = self.cmap.get(record.levelname, record.levelname)
        else:
            record.levelname = self.cmap_plain.get(record.levelname, record.levelname)
        super().emit(record)


def _setup_logger() -> logging.Logger:
    """setup logger"""
    if not LOG_DIR.exists():
        LOG_DIR.mkdir(parents=True)

    logger = logging.getLogger(PROJECT_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d %(message)s",
        datefmt="%Y/%m/%d-%H:%M:%S",
    )

    stream_handler = ColoredStreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(LOG_DIR / "all.log", mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    warning_handler = logging.FileHandler(
        LOG_DIR / "warning.log", mode="a", encoding="utf-8"
    )
    warning_handler.setLevel(logging.WARN)
    warning_handler.setFormatter(formatter)
    logger.addHandler(warning_handler)

    return logger


logger = _setup_logger()
