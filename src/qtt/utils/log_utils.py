import logging
from typing import Optional


def verbosity2loglevel(verbosity):
    """Translates verbosity to logging level. Suppresses warnings if verbosity = 0."""
    if verbosity <= 0:  # only errors
        # print("Caution: all warnings suppressed")
        log_level = 40
    elif verbosity == 1:  # only warnings and critical print statements
        log_level = 25
    elif verbosity == 2:  # key print statements which should be shown by default
        log_level = 20
    elif verbosity == 3:  # more-detailed printing
        log_level = 15
    else:
        log_level = 10  # print everything (ie. debug mode)
    return log_level


def set_logger_verbosity(verbosity: int, logger=None):
    if logger is None:
        logger = logging.root
    if verbosity < 0:
        verbosity = 0
    elif verbosity > 4:
        verbosity = 4
    logger.setLevel(verbosity2loglevel(verbosity))


def add_log_to_file(
    file_path: str,
    logger: Optional[logging.Logger] = None,
    fmt: Optional[str] = None,
    datefmt: Optional[str] = None,
):
    """
    Add a FileHandler to the logger so that it can log to a file

    Parameters
    ----------
    file_path: str
        File path to save the log
    logger: Optional[logging.Logger], default = None
        The log to add FileHandler.
        If not provided, will add to the default AG logger, `logging.getLogger('autogluon')`
    """
    if logger is None:
        logger = logging.root
    fh = logging.FileHandler(file_path)
    if fmt is None:
        fmt = "%(asctime)s - %(name)16s: [%(levelname)s] %(message)s"
    if datefmt is None:
        datefmt = "%y.%m.%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

# def _add_stream_handler():
#     if not any(isinstance(h, logging.StreamHandler) for h in _logger_.handlers):
#         stream_handler = logging.StreamHandler()
#         formatter = logging.Formatter("%(message)s")
#         stream_handler.setFormatter(formatter)
#         _logger_.addHandler(stream_handler)
#         _logger_.propagate = False

def setup_default_logging(
    default_level=logging.INFO,
    fmt: Optional[str] = None,
    datefmt: Optional[str] = None,
):
    if fmt is None:
        fmt = "%(asctime)s - %(name)s: [%(levelname)s] %(message)s"
    if datefmt is None:
        datefmt = "%y.%m.%d %H:%M:%S"
    logging.basicConfig(format=fmt, datefmt=datefmt, level=default_level)
    # formatter = logging.Formatter(fmt, datefmt)
    # sh = logging.StreamHandler()
    # sh.setFormatter(formatter)
    # logging.root.addHandler(sh)
    # logging.root.setLevel(default_level)
