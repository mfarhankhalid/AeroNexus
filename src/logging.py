"""A tiny logging helper built on the standard ``logging`` module.

Use ``get_logger(__name__)`` anywhere in the project to get a logger that
prints messages to the console. The logger name usually matches the module
name, so log lines show you where the message came from.
"""

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger with the given name.

    The logger writes to the console (stderr) and avoids duplicate handlers
    if the same name is requested more than once.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger