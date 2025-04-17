import logging

logger = logging


class BinaryLogFilter(logging.Filter):
    def filter(self, record):
        return not (record.getMessage().startswith('< BINARY') or '< BINARY' in record.getMessage())


def setLogger(level: str):
    global logger

    levels = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
    }

    logger.basicConfig(format='%(asctime)s - %(message)s', level=levels[level])
    for handler in logger.getLogger().handlers:
        handler.addFilter(BinaryLogFilter())
