import logging

logger = logging.getLogger('homage_fl')
logger.handlers.clear()

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter(fmt='[%(asctime)s] %(message)s', datefmt='%H:%M:%S'))

logger.addHandler(stream_handler)


def get_logger():
    logger = logging.getLogger('homage_fl')
    logger.handlers.clear()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(fmt='[%(asctime)s] %(message)s', datefmt='%H:%M:%S'))

    logger.addHandler(stream_handler)
    return logger
