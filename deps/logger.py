import logging

logger = logging.getLogger('homage_fl')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter(fmt='[%(asctime)s]\n%(message)s', datefmt='%H:%M:%S'))
logger.addHandler(stream_handler)
