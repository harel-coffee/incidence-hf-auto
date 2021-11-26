import logging

logger = logging.getLogger('homage_fl')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(
    logging.Formatter(fmt='[%(asctime)s] %(module)s:%(lineno)d\n%(message)s\n', datefmt='%H:%M:%S')
)
logger.addHandler(stream_handler)
