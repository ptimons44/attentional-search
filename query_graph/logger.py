import logging
from celery.signals import after_setup_logger

def setup_logger(logger_name='query_graph', log_filename='query_graph.log'):
    # Create or retrieve the logger
    logger = logging.getLogger(logger_name)

    # Check if the logger already has handlers attached to it. If it does, we can return early.
    if logger.hasHandlers():
        return logger

    # Set the log level
    logger.setLevel(logging.DEBUG)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a file handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

@after_setup_logger.connect
def on_after_setup_logger(logger, **kwargs):
    custom_logger = setup_logger('celery', 'celery.log')
    for handler in custom_logger.handlers:
        logger.addHandler(handler)

# Setting up the celery logger right away
logger = setup_logger('celery', 'celery.log')

if __name__ == "__main__":
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")