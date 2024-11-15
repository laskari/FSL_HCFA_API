import logging
import os

def setup_logger(log_file: str):
    """
    Set up the logger to write logs to the specified file.

    Args:
    - log_file (str): Path to the log file.
    
    Returns:
    - logger: Configured logger instance.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Create and configure logger
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)

    # Create a file handler for logging
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    if not logger.hasHandlers():
        logger.addHandler(file_handler)

    return logger


def log_message(logger, message: str, level: str = "INFO"):
    """
    Log a message using the provided logger at the specified log level.

    Args:
    - logger: Logger instance created by setup_logger.
    - message (str): The message to log.
    - level (str): The log level ('INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL').
    """
    if level == "DEBUG":
        logger.debug(message)
    elif level == "INFO":
        logger.info(message)
    elif level == "WARNING":
        logger.warning(message)
    elif level == "ERROR":
        logger.error(message)
    elif level == "CRITICAL":
        logger.critical(message)
    else:
        logger.info(message)
