import logging
import os
import sys

from from_root import from_root
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

log_dir = 'logs'

# ensure the logs directory exists at the project root
logs_dir_path = os.path.join(from_root(), log_dir)
logs_path = os.path.join(logs_dir_path, LOG_FILE)
os.makedirs(logs_dir_path, exist_ok=True)

# attempt to configure file logging; first verify we can open the file.
log_format = "[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s"
try:
    # ensure we can open the log file for appending before configuring logging
    with open(logs_path, "a", encoding="utf-8"):
        pass
    logging.basicConfig(
        filename=logs_path,
        format=log_format,
        level=logging.DEBUG,
    )
except OSError as e:
    # If writing to the file fails (permissions, missing dir, etc.),
    # fall back to console logging so the application still runs.
    logging.basicConfig(
        stream=sys.stdout,
        format=log_format,
        level=logging.DEBUG,
    )
    logger = logging.getLogger(__name__)
    logger.warning(f"Unable to write log file at '{logs_path}'; using stdout instead. Error: {e}")