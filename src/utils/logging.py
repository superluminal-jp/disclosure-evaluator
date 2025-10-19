"""
Logging configuration utilities.
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any


def setup_logging(config_manager) -> logging.Logger:
    """Setup structured logging from configuration"""
    # Create logs and outputs directories if they don't exist
    logs_dir = config_manager.get("logging.directories.logs", "logs")
    outputs_dir = config_manager.get("logging.directories.outputs", "outputs")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    # Generate timestamped log filename
    timestamp = datetime.now().strftime(
        config_manager.get("output.timestamp_format", "%Y%m%d_%H%M%S")
    )
    log_filename = f"{logs_dir}/evaluation_{timestamp}.log"

    # Configure structured logging from config
    logging_config = config_manager.get_logging_config()
    logging.basicConfig(
        level=getattr(logging, logging_config.get("level", "INFO")),
        format=logging_config.get(
            "format",
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s", "correlation_id": "%(name)s"}',
        ),
        handlers=[
            logging.FileHandler(
                log_filename, encoding=logging_config.get("file_encoding", "utf-8")
            ),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)
