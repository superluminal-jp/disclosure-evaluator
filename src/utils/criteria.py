"""
Criteria loading utilities.
"""

import json
import logging

logger = logging.getLogger(__name__)


def load_criteria():
    """Load criteria configuration from JSON file with error handling"""
    try:
        with open(
            "criteria/disclosure_evaluation_criteria.json", "r", encoding="utf-8"
        ) as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("Criteria file not found")
        raise ValueError("Criteria file not found")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in criteria file: {e}")
        raise ValueError(f"Invalid JSON in criteria file: {e}")
