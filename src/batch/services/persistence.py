"""
Service for persisting and loading batch processing state.
"""

import json
import logging
from pathlib import Path
from typing import Optional
from ...models.batch import BatchEvaluation


class BatchStatePersistenceService:
    """Service for persisting and loading batch processing state"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("BatchStatePersistenceService")
        self.state_dir = Path("batch_state")
        self.active_dir = self.state_dir / "active_batches"
        self.completed_dir = self.state_dir / "completed_batches"

        # Create directories if they don't exist
        self.active_dir.mkdir(parents=True, exist_ok=True)
        self.completed_dir.mkdir(parents=True, exist_ok=True)

    def save_batch_state(self, batch: BatchEvaluation) -> None:
        """Save batch state to file"""
        try:
            state_file = self.active_dir / f"{batch.batch_id}.json"
            with open(state_file, "w", encoding="utf-8") as f:
                # Convert enum values to their string representations
                data = batch.model_dump()
                # Convert enums to strings for JSON serialization
                if "status" in data:
                    data["status"] = (
                        data["status"].value
                        if hasattr(data["status"], "value")
                        else str(data["status"])
                    )
                json.dump(data, f, indent=2, default=str)
            self.logger.info(f"Batch state saved: {state_file}")
        except Exception as e:
            self.logger.error(f"Failed to save batch state: {str(e)}")
            raise

    def load_batch_state(self, batch_id: str) -> Optional[BatchEvaluation]:
        """Load batch state from file"""
        try:
            state_file = self.active_dir / f"{batch_id}.json"
            if not state_file.exists():
                return None

            with open(state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return BatchEvaluation(**data)
        except Exception as e:
            self.logger.error(f"Failed to load batch state: {str(e)}")
            return None

    def move_to_completed(self, batch_id: str) -> None:
        """Move batch state from active to completed"""
        try:
            active_file = self.active_dir / f"{batch_id}.json"
            completed_file = self.completed_dir / f"{batch_id}.json"

            if active_file.exists():
                active_file.rename(completed_file)
                self.logger.info(f"Batch moved to completed: {batch_id}")
        except Exception as e:
            self.logger.error(f"Failed to move batch to completed: {str(e)}")
            raise
