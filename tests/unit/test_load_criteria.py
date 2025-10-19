"""
Unit tests for load_criteria function.
Tests lines 1313-1326: load_criteria function
"""

import json
import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path

from src.utils import load_criteria


class TestLoadCriteria:
    """Test load_criteria function."""

    def test_load_criteria_success(self, mock_criteria_file):
        """Test successful criteria loading."""
        with patch("main.logger") as mock_logger:
            result = load_criteria()

            assert isinstance(result, dict)
            assert result["name"] == "テスト評価基準"
            assert result["version"] == "1.0"
            assert "criteria" in result
            assert len(result["criteria"]) == 2
            assert result["criteria"][0]["id"] == "article_5_1"
            assert result["criteria"][1]["id"] == "article_5_2"

    def test_load_criteria_file_not_found(self):
        """Test load_criteria with missing criteria file."""
        with patch("main.logger") as mock_logger:
            with patch(
                "builtins.open", side_effect=FileNotFoundError("File not found")
            ):
                with pytest.raises(ValueError, match="Criteria file not found"):
                    load_criteria()

                mock_logger.error.assert_called_with("Criteria file not found")

    def test_load_criteria_invalid_json(self):
        """Test load_criteria with invalid JSON."""
        with patch("main.logger") as mock_logger:
            with patch("builtins.open", mock_open(read_data="{ invalid json }")):
                with pytest.raises(ValueError, match="Invalid JSON in criteria file:"):
                    load_criteria()

                mock_logger.error.assert_called()
                # Verify the error message contains the JSON decode error
                error_call = mock_logger.error.call_args[0][0]
                assert "Invalid JSON in criteria file:" in error_call

    def test_load_criteria_empty_file(self):
        """Test load_criteria with empty file."""
        with patch("main.logger") as mock_logger:
            with patch("builtins.open", mock_open(read_data="")):
                with pytest.raises(ValueError, match="Invalid JSON in criteria file:"):
                    load_criteria()

                mock_logger.error.assert_called()

    def test_load_criteria_partial_json(self):
        """Test load_criteria with partial JSON (incomplete)."""
        with patch("main.logger") as mock_logger:
            with patch("builtins.open", mock_open(read_data='{"name": "Test"')):
                with pytest.raises(ValueError, match="Invalid JSON in criteria file:"):
                    load_criteria()

                mock_logger.error.assert_called()

    def test_load_criteria_unicode_content(self):
        """Test load_criteria with Unicode content."""
        unicode_content = {
            "name": "テスト評価基準",
            "version": "1.0",
            "criteria": [
                {
                    "id": "article_5_1",
                    "name": "個人情報保護",
                    "article": "第5条第1号",
                    "evaluation_steps": [
                        "個人に関する情報か",
                        "特定の個人を識別できるか",
                    ],
                }
            ],
        }

        with patch("main.logger") as mock_logger:
            with patch(
                "builtins.open",
                mock_open(read_data=json.dumps(unicode_content, ensure_ascii=False)),
            ):
                result = load_criteria()

                assert result["name"] == "テスト評価基準"
                assert result["criteria"][0]["name"] == "個人情報保護"
                assert "個人に関する情報か" in result["criteria"][0]["evaluation_steps"]

    def test_load_criteria_large_file(self):
        """Test load_criteria with large criteria file."""
        large_criteria = {
            "name": "Large Test Criteria",
            "version": "1.0",
            "criteria": [],
        }

        # Create a large number of criteria
        for i in range(100):
            large_criteria["criteria"].append(
                {
                    "id": f"article_5_{i}",
                    "name": f"テスト基準{i}",
                    "article": f"第5条第{i}号",
                    "evaluation_steps": [f"ステップ{i}_1", f"ステップ{i}_2"],
                }
            )

        with patch("main.logger") as mock_logger:
            with patch(
                "builtins.open",
                mock_open(read_data=json.dumps(large_criteria, ensure_ascii=False)),
            ):
                result = load_criteria()

                assert len(result["criteria"]) == 100
                assert result["criteria"][0]["id"] == "article_5_0"
                assert result["criteria"][99]["id"] == "article_5_99"

    def test_load_criteria_missing_required_fields(self):
        """Test load_criteria with missing required fields."""
        incomplete_criteria = {
            "name": "Incomplete Criteria"
            # Missing version and criteria fields
        }

        with patch("main.logger") as mock_logger:
            with patch(
                "builtins.open", mock_open(read_data=json.dumps(incomplete_criteria))
            ):
                result = load_criteria()

                # Should still load successfully, just with missing fields
                assert result["name"] == "Incomplete Criteria"
                assert "version" not in result
                assert "criteria" not in result

    def test_load_criteria_nested_structure(self):
        """Test load_criteria with nested structure."""
        nested_criteria = {
            "name": "Nested Test Criteria",
            "version": "1.0",
            "criteria": [
                {
                    "id": "article_5_1",
                    "name": "個人情報保護",
                    "article": "第5条第1号",
                    "evaluation_steps": ["ステップ1", "ステップ2"],
                    "scoring_interpretation": {
                        "1": "強く不開示",
                        "2": "不開示の可能性が高い",
                        "3": "不明確",
                        "4": "開示の可能性が高い",
                        "5": "明確に開示",
                    },
                    "examples": {
                        "disclosure": "公務員の職務遂行に関する情報",
                        "non_disclosure": "一般市民の個人情報",
                    },
                }
            ],
        }

        with patch("main.logger") as mock_logger:
            with patch(
                "builtins.open",
                mock_open(read_data=json.dumps(nested_criteria, ensure_ascii=False)),
            ):
                result = load_criteria()

                assert result["name"] == "Nested Test Criteria"
                assert len(result["criteria"]) == 1
                criterion = result["criteria"][0]
                assert criterion["id"] == "article_5_1"
                assert "scoring_interpretation" in criterion
                assert "examples" in criterion
                assert criterion["scoring_interpretation"]["1"] == "強く不開示"
                assert (
                    criterion["examples"]["disclosure"]
                    == "公務員の職務遂行に関する情報"
                )

    def test_load_criteria_file_encoding(self):
        """Test load_criteria with different file encodings."""
        criteria_content = {
            "name": "エンコーディングテスト",
            "version": "1.0",
            "criteria": [],
        }

        with patch("main.logger") as mock_logger:
            with patch(
                "builtins.open",
                mock_open(read_data=json.dumps(criteria_content, ensure_ascii=False)),
            ):
                result = load_criteria()

                assert result["name"] == "エンコーディングテスト"

    def test_load_criteria_permission_error(self):
        """Test load_criteria with permission error."""
        with patch("main.logger") as mock_logger:
            with patch(
                "builtins.open", side_effect=PermissionError("Permission denied")
            ):
                with pytest.raises(PermissionError, match="Permission denied"):
                    load_criteria()

    def test_load_criteria_io_error(self):
        """Test load_criteria with IO error."""
        with patch("main.logger") as mock_logger:
            with patch("builtins.open", side_effect=OSError("IO Error")):
                with pytest.raises(OSError, match="IO Error"):
                    load_criteria()

    def test_load_criteria_memory_error(self):
        """Test load_criteria with memory error (very large file)."""
        with patch("main.logger") as mock_logger:
            with patch("builtins.open", side_effect=MemoryError("Out of memory")):
                with pytest.raises(MemoryError, match="Out of memory"):
                    load_criteria()

    def test_load_criteria_corrupted_file(self):
        """Test load_criteria with corrupted file content."""
        with patch("main.logger") as mock_logger:
            with patch(
                "builtins.open",
                mock_open(read_data="corrupted binary data \x00\x01\x02"),
            ):
                with pytest.raises(ValueError, match="Invalid JSON in criteria file:"):
                    load_criteria()

                mock_logger.error.assert_called()

    def test_load_criteria_special_characters(self):
        """Test load_criteria with special characters in content."""
        special_criteria = {
            "name": "Special Characters Test",
            "version": "1.0",
            "description": "Test with special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?",
            "criteria": [
                {
                    "id": "article_5_1",
                    "name": "テスト基準（特殊文字）",
                    "article": "第5条第1号",
                    "evaluation_steps": ["ステップ1: 確認", "ステップ2: 検証"],
                }
            ],
        }

        with patch("main.logger") as mock_logger:
            with patch(
                "builtins.open",
                mock_open(read_data=json.dumps(special_criteria, ensure_ascii=False)),
            ):
                result = load_criteria()

                assert result["name"] == "Special Characters Test"
                assert "!@#$%^&*()_+-=[]{}|;':\",./<>?" in result["description"]
                assert result["criteria"][0]["name"] == "テスト基準（特殊文字）"
                assert "ステップ1: 確認" in result["criteria"][0]["evaluation_steps"]
