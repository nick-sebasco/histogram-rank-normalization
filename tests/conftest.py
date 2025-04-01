import pytest
from pathlib import Path
import logging

# ---------------------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------------------
@pytest.fixture
def output_score_path() -> str:
    return "histogram-rank-norm-test-data/T_SAVED_SCORES"


@pytest.fixture
def image_dir() -> str:
    return "histogram-rank-norm-test-data/Images"


@pytest.fixture
def combined_score_path(tmp_path) -> Path:
    combined_score_path = tmp_path / "combined_scores"
    combined_score_path.mkdir()
    logging.info("Combined scores saved here: %s", combined_score_path)
    return combined_score_path