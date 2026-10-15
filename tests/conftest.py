import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bearing_audio.data import list_recordings  # noqa: E402
from bearing_audio.features import FeatureTable  # noqa: E402


@pytest.fixture(scope="session")
def root():
    return ROOT


@pytest.fixture(scope="session")
def table():
    """Window features for all 60 recordings (about 2 minutes the first time, then cached in artifacts/)."""
    return FeatureTable(list_recordings(), cache=ROOT / "artifacts" / "features.npz")
