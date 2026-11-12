import os
from pathlib import Path

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: needs llama.cpp build and DORY_GGUF")
    config.addinivalue_line("markers", "network: needs network access")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("DORY_GGUF"):
        return
    skip = pytest.mark.skip(reason="DORY_GGUF unset")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def gguf():
    return Path(os.environ["DORY_GGUF"])
