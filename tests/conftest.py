# content of conftest.py

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )

    parser.addoption(
        "--runcuda", action="store_true", default=False, help="run cuda dependent tests"
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "cuda: marks tests to require cuda")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    if not config.getoption("--runcuda"):
        skip_cuda = pytest.mark.skip(reason="need --runscuda option to run")

        for item in items:
            if "cuda" in item.keywords:
                item.add_marker(skip_cuda)
