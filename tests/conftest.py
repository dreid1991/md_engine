import pytest
'''
    This file contains directory-specific hook implementations

    Here, we add support for tests that would take a very long time, and are
    thus better suited to be run on an external machine (e.g. appveyor),
    while shorter tests are, by default, always run, allowing testing to proceed
    on local machines.

'''
def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true",
            default=False, help="run slow tests")

def pytest_collection_modifyitems(config,items):
    if config.getoption("--run-slow"):
        # --run-slow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)

