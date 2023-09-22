import pytest
from tradingenv.contracts import AbstractContract
from datetime import datetime


@pytest.fixture(autouse=True)
def reset_global_attributes():
    AbstractContract.now = datetime.min  # before the test
    yield  # run the test
    AbstractContract.now = datetime.min  # after the test
