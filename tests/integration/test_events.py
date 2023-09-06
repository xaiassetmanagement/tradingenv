from tradingenv.contracts import Rate, ETF
from tradingenv.events import EventNBBO
from datetime import datetime
import pytest


class TestEventNBBO:
    def test_rate_does_not_raise_if_lower_than_25_pct(self):
        EventNBBO(datetime(2019, 1, 1), Rate('EONIA'), 0.24, 0.24)

    def test_rate_raise_if_greater_than_25_pct(self):
        with pytest.raises(ValueError):
            EventNBBO(datetime(2019, 1, 1), Rate('EONIA'), 0.25, 0.25)

    def test_raise_if_non_rate_is_negative(self):
        with pytest.raises(ValueError):
            EventNBBO(datetime(2019, 1, 1), ETF('SPY'), -1, -1)
