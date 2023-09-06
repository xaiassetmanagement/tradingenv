from tradingenv.broker.rebalancing import Rebalancing
from tradingenv.broker.allocation import Weights, NrContracts
from tradingenv.contracts import ETF
from collections import namedtuple
from datetime import datetime
_TestSubtraction = namedtuple('TestSubtraction', ['right', 'expected'])


class TestRebalancing:
    def test_default_initialization(self):
        rebalancing = Rebalancing()
        assert rebalancing.allocation == dict()
        assert isinstance(rebalancing.allocation, Weights)
        assert rebalancing.absolute is True
        assert rebalancing.fractional is True
        assert rebalancing.margin == 0.0
        assert (rebalancing.time - datetime.now()).total_seconds() < 5

    def test_custom_initialization(self):
        now = datetime.now()
        rebalancing = Rebalancing(
            [ETF("SPY")], [0.8], 'nr-contracts', False, False, 0.02, now
        )
        assert rebalancing.allocation == {ETF("SPY"): 0.8}
        assert isinstance(rebalancing.allocation, NrContracts)
        assert rebalancing.absolute is False
        assert rebalancing.fractional is False
        assert rebalancing.margin == 0.02
        assert rebalancing.time == now
