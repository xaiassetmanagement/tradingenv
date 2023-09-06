from tradingenv.broker.fees import IBrokerFees
from tradingenv.broker.trade import Trade
from tradingenv.contracts import Cash, ETF, ES, ZN
from datetime import datetime, timedelta
import numpy as np
import pytest


class TestTrade:
    def test_initialization(self):
        now = datetime.now()
        spy = ETF("SPY")
        trade = Trade(now, spy, 1.0, 2.0, 3.0)
        assert trade.time == now
        assert trade.contract == spy
        assert trade.quantity == 1.0
        assert trade.bid_price == 2.0
        assert trade.ask_price == 3.0
        assert trade.notional == trade.quantity * spy.multiplier * trade.ask_price

    def test_notional_with_futures(self):
        es = ES(2019, 6)
        trade = Trade(datetime.now(), es, 1.0, 2.0, 3.0)
        assert trade.notional == trade.quantity * es.multiplier * trade.ask_price

    def test_raises_if_bid_price_is_nan(self):
        now = datetime.now()
        with pytest.raises(ValueError):
            Trade(now, ETF("SPY"), 1.0, np.nan, 3.0)

    def test_raises_if_ask_price_is_nan(self):
        now = datetime.now()
        with pytest.raises(ValueError):
            Trade(now, ETF("SPY"), 1.0, 2.0, np.nan)

    def test_raises_if_quantity_is_nan(self):
        now = datetime.now()
        with pytest.raises(ValueError):
            Trade(now, ETF("SPY"), np.nan, 2.0, 3.0)

    def test_raises_if_quantity_is_zero(self):
        now = datetime.now()
        with pytest.raises(ValueError):
            Trade(now, ETF("SPY"), 0.0, 2.0, 3.0)

    def test_raises_if_contract_is_cash(self):
        now = datetime.now()
        with pytest.raises(ValueError):
            Trade(now, Cash(), 0.0, 2.0, 3.0)

    def test_avg_price_when_buy(self):
        now = datetime.now()
        trade = Trade(now, ETF("SPY"), 1.0, 2.0, 3.0)
        assert trade.acq_price == 3.0

    def test_avg_price_when_sell(self):
        now = datetime.now()
        trade = Trade(now, ETF("SPY"), -1.0, 2.0, 3.0)
        assert trade.acq_price == 2.0

    def test_cost_of_cash_with_etf(self):
        spy = ETF("SPY")
        now = datetime.now()
        trade = Trade(now, spy, -1.5, 2.0, 3.0)
        assert trade.cost_of_cash == -1.5 * 2 * spy.multiplier * spy.cash_requirement

    def test_cost_of_cash_with_future(self):
        es = ES(2019, 6)
        now = datetime.now()
        trade = Trade(now, es, -1.5, 2.0, 3.0)
        assert trade.cost_of_cash == 0.0

    def test_cost_of_commissions_are_zero_by_default(self):
        now = datetime.now()
        trade = Trade(now, ETF("SPY"), 1.0, 2.0, 3.0)
        assert trade.cost_of_commissions == 0.0

    def test_cost_of_commissions_called_with_trade(self, mocker):
        class Fees(IBrokerFees):
            def commissions(self, trade: "Trade") -> float:
                return 0.42

        fees = Fees()
        mocker.patch.object(fees, "commissions")
        now = datetime.now()
        trade = Trade(now, ETF("SPY"), -5.0, 2.0, 4.0, fees)
        fees.commissions.called_once_with(trade)

    def test_cost_of_commissions(self):
        class Commissions(IBrokerFees):
            def commissions(self, trade: "Trade") -> float:
                return 0.42

        commissions = Commissions()
        now = datetime.now()
        trade = Trade(now, ETF("SPY"), -5.0, 2.0, 4.0, commissions)
        assert trade.cost_of_commissions == 0.42

    def test_cost_of_spread_when_negative_quantity(self):
        es = ES(2019, 6)
        now = datetime.now()
        trade = Trade(now, es, -1.5, 2.0, 3.0)
        actual = trade.cost_of_spread
        expected = (
            abs(trade.quantity)
            * es.multiplier
            * (trade.ask_price - trade.bid_price)
        )
        assert actual == expected

    def test_cost_of_spread_when_positive_quantity(self):
        es = ES(2019, 6)
        now = datetime.now()
        trade = Trade(now, es, 3, 2.0, 3.0)
        actual = trade.cost_of_spread
        expected = (
            trade.quantity * es.multiplier * (trade.ask_price - trade.bid_price)
        )
        assert actual == expected

    def test_equality(self):
        es = ES(2019, 6)
        now = datetime.now()
        assert Trade(now, es, 3, 2, 3) == Trade(now, es, 3, 2, 3)

    def test_equality_different_time(self):
        es = ES(2019, 6)
        now = datetime.now()
        assert Trade(now, es, 3, 2, 3) != Trade(now + timedelta(seconds=1), es, 3, 2, 3)

    def test_equality_different_contract(self):
        es = ES(2019, 6)
        zn = ZN(2019, 6)
        now = datetime.now()
        assert Trade(now, es, 3, 2, 3) != Trade(now, zn, 3, 2, 3)

    def test_equality_different_quantity(self):
        es = ES(2019, 6)
        now = datetime.now()
        assert Trade(now, es, 3, 2, 3) != Trade(now, es, 4, 2, 3)

    def test_equality_different_bid(self):
        es = ES(2019, 6)
        now = datetime.now()
        assert Trade(now, es, -3, 2, 3) != Trade(now, es, -3, 2.5, 3)

    def test_equality_different_ask(self):
        es = ES(2019, 6)
        now = datetime.now()
        assert Trade(now, es, 3, 2, 3) != Trade(now, es, 3, 2, 3.5)
