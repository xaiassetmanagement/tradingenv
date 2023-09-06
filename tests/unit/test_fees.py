from tradingenv.broker.fees import IBrokerFees, BrokerFees
from tradingenv.broker.trade import Trade
from tradingenv.contracts import ETF, Rate, ES
from datetime import datetime
from abc import ABC


class TestAbstractBrokerFees:
    def test_is_abc(self):
        assert issubclass(IBrokerFees, ABC)

    def test_commissions_is_abstractmethod(self):
        assert IBrokerFees.commissions.__isabstractmethod__

    def test_initialization(self):
        class Fees(IBrokerFees):
            def commissions(self, trade):
                return 0.0

        fees = Fees(0.18)
        assert fees.markup == 0.18
        assert fees.interest_rate == Rate("FED funds rate")
        assert fees.proportional == 0.0
        assert fees.fixed == 0.0


class TestBrokerFees:
    def test_commissions(self):
        fees = BrokerFees()
        trade = Trade(datetime.now(), ETF("SPY"), 2.0, 3.0, 4.0, fees)
        assert fees.commissions(trade) == 0.0

    def test_commissions_proportional_when_negative(self):
        fees = BrokerFees(proportional=0.01)
        trade = Trade(datetime.now(), ES(2019, 6), -2.0, 3.0, 4.0, fees)
        assert fees.commissions(trade) == abs(trade.notional) * fees.proportional

    def test_commissions_fixed_when_negative(self):
        fees = BrokerFees(fixed=0.01)
        trade = Trade(datetime.now(), ES(2019, 6), -2.0, 3.0, 4.0, fees)
        assert fees.commissions(trade) == fees.fixed

    def test_commissions_proportional_and_fixed_when_negative(self):
        fees = BrokerFees(proportional=0.01, fixed=0.02)
        trade = Trade(datetime.now(), ES(2019, 6), -2.0, 3.0, 4.0, fees)
        assert (
            fees.commissions(trade)
            == fees.fixed + abs(trade.notional) * fees.proportional
        )

    def test_commissions_proportional_when_positive(self):
        fees = BrokerFees(proportional=0.01)
        trade = Trade(datetime.now(), ES(2019, 6), 2.0, 3.0, 4.0, fees)
        assert fees.commissions(trade) == abs(trade.notional) * fees.proportional

    def test_commissions_fixed_when_positive(self):
        fees = BrokerFees(fixed=0.01)
        trade = Trade(datetime.now(), ES(2019, 6), 2.0, 3.0, 4.0, fees)
        assert fees.commissions(trade) == fees.fixed

    def test_commissions_proportional_and_fixed_when_positive(self):
        fees = BrokerFees(proportional=0.01, fixed=0.02)
        trade = Trade(datetime.now(), ES(2019, 6), 2.0, 3.0, 4.0, fees)
        assert (
            fees.commissions(trade)
            == fees.fixed + abs(trade.notional) * fees.proportional
        )
