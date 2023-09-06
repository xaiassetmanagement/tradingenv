from tradingenv.broker.rebalancing import Rebalancing
from tradingenv.broker.fees import IBrokerFees, BrokerFees
from tradingenv.broker.trade import Trade
from tradingenv.broker.broker import (
    Context,
    Broker,
    EndOfEpisodeError,
    SECONDS_IN_YEAR,
)
from tradingenv.exchange import Exchange
from tradingenv.contracts import Cash, Rate, ETF, ES, VX
from tradingenv.events import EventNBBO
from datetime import datetime, timedelta
from typing import List
import numpy as np
import pytest


class TestBroker:
    def make_exchange(self, nbbo_events: List[EventNBBO] = None):
        if nbbo_events is None:
            nbbo_events = list()
        nbbo_events += [
            EventNBBO(datetime.now(), Cash(), 1, 1),
            EventNBBO(datetime.now(), Rate("FED funds rate"), 0.0, 0.0),
        ]
        exchange = Exchange()
        for nbbo in nbbo_events:
            exchange.process_EventNBBO(nbbo)
        return exchange

    def test_default_initialization(self):
        exchange = Exchange()
        broker = Broker(exchange)
        assert exchange is exchange
        assert broker.base_currency == Cash()
        assert broker.fees.interest_rate == Rate("FED funds rate")
        assert broker.holdings_quantity == {broker.base_currency: 100.0}
        assert isinstance(broker.fees, BrokerFees)
        assert broker.holdings_margins == dict()
        assert broker._last_accrual is None
        assert broker._initial_deposit == 100.0

    def test_custom_initialization(self):
        exchange = Exchange()
        broker = Broker(
            exchange,
            Cash("EUR"),
            42.0,
            BrokerFees(markup=0.1, interest_rate=Rate("Euribor")),
        )
        assert exchange is exchange
        assert broker.base_currency == Cash("EUR")
        assert broker.fees.interest_rate == Rate("Euribor")
        assert broker._last_accrual is None
        assert broker.fees.markup == 0.1
        assert broker.holdings_margins == dict()
        assert broker.holdings_quantity == {broker.base_currency: 42.0}
        assert broker._initial_deposit == 42.0

    def test_holdings_quantity_returns_copy(self):
        broker = Broker(self.make_exchange())
        first = broker.holdings_quantity
        second = broker.holdings_quantity
        assert first is not second

    def test_holdings_margins_returns_copy(self):
        broker = Broker(self.make_exchange())
        first = broker.holdings_margins
        second = broker.holdings_margins
        assert first is not second

    def test_holdings_values_raises_when_missing_prices(self):
        exchange = self.make_exchange()
        broker = Broker(exchange)
        broker._holdings_quantity[ETF("SPY")] = 1.0
        msg = "Missing liquidation transaction_price for"
        with pytest.raises(ValueError, match=msg):
            broker.holdings_values()

    def test_holdings_values_works_when_missing_non_needed_prices(self):
        exchange = self.make_exchange()
        broker = Broker(exchange)
        broker._holdings_quantity[ETF("SPY")] = 0.0
        actual = broker.holdings_values()
        expected = {broker.base_currency: broker._initial_deposit, ETF("SPY"): 0.0}
        assert actual, expected

    def test_holdings_values_when_price_spread_is_zero(self):
        exchange = self.make_exchange(
            nbbo_events=[
                EventNBBO(datetime.now(), ETF("SPY"), 2, 2),
                EventNBBO(datetime.now(), ETF("IEF"), 3, 3),
            ]
        )
        broker = Broker(exchange)
        broker._holdings_quantity[broker.base_currency] = 10.0
        broker._holdings_quantity[ETF("SPY")] = -42
        broker._holdings_quantity[ETF("IEF")] = 18
        actual = broker.holdings_values()
        expected = {
            broker.base_currency: 10 * 1,
            ETF("SPY"): -42 * 2,
            ETF("IEF"): 18 * 3,
        }
        assert actual == expected

    def test_holdings_values_when_price_spread_is_positive(self):
        exchange = self.make_exchange(
            nbbo_events=[
                EventNBBO(datetime.now(), ETF("SPY"), 1.99, 2.01),
                EventNBBO(datetime.now(), ETF("IEF"), 2.98, 3.02),
            ]
        )
        broker = Broker(exchange)
        broker._holdings_quantity[broker.base_currency] = 10.0
        broker._holdings_quantity[ETF("SPY")] = -42
        broker._holdings_quantity[ETF("IEF")] = 18
        actual = broker.holdings_values()
        expected = {
            broker.base_currency: 10 * 1,
            ETF("SPY"): -42 * 2.01,
            ETF("IEF"): 18 * 2.98,
        }
        assert actual == expected

    def test_holdings_values_is_multiplied_by_multiplier(self):
        exchange = self.make_exchange(
            nbbo_events=[EventNBBO(datetime.now(), ETF("SPY"), 2, 2),
                         EventNBBO(datetime.now(), ES(2019, 6), 3, 3)]
        )
        broker = Broker(exchange)
        broker._holdings_quantity[broker.base_currency] = 10.0
        broker._holdings_quantity[ETF("SPY")] = -42
        broker._holdings_quantity[ES(2019, 6)] = 18
        actual = broker.holdings_values()
        expected = {
            broker.base_currency: 10 * 1,
            ETF("SPY"): -42 * 2 * ETF("SPY").multiplier,
            ES(2019, 6): 18 * 3 * ES(2019, 6).multiplier,
        }
        assert actual == expected

    def test_net_liquidation_value(self):
        exchange = self.make_exchange(
            nbbo_events=[
                EventNBBO(datetime.now(), ETF("SPY"), 1.99, 2.01),
                EventNBBO(datetime.now(), ETF("IEF"), 2.98, 3.02),
            ]
        )
        broker = Broker(exchange)
        broker._holdings_quantity[broker.base_currency] = 10.0
        broker._holdings_quantity[ETF("SPY")] = -2
        broker._holdings_quantity[ETF("IEF")] = 18
        actual = broker.net_liquidation_value()
        expected = 10 * 1 - 2 * 2.01 + 18 * 2.98
        assert actual == expected

    def test_net_liquidation_value_raises_if_nlv_is_zero(self):
        exchange = self.make_exchange(
            nbbo_events=[
                EventNBBO(datetime.now(), ETF("SPY"), 1.99, 2.01),
                EventNBBO(datetime.now(), ETF("IEF"), 2.98, 3.02),
            ]
        )
        broker = Broker(exchange)
        broker._holdings_quantity[broker.base_currency] = 0.0
        broker._holdings_quantity[ETF("SPY")] = 0.0
        broker._holdings_quantity[ETF("IEF")] = 0.0
        with pytest.raises(EndOfEpisodeError):
            broker.net_liquidation_value()

    def test_net_liquidation_value_raises_if_nlv_is_negative(self):
        exchange = self.make_exchange(
            nbbo_events=[
                EventNBBO(datetime.now(), ETF("SPY"), 1.99, 2.01),
                EventNBBO(datetime.now(), ETF("IEF"), 2.98, 3.02),
            ]
        )
        broker = Broker(exchange)
        broker._holdings_quantity[broker.base_currency] = 1
        broker._holdings_quantity[ETF("SPY")] = -1
        broker._holdings_quantity[ETF("IEF")] = 0.0
        with pytest.raises(EndOfEpisodeError):
            broker.net_liquidation_value()

    def test_net_liquidation_value_when_marking_to_market_is_needed_1(self):
        exchange = self.make_exchange([EventNBBO(datetime.now(), ES(2019, 6), 99, 100)])
        broker = Broker(exchange, deposit=100.0)
        notional_value_of_one_contract = 100 * ES(2019, 6).multiplier
        quantity = broker._initial_deposit / notional_value_of_one_contract
        trade = Trade(datetime.now(), ES(2019, 6), quantity, 99, 100)
        broker.transact(trade)
        trade = Trade(datetime.now(), ES(2019, 6), -quantity, 99, 100)
        broker.transact(trade)
        # bid_ask spread deduced from cash
        assert broker.net_liquidation_value() == 99

    def test_net_liquidation_value_when_marking_to_market_is_needed_2(self):
        exchange = self.make_exchange([EventNBBO(datetime.now(), ES(2019, 6), 100, 100)])
        broker = Broker(exchange, deposit=100.0)
        notional_value_of_one_contract = 100 * ES(2019, 6).multiplier
        quantity = broker._initial_deposit / notional_value_of_one_contract
        trade = Trade(datetime.now(), ES(2019, 6), quantity, 100, 100)
        broker.transact(trade)
        exchange.process_EventNBBO(EventNBBO(datetime.now(), ES(2019, 6), 101, 101))
        # bid_ask spread deduced from cash
        assert broker.net_liquidation_value() == 101.0

    @pytest.mark.parametrize(
        "holdings_quantity, expected",
        [
            (
                {Cash(): 100, ETF("SPY"): 50, ETF("IEF"): 100 / 3},
                {Cash(): 1 / 3, ETF("SPY"): 1 / 3, ETF("IEF"): 1 / 3},
            ),
            (
                {Cash(): 200, ETF("SPY"): 100, ETF("IEF"): -200 / 3},
                {Cash(): 1, ETF("SPY"): 1, ETF("IEF"): -1},
            ),
            (
                {Cash(): 100, ETF("SPY"): 0, ETF("IEF"): 0},
                {Cash(): 1, ETF("SPY"): 0, ETF("IEF"): 0},
            ),
            (
                {Cash(): 1, ETF("SPY"): 0, ETF("IEF"): 0},
                {Cash(): 1, ETF("SPY"): 0, ETF("IEF"): 0},
            ),
            (
                {Cash(): -100, ETF("SPY"): 100, ETF("IEF"): 0},
                {Cash(): -1, ETF("SPY"): 2, ETF("IEF"): 0},
            ),
        ],
    )
    def test_holdings_weights(self, holdings_quantity, expected):
        exchange = self.make_exchange(
            nbbo_events=[
                EventNBBO(datetime.now(), ETF("SPY"), 2, 2),
                EventNBBO(datetime.now(), ETF("IEF"), 3, 3)]
        )
        broker = Broker(exchange)
        broker._holdings_quantity = holdings_quantity
        actual = broker.holdings_weights()
        assert actual == expected

    def test_holdings_weights_with_drifting_prices(self):
        exchange = self.make_exchange(
            nbbo_events=[EventNBBO(datetime.now(), ETF("SPY"), 2, 2),
                         EventNBBO(datetime.now(), ETF("IEF"), 3, 3)]
        )
        broker = Broker(exchange)
        assert broker.holdings_quantity == {broker.base_currency: 100.0}

        broker._holdings_quantity = {
            broker.base_currency: 30,
            ETF("SPY"): 20,
            ETF("IEF"): -10,
        }
        exchange.process_EventNBBO(EventNBBO(datetime.now(), ETF("SPY"), 1.99, 2.01))
        exchange.process_EventNBBO(EventNBBO(datetime.now(), ETF("IEF"), 2.98, 3.02))
        actual = broker.holdings_weights()
        holdings_values = {
            broker.base_currency: 30 * 1,
            ETF("SPY"): 20 * 1.99,
            ETF("IEF"): -10 * 3.02,
        }
        nlv = sum(holdings_values.values())
        expected = {
            broker.base_currency: 30 * 1 / nlv,
            ETF("SPY"): 20 * 1.99 / nlv,
            ETF("IEF"): -10 * 3.02 / nlv,
        }
        assert actual == expected

        exchange.process_EventNBBO(EventNBBO(datetime.now(), ETF("SPY"), 2.44, 2.48))
        exchange.process_EventNBBO(EventNBBO(datetime.now(), ETF("IEF"), 2.14, 2.31))
        actual = broker.holdings_weights()
        holdings_values = {
            broker.base_currency: 30 * 1,
            ETF("SPY"): 20 * 2.44,
            ETF("IEF"): -10 * 2.31,
        }
        nlv = sum(holdings_values.values())
        expected = {
            broker.base_currency: 30 * 1 / nlv,
            ETF("SPY"): 20 * 2.44 / nlv,
            ETF("IEF"): -10 * 2.31 / nlv,
        }
        assert actual == expected

    @pytest.mark.parametrize(
        "deposit, rate, markup, expected",
        [
            (1.0, 0, 0, 0),
            (1, 0.01, 0, 0.01),
            (1, 0.01, 0.01, 0),
            (-1, 0.01, 0, -0.01),
            (-1, 0.01, 0.01, -0.02),
            (1, 0.03, 0.01, 0.02),
            (-1, 0.03, 0.01, -0.04),
            (1, 0.05, 0.005, 0.045),
            (-1, 0.05, 0.005, -0.055),
            (1, -0.05, 0.005, 0),  # don't accrue negative interest rates
        ],
    )
    def test_accrued_interest(self, deposit, rate, markup, expected):
        exchange = Exchange()
        broker = Broker(exchange, deposit=deposit, fees=BrokerFees(markup))
        exchange.process_EventNBBO(EventNBBO(datetime.now(), broker.fees.interest_rate, rate, rate))
        broker._last_accrual = datetime(2019, 1, 1)
        now = broker._last_accrual + timedelta(seconds=SECONDS_IN_YEAR)

        # Assert value is correct...
        np.testing.assert_almost_equal(
            actual=broker.accrued_interest(now), desired=expected, decimal=7
        )
        # ...but not reflected in position holdings as accrue=False by default.
        assert broker.holdings_quantity[broker.base_currency] == deposit
        assert broker._last_accrual == datetime(2019, 1, 1)

        # Assert value is correct...
        np.testing.assert_almost_equal(
            actual=broker.accrued_interest(now, accrue=True),
            desired=expected,
            decimal=7,
        )
        # ...but not reflected in position holdings when accrue=True.
        assert broker.holdings_quantity[broker.base_currency] == deposit + expected
        assert broker._last_accrual == now

    def test_accrued_interest_sets_last_update_when_run_for_the_first_time(self):
        exchange = Exchange()
        broker = Broker(exchange)
        assert broker._last_accrual is None
        broker.accrued_interest(datetime(2019, 1, 1))
        assert broker._last_accrual == datetime(2019, 1, 1)

    def test_accrued_interest_raise_if_now_less_then_last_update_time(self):
        exchange = Exchange()
        broker = Broker(exchange)
        broker._last_accrual = datetime(2019, 1, 1)
        with pytest.raises(ValueError):
            broker.accrued_interest(datetime(2018, 6, 6))

    @pytest.mark.parametrize(
        "data",
        [
            {
                "trades": [
                    Trade(datetime.now(), ETF("SPY"), 1, 41, 43),
                    Trade(datetime.now(), ETF("IEF"), 2.5, 17, 19),
                ],
                "holdings_quantity_expected": {
                    Cash(): 100 - 1 * 43 - 2.5 * 19,
                    ETF("SPY"): 1,
                    ETF("IEF"): 2.5,
                },
                "holdings_margin_expected": {ETF("SPY"): 0.0, ETF("IEF"): 0.0},
            },
            {
                "trades": [
                    Trade(datetime.now(), ETF("SPY"), -1, 41, 43),
                    Trade(datetime.now(), ETF("IEF"), 2.5, 17, 19),
                ],
                "holdings_quantity_expected": {
                    Cash(): 100 + 1 * 41 - 2.5 * 19,
                    ETF("SPY"): -1,
                    ETF("IEF"): 2.5,
                },
                "holdings_margin_expected": {ETF("SPY"): 0.0, ETF("IEF"): 0.0},
            },
            {
                "trades": [
                    Trade(datetime.now(), ETF("SPY"), 1, 41, 43),
                    Trade(datetime.now(), ETF("IEF"), -2.5, 17, 19),
                ],
                "holdings_quantity_expected": {
                    Cash(): 100 - 1 * 43 + 2.5 * 17,
                    ETF("SPY"): 1,
                    ETF("IEF"): -2.5,
                },
                "holdings_margin_expected": {ETF("SPY"): 0.0, ETF("IEF"): 0.0},
            },
            {
                "trades": [
                    Trade(datetime.now(), ETF("SPY"), -1, 41, 43),
                    Trade(datetime.now(), ETF("IEF"), -2.5, 17, 19),
                ],
                "holdings_quantity_expected": {
                    Cash(): 100 + 1 * 41 + 2.5 * 17,
                    ETF("SPY"): -1,
                    ETF("IEF"): -2.5,
                },
                "holdings_margin_expected": {ETF("SPY"): 0.0, ETF("IEF"): 0.0},
            },
            {
                "trades": [Trade(datetime.now(), ES(2019, 6), 2.5, 41, 43)],
                "holdings_quantity_expected": {
                    Cash(): 100
                    - 2.5
                    * 43
                    * ES(2019, 6).multiplier
                    * ES(2019, 6).margin_requirement,
                    ES(2019, 6): 2.5,
                },
                "holdings_margin_expected": {
                    ES(2019, 6): 2.5
                    * 43
                    * ES(2019, 6).multiplier
                    * ES(2019, 6).margin_requirement
                },
            },
            {
                "trades": [Trade(datetime.now(), ES(2019, 6), -2.5, 41, 43)],
                "holdings_quantity_expected": {
                    Cash(): 100
                    - 2.5
                    * 41
                    * ES(2019, 6).multiplier
                    * ES(2019, 6).margin_requirement,
                    ES(2019, 6): -2.5,
                },
                # Note positive _margin when shorting.
                "holdings_margin_expected": {
                    ES(2019, 6): 2.5
                    * 41
                    * ES(2019, 6).multiplier
                    * ES(2019, 6).margin_requirement
                },
            },
        ],
    )
    def test_transact(self, data):
        exchange = self.make_exchange()
        broker = Broker(exchange, deposit=100.0)
        for trade in data["trades"]:
            broker.transact(trade)
        assert broker.holdings_quantity == data["holdings_quantity_expected"]
        assert broker.holdings_margins == data["holdings_margin_expected"]

    def test_transact_futures_lifecycle_without_bid_ask_spread(self):
        exchange = self.make_exchange([EventNBBO(datetime.now(), ES(2019, 6), 42, 42)])
        broker = Broker(exchange, deposit=100.0)
        trade = Trade(datetime.now(), ES(2019, 6), 2.5, 42, 42)
        broker.transact(trade)
        steps = 1000
        for _ in range(steps):
            trade = Trade(datetime.now(), ES(2019, 6), -2.5 / steps, 42, 42)
            broker.transact(trade)
        np.testing.assert_almost_equal(
            actual=broker.holdings_quantity[ES(2019, 6)], desired=0.0
        )
        np.testing.assert_almost_equal(
            actual=broker.holdings_margins[ES(2019, 6)], desired=0.0
        )

    def test_transact_futures_lifecycle_with_bid_ask_spread_single_lot(self):
        exchange = self.make_exchange([EventNBBO(datetime.now(), ES(2019, 6), 99, 100)])
        broker = Broker(exchange, deposit=100.0)
        notional_value_of_one_contract = 100 * ES(2019, 6).multiplier
        quantity = broker._initial_deposit / notional_value_of_one_contract
        trade = Trade(datetime.now(), ES(2019, 6), quantity, 99, 100)
        broker.transact(trade)
        trade = Trade(datetime.now(), ES(2019, 6), -quantity, 99, 100)
        broker.transact(trade)
        np.testing.assert_almost_equal(
            actual=broker.holdings_quantity[Cash()],
            desired=99,  # bid_ask spread deduced from cash
        )

    def test_transact_futures_lifecycle_with_bid_ask_spread_multiple_lots(self):
        exchange = self.make_exchange([EventNBBO(datetime.now(), ES(2019, 6), 99, 100)])
        broker = Broker(exchange, deposit=100.0)
        notional_value_of_one_contract = 100 * ES(2019, 6).multiplier
        quantity = broker._initial_deposit / notional_value_of_one_contract
        trade = Trade(datetime.now(), ES(2019, 6), quantity, 99, 100)
        broker.transact(trade)
        steps = 1000
        for _ in range(steps):
            trade = Trade(datetime.now(), ES(2019, 6), -quantity / steps, 99, 100)
            broker.transact(trade)
        np.testing.assert_almost_equal(
            actual=broker.holdings_quantity[Cash()],
            desired=99,  # bid_ask spread deduced from cash
        )

    def test_transact_deduces_cost_of_commissions(self):
        class Fees(IBrokerFees):
            def commissions(self, trade) -> float:
                return 18

        fees = Fees()
        exchange = self.make_exchange([EventNBBO(datetime.now(), ETF("SPY"), 42, 42)])
        broker = Broker(exchange, deposit=100.0)
        trade = Trade(datetime.now(), ETF("SPY"), 2.5, 42, 42, fees)
        broker.transact(trade)
        trade = Trade(datetime.now(), ETF("SPY"), -2.5, 42, 42, fees)
        broker.transact(trade)
        np.testing.assert_almost_equal(
            actual=broker.holdings_quantity[ETF("SPY")], desired=0.0
        )
        np.testing.assert_almost_equal(
            actual=broker.holdings_margins[ETF("SPY")], desired=0.0
        )
        np.testing.assert_almost_equal(
            actual=broker.holdings_quantity[Cash()],
            desired=100 - 18 * 2,  # bid_ask spread deduced from cash
        )

    @pytest.mark.parametrize("arg", [ES(2019, 6), None])
    @pytest.mark.parametrize(
        "bid_prices, ask_prices",
        [
            # No bid-ask spread.
            [[1872.5, 1847.5, 1900.0, 1872.5], [1872.5, 1847.5, 1900.0, 1872.5]],
            # Bid-ask spread.
            [[1872.0, 1847.0, 1899.5, 1872.0], [1873.0, 1848.0, 1900.5, 1873.0]],
        ],
    )
    def test_marking_to_market(self, arg, bid_prices, ask_prices):
        es = ES(2019, 6)
        exchange = self.make_exchange()
        exchange.process_EventNBBO(EventNBBO(datetime.now(), es, bid_prices[0], ask_prices[0]))
        deposit = 100
        broker = Broker(exchange, deposit=deposit)

        # Buy ES.
        lob = exchange[es]
        trade = Trade(datetime.now(), es, 0.002, lob.bid_price, lob.ask_price)
        broker.transact(trade)

        # Check status.
        initial_margin = (
            lob.liq_price(trade.quantity)
            * es.margin_requirement
            * trade.quantity
            * es.multiplier
        )
        roundtrip_spread = trade.quantity * es.multiplier * lob.spread
        cash = broker.holdings_quantity[broker.base_currency]
        assert broker.holdings_margins[es] == initial_margin
        assert broker.holdings_quantity[es] == trade.quantity
        np.testing.assert_almost_equal(
            actual=broker.holdings_quantity[broker.base_currency],
            desired=deposit - initial_margin - roundtrip_spread,
        )
        np.testing.assert_almost_equal(
            actual=sum(broker.holdings_margins.values()) + cash,
            desired=deposit - roundtrip_spread,
        )

        # ES price moves
        exchange.process_EventNBBO(EventNBBO(datetime.now(), es, bid_prices[1], ask_prices[1]))

        # Check status. Nothing should have changed because mark-to-market
        # has not been called (yet).
        assert broker.holdings_margins[es] == initial_margin
        cash = broker.holdings_quantity[broker.base_currency]
        assert broker.holdings_quantity[es] == trade.quantity
        np.testing.assert_almost_equal(
            actual=broker.holdings_quantity[broker.base_currency],
            desired=deposit - initial_margin - roundtrip_spread,
        )
        np.testing.assert_almost_equal(
            actual=sum(broker.holdings_margins.values()) + cash,
            desired=deposit - roundtrip_spread,
        )

        # Check status. Something should have changed now that mark-to-market
        # has been called. Prices have moved down, so loss (<).
        broker.marking_to_market(arg)
        new_margin = (
            lob.liq_price(trade.quantity)
            * es.margin_requirement
            * trade.quantity
            * es.multiplier
        )
        capital_gain1 = es.multiplier * (bid_prices[1] - bid_prices[0]) * trade.quantity
        assert new_margin < initial_margin
        cash = broker.holdings_quantity[broker.base_currency]
        np.testing.assert_almost_equal(
            actual=broker.holdings_margins[es], desired=new_margin
        )
        assert broker.holdings_quantity[es] == trade.quantity
        np.testing.assert_almost_equal(
            actual=sum(broker.holdings_margins.values()) + cash,
            desired=deposit + capital_gain1 - roundtrip_spread,
        )
        np.testing.assert_almost_equal(
            actual=cash, desired=deposit - new_margin + capital_gain1 - roundtrip_spread
        )

        # ES price moves
        exchange.process_EventNBBO(EventNBBO(datetime.now(), es, bid_prices[2], ask_prices[2]))

        # Check status. Nothing should have changed because mark-to-market
        # has not been called (yet).
        assert new_margin < initial_margin
        np.testing.assert_almost_equal(
            actual=broker.holdings_margins[es], desired=new_margin
        )
        assert broker.holdings_quantity[es] == trade.quantity
        np.testing.assert_almost_equal(
            actual=sum(broker.holdings_margins.values()) + cash,
            desired=deposit + capital_gain1 - roundtrip_spread,
        )
        np.testing.assert_almost_equal(
            actual=broker.holdings_quantity[broker.base_currency],
            desired=deposit - new_margin + capital_gain1 - roundtrip_spread,
        )

        # Check status. Something should have changed now that mark-to-market
        # has been called. Prices have moved down, so loss (<).
        broker.marking_to_market(arg)
        previous_margin = new_margin
        new_margin = (
            lob.liq_price(trade.quantity)
            * es.margin_requirement
            * trade.quantity
            * es.multiplier
        )
        capital_gain2 = es.multiplier * (bid_prices[2] - bid_prices[1]) * trade.quantity
        assert new_margin > previous_margin
        cash = broker.holdings_quantity[broker.base_currency]
        np.testing.assert_almost_equal(
            actual=broker.holdings_margins[es], desired=new_margin
        )
        assert broker.holdings_quantity[es] == trade.quantity
        np.testing.assert_almost_equal(
            actual=sum(broker.holdings_margins.values()) + cash,
            desired=deposit + capital_gain1 + capital_gain2 - roundtrip_spread,
        )
        np.testing.assert_almost_equal(
            actual=cash,
            desired=deposit
            - new_margin
            + capital_gain1
            + capital_gain2
            - roundtrip_spread,
        )

        # ES price moves and close position. Price is start price, so PnL
        # should be zero.
        exchange.process_EventNBBO(EventNBBO(datetime.now(), es, bid_prices[3], ask_prices[3]))
        lob = exchange[es]
        trade = Trade(datetime.now(), es, -trade.quantity, lob.bid_price, lob.ask_price)
        broker.transact(trade)

        # Check status. Something should have changed now that mark-to-market
        # has been called. Prices have moved down, so loss (<).
        broker.marking_to_market(arg)
        capital_gain3 = es.multiplier * (bid_prices[3] - bid_prices[2]) * trade.quantity
        cash = broker.holdings_quantity[broker.base_currency]
        assert broker.holdings_margins[es] == 0
        assert broker.holdings_quantity[es] == 0
        np.testing.assert_almost_equal(actual=cash, desired=deposit - roundtrip_spread)

    def test_marking_to_market_does_nothing_when_prices_do_not_change(self):
        es = ES(2019, 6)
        exchange = self.make_exchange()
        exchange.process_EventNBBO(EventNBBO(datetime.now(), es, 1872.5, 1872.5))
        deposit = 100
        broker = Broker(exchange, deposit=deposit)

        # Buy ES.
        lob = exchange[es]
        trade = Trade(datetime.now(), es, 0.002, lob.bid_price, lob.ask_price)
        broker.transact(trade)

        # Check status.
        initial_margin = (
            lob.mid_price * es.margin_requirement * trade.quantity * es.multiplier
        )
        cash = broker.holdings_quantity[broker.base_currency]
        assert broker.holdings_margins[es] == initial_margin
        assert broker.holdings_quantity[es] == trade.quantity
        assert (
            broker.holdings_quantity[broker.base_currency] == deposit - initial_margin
        )
        assert sum(broker.holdings_margins.values()) + cash == deposit

        # ES price moves
        exchange.process_EventNBBO(EventNBBO(datetime.now(), es, 1872.5, 1872.5))

        # Check status. Nothing should have changed because mark-to-market
        # has not been called.
        assert broker.holdings_margins[es] == initial_margin
        cash = broker.holdings_quantity[broker.base_currency]
        assert broker.holdings_quantity[es] == trade.quantity
        assert cash == deposit - initial_margin
        assert sum(broker.holdings_margins.values()) + cash == deposit

        # Check status. Nothing should have changed because prices haven't
        # changed.
        broker.marking_to_market()
        new_margin = (
            lob.mid_price * es.margin_requirement * trade.quantity * es.multiplier
        )
        capital_gain1 = es.multiplier * (1872.5 - 1872.5) * trade.quantity
        cash = broker.holdings_quantity[broker.base_currency]
        assert broker.holdings_margins[es] == new_margin
        assert broker.holdings_quantity[es] == trade.quantity
        assert sum(broker.holdings_margins.values()) + cash == deposit + capital_gain1
        assert cash == deposit - new_margin + capital_gain1

    def test_marking_to_market_does_not_raise_with_spot_products(self):
        spy = ETF("SPY")
        exchange = self.make_exchange()
        exchange.process_EventNBBO(EventNBBO(datetime.now(), spy, 1872.5, 1872.5))
        broker = Broker(exchange, deposit=100)
        lob = exchange[spy]
        trade = Trade(datetime.now(), spy, 0.002, lob.bid_price, lob.ask_price)
        broker.transact(trade)
        broker.marking_to_market(spy)  # does not raise

    @pytest.mark.parametrize('quantity', [-0.002, 0.002])
    def test_marking_to_market_margin_is_agnostic_to_sign_of_position(self, quantity):
        es = ES(2019, 3)
        exchange = self.make_exchange()
        exchange.process_EventNBBO(EventNBBO(datetime.now(), es, 1872.5, 1872.5))
        broker = Broker(exchange, deposit=100)
        lob = exchange[es]
        trade = Trade(datetime.now(), es, quantity, lob.bid_price, lob.ask_price)
        broker.transact(trade)
        broker.marking_to_market(es)
        assert broker.holdings_margins[es] == 1872.5 * abs(quantity) * es.multiplier * es.margin_requirement

    def test_spread_is_calculated(self):
        # Initial portfolio is $100 in cash. Target portfolio is 60:40 in
        # equities and bonds, both with a 0.02% of spread everywhere. So LOB is
        # 99.99<->100.01. So roundtrip cost in PnL is $100 to buy at 100.01
        # means that I'll own 100/100.01 shares to be sold at 99.99
        exchange = self.make_exchange()
        spread = 0.0002

        mid_price = 2
        halftrip_spread = mid_price * spread / 2
        exchange.process_EventNBBO(
            EventNBBO(
                datetime.now(),
                contract=ETF("SPY"),
                bid_price=mid_price - halftrip_spread,
                ask_price=mid_price + halftrip_spread,
            )
        )

        mid_price = 3
        halftrip_spread = mid_price * spread / 2
        exchange.process_EventNBBO(
            EventNBBO(
                datetime.now(),
                contract=ETF("IEF"),
                bid_price=mid_price - halftrip_spread,
                ask_price=mid_price + halftrip_spread,
            )
        )
        broker = Broker(exchange)

        request = Rebalancing(
            time=datetime(2019, 1, 6),
            contracts=[Cash(), ETF("SPY"), ETF("IEF")],
            allocation=[0, 0.75, 0.25],
        )
        broker.rebalance(request)
        np.testing.assert_approx_equal(
            actual=request.context_post.nlv,
            desired=broker._initial_deposit / 100.01 * 99.99
        )
        assert broker.holdings_quantity == {
            Cash(): 0,
            ETF("SPY"): broker._initial_deposit * 0.75 / exchange[ETF("SPY")].ask_price,
            ETF("IEF"): broker._initial_deposit * 0.25 / exchange[ETF("IEF")].ask_price,
        }

    def test_context(self):
        exchange = self.make_exchange(
            nbbo_events=[
                EventNBBO(datetime.now(), ETF("SPY"), 1.99, 2.01),
                EventNBBO(datetime.now(), ETF("IEF"), 2.98, 3.02),
            ]
        )
        broker = Broker(exchange)
        broker._holdings_quantity[broker.base_currency] = 10.0
        broker._holdings_quantity[ETF("SPY")] = -2
        broker._holdings_quantity[ETF("IEF")] = 18
        context = broker.context()
        assert context.nlv == broker.net_liquidation_value()
        assert context.nr_contracts == broker.holdings_quantity
        assert context.weights == broker.holdings_weights()
        assert context.margins == broker.holdings_margins
        assert context.values == broker.holdings_values()


class TestBrokerRebalance:
    def make_broker(self, deposit: float=100):
        nbbos = [
            EventNBBO(datetime.now(), Cash(), 1, 1),
            EventNBBO(datetime.now(), Rate("FED funds rate"), 0.0182, 0.0182),
            EventNBBO(datetime.now(), ETF('SPY'), 293.24, 293.25),
            EventNBBO(datetime.now(), ETF('IEF'), 112.67, 112.68),
            EventNBBO(datetime.now(), VX(2020, 3), 19.20, 19.25),
            EventNBBO(datetime.now(), ES(2020, 3), 2941.75, 2942),
            EventNBBO(datetime.now(), ES(2020, 6), 2940, 2940.25),
            EventNBBO(datetime.now(), ES(2020, 9), 2938.75, 2939),
            EventNBBO(datetime.now(), ES(2020, 12), 2935.25, 2935),
            EventNBBO(datetime.now(), ETF("Equities"), 2, 2),
            EventNBBO(datetime.now(), ETF("Bonds"), 3, 3),

        ]
        exchange = Exchange()
        for nbbo in nbbos:
            exchange.process_EventNBBO(nbbo)
        return Broker(exchange=exchange, base_currency=Cash(), deposit=deposit)

    def test_rebalance_interest_rate_on_idle_cash_is_accrued(self):
        tic = datetime(2019, 1, 1)
        toc = tic + timedelta(days=252)
        broker = self.make_broker(deposit=100)
        broker._last_accrual = tic

        assert broker.net_liquidation_value() == 100
        expected = broker.accrued_interest(toc, accrue=False)
        rebalancing = Rebalancing(time=toc)
        broker.rebalance(rebalancing)
        # we are not testing how much, but check that something is accrued.
        assert expected > 0
        assert rebalancing.profit_on_idle_cash == expected
        assert broker.net_liquidation_value() == 100 + expected

    def test_rebalance_rebalancing_object_is_added_to_track_record(self):
        broker = self.make_broker()
        rebalancing = Rebalancing()
        broker.rebalance(rebalancing)
        assert broker.track_record._rebalancing[rebalancing.time] is rebalancing

    def test_rebalance_trades_are_saved_to_attribute_in_rebalancing(self):
        broker = self.make_broker()
        rebalancing = Rebalancing([ETF('SPY')], [0.8])
        expected = rebalancing.make_trades(broker)
        assert not isinstance(rebalancing.trades, list)
        broker.rebalance(rebalancing)
        assert rebalancing.trades == expected

    def test_rebalance_context_pre(self):
        broker = self.make_broker()
        rebalancing = Rebalancing([ETF('SPY')], [0.8])
        expected = broker.context()
        assert not isinstance(rebalancing.context_pre, Context)
        broker.rebalance(rebalancing)
        assert rebalancing.context_pre == expected

    def test_rebalance_context_post(self):
        broker = self.make_broker()
        rebalancing = Rebalancing([ETF('SPY')], [0.8])
        assert not isinstance(rebalancing.context_post, Context)
        broker.rebalance(rebalancing)
        expected = broker.context()
        assert rebalancing.context_post == expected

    def test_rebalance_simplest_case_from_all_cash_to_all_cash(self):
        broker = self.make_broker(deposit=100)
        time = datetime.now()
        request = Rebalancing(
            contracts=[Cash(), ETF("SPY"), ETF("IEF")],
            allocation=[1, 0, 0],
            time=time,
        )
        broker.rebalance(request)

        assert broker.holdings_quantity == {Cash(): broker._initial_deposit}
        assert broker.net_liquidation_value() == broker._initial_deposit
        assert broker.holdings_weights() == {Cash(): 1}

    def test_rebalance_from_all_cash_to_balanced_portfolio(self):
        broker = self.make_broker(deposit=100)
        time = datetime.now()
        rebalancing = Rebalancing(
            contracts=[Cash(), ETF("Equities"), ETF("Bonds")],
            allocation=[0, 0.75, 0.25],
            time=time,
        )
        broker.rebalance(rebalancing)
        assert broker.holdings_quantity == {
            Cash(): 0,
            ETF("Equities"): broker._initial_deposit * 0.75 / 2,
            ETF("Bonds"): broker._initial_deposit * 0.25 / 3,
        }
        assert broker.net_liquidation_value() == broker._initial_deposit
        assert broker.holdings_weights() == {Cash(): 0, ETF('Equities'): 0.75, ETF('Bonds'): 0.25}
