from tradingenv.broker.allocation import NrContracts, Weights
from tradingenv.broker.rebalancing import Rebalancing
from tradingenv.broker.broker import Broker
from tradingenv.broker.trade import Trade
from tradingenv.events import EventNBBO, EventContractDiscontinued
from tradingenv.exchange import Exchange
from tradingenv.contracts import Cash, ETF, VX, ES, Rate, FutureChain
from collections import namedtuple
from datetime import datetime
import numpy as np
import pytest
Pair = namedtuple('Pair', ['allocation', 'expected'])
Imbalance = namedtuple('Pair', ['holdings', 'target', 'expected'])
MEASURES = ['weight', 'nr-contracts']
NOW = datetime.now()


def make_broker(deposit: float = 1000000, initial_holdings=None):
    initial_holdings = initial_holdings or dict()
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
    ]
    exchange = Exchange()
    for nbbo in nbbos:
        exchange.process_EventNBBO(nbbo)
    broker = Broker(exchange=exchange, base_currency=Cash(), deposit=deposit)
    for contract, quantity in initial_holdings.items():
        broker._holdings_quantity[contract] = quantity
    return broker


class TestRebalancing:
    def test_make_trades_does_not_raise_if_missing_not_needed_price(self):
        """Note that the price for ETF('IEF') if missing, but the _rebalancing
        proceeds anyway because the target weight is zero, so the price is not
        needed."""
        nlv = 100
        broker = make_broker(deposit=nlv)
        event = EventContractDiscontinued(datetime.now(), ETF('IEF'))
        broker.exchange[ETF('IEF')].terminate(event)
        assert np.isnan(broker.exchange[ETF('IEF')].mid_price)
        rebalancing = Rebalancing(
            contracts=[Cash(), ETF('SPY'), ETF('IEF')],
            allocation=[0.4, 0.6, 0.0],
            time=NOW,
        )
        actual = rebalancing.make_trades(broker)
        expected = [Trade(NOW, ETF('SPY'), nlv * 0.6 / 293.25, 293.24, 293.25)]
        assert actual == expected

    def test_make_trades_raise_if_missing_needed_price(self):
        """Note that the price for ETF('IEF') if missing, but the _rebalancing
        proceeds anyway because the target weight is zero, so the price is not
        needed."""
        nlv = 100
        broker = make_broker(deposit=nlv)
        event = EventContractDiscontinued(datetime.now(), ETF('IEF'))
        broker.exchange[ETF('IEF')].terminate(event)
        assert np.isnan(broker.exchange[ETF('IEF')].mid_price)
        rebalancing = Rebalancing(
            contracts=[Cash(), ETF('SPY'), ETF('IEF')],
            allocation=[0.4, 0.6, 0.2],
            time=NOW,
        )
        with pytest.raises(ValueError):
            rebalancing.make_trades(broker)

    def test_make_trades_when_absolute_is_false(self, mocker):
        """Absolute is True by default."""
        imbalance = {
            Cash(): 100,
            ETF('SPY'): -12.1,
            VX(2020, 3): 1.1,
            FutureChain(contracts=[ES(2020, 3), ES(2020, 6), ES(2020, 9)]): -1.2,
        }
        mocker.patch.object(
            target=Weights,
            attribute=Weights._to_nr_contracts.__name__,
            return_value=NrContracts(imbalance),
        )
        rebalancing = Rebalancing(absolute=False, time=NOW)
        broker = make_broker(deposit=100)
        actual = rebalancing.make_trades(broker)
        assert actual == [
            Trade(NOW, ETF('SPY'), -12.1, 293.24, 293.25),
            Trade(NOW, VX(2020, 3), 1.1, 19.20, 19.25),
            Trade(NOW, ES(2020, 3), -1.2, 2941.75, 2942),
        ]

    @pytest.mark.parametrize(
        argnames=['allocation', 'expected'],
        argvalues=[
            Pair(
                # to all cash.
                allocation={Cash(): 1, ETF('SPY'): 0, ETF('IEF'): 0},
                expected=[],
            ),
            Pair(
                # to balanced portfolio.
                allocation={Cash(): 0.1, ETF('SPY'): 0.6, ETF('IEF'): 0.3},
                expected=[
                    Trade(NOW, ETF('SPY'), 60 / 293.25, 293.24, 293.25),
                    Trade(NOW, ETF('IEF'), 30 / 112.68, 112.67, 112.68),
                ],
            ),
            Pair(
                # to leveraged balanced portfolio.
                allocation={Cash(): -0.8, ETF('SPY'): 1.2, ETF('IEF'): 0.6},
                expected=[
                    Trade(NOW, ETF('SPY'), 120 / 293.25, 293.24, 293.25),
                    Trade(NOW, ETF('IEF'), 60 / 112.68, 112.67, 112.68),
                ],
            ),
            Pair(
                # to short balanced portfolio.
                allocation={Cash(): 1.9, ETF('SPY'): -0.6, ETF('IEF'): -0.3},
                expected=[
                    Trade(NOW, ETF('SPY'), -60 / 293.24, 293.24, 293.25),
                    Trade(NOW, ETF('IEF'), -30 / 112.67, 112.67, 112.68),
                ],
            ),
            Pair(
                # to leveraged short balanced portfolio.
                allocation={Cash(): 2.8, ETF('SPY'): -1.2, ETF('IEF'): -0.6},
                expected=[
                    Trade(NOW, ETF('SPY'), -120 / 293.24, 293.24, 293.25),
                    Trade(NOW, ETF('IEF'), -60 / 112.67, 112.67, 112.68),
                ],
            ),
            Pair(
                # to leveraged short mixed portfolio.
                allocation={Cash(): 0.7, ETF('SPY'): 0.6, ETF('IEF'): -0.3},
                expected=[
                    Trade(NOW, ETF('SPY'), 60 / 293.25, 293.24, 293.25),
                    Trade(NOW, ETF('IEF'), -30 / 112.67, 112.67, 112.68),
                ],
            ),
        ]
    )
    def test_make_trades_from_weights_without_holdings(self, allocation, expected):
        rebalancing = Rebalancing(
            contracts=allocation.keys(),
            allocation=allocation.values(),
            absolute=True,
            measure='weight',
            time=NOW,
        )
        broker = make_broker(deposit=100)
        actual = rebalancing.make_trades(broker)
        assert actual == expected

    def test_make_trades_from_weights_with_holdings(self):
        broker = make_broker(deposit=100)
        broker._holdings_quantity = {
            Cash(): -20,
            ETF('SPY'): 80,
            ETF('IEF'): -10
        }
        rebalancing = Rebalancing(
            contracts=[Cash(), ETF('SPY'), ETF('IEF')],
            allocation=[-0.2, -0.5, 1.7],
            time=NOW,
        )
        trades_actual = rebalancing.make_trades(broker)
        nlv = broker.net_liquidation_value()
        positions_value_diff = {
            # Target position state minus current position state
            Cash(): (-0.2 * nlv) - (-20 * 1),
            ETF('SPY'): (-0.5 * nlv) - (80 * 293.24),
            ETF('IEF'): (1.7 * nlv) - (-10 * 112.68),
        }
        trades_expected = [
            Trade(NOW, ETF('SPY'), positions_value_diff[ETF('SPY')] / 293.24, 293.24, 293.25),
            Trade(NOW, ETF('IEF'), positions_value_diff[ETF('IEF')] / 112.68, 112.67, 112.68),
        ]
        assert trades_actual == trades_expected

    @pytest.mark.parametrize(
        argnames=['holdings', 'target', 'expected'],
        argvalues=[
            Imbalance(
                holdings={},
                target={Cash(): 100},
                expected=[],
            ),
            Imbalance(
                holdings={ETF('SPY'): 80},
                target={ETF('SPY'): 180},
                expected=[Trade(NOW, ETF('SPY'), 100, 293.24, 293.25)],
            ),
            Imbalance(
                holdings={ETF('SPY'): 10, ETF('IEF'): -5.5, ES(2020, 3): 0.41, VX(2020, 3): -1.2},
                target={},
                expected=[
                    Trade(NOW, ETF('SPY'), -10, 293.24, 293.25),
                    Trade(NOW, ETF('IEF'), 5.5, 112.67, 112.68),
                    Trade(NOW, ES(2020, 3), -0.41, 2941.75, 2942),
                    Trade(NOW, VX(2020, 3), 1.2, 19.20, 19.25),
                ],
            ),
            Imbalance(
                holdings={ETF('SPY'): 10, ETF('IEF'): -5.5, ES(2020, 3): 0.41, VX(2020, 3): -1.2},
                target={Cash(): 100},
                expected=[
                    Trade(NOW, ETF('SPY'), -10, 293.24, 293.25),
                    Trade(NOW, ETF('IEF'), 5.5, 112.67, 112.68),
                    Trade(NOW, ES(2020, 3), -0.41, 2941.75, 2942),
                    Trade(NOW, VX(2020, 3), 1.2, 19.20, 19.25),
                ],
            ),
            Imbalance(
                holdings={ES(2020, 3): 3},
                target={FutureChain(contracts=[ES(2020, 3)]): 10},
                expected=[
                    Trade(NOW, ES(2020, 3), 7, 2941.75, 2942),
                ],
            ),
        ]
    )
    def test_make_trades_from_nr_contracts(self, holdings, target, expected):
        rebalancing = Rebalancing(
            contracts=target.keys(),
            allocation=target.values(),
            absolute=True,
            measure='nr-contracts',
            time=NOW,
        )
        broker = make_broker(deposit=100, initial_holdings=holdings)
        actual = rebalancing.make_trades(broker)
        assert actual == expected

    def test_fractional(self):
        rebalancing = Rebalancing(
            contracts=[ETF('SPY'), ETF('IEF'), ES(2020, 3)],
            allocation=[21.8, -25.9, -1.9],
            absolute=True,
            measure='nr-contracts',
            fractional=False,
            time=NOW,
        )
        broker = make_broker(
            deposit=100,
            initial_holdings={
                ETF('SPY'): 0.2,
                ETF('IEF'): 25.7,
            },
        )
        actual = rebalancing.make_trades(broker)
        assert actual == [
            Trade(NOW, ETF('SPY'), 21, 293.24, 293.25),
            Trade(NOW, ETF('IEF'), -51, 112.67, 112.68),
            Trade(NOW, ES(2020, 3), -1, 2941.75, 2942),
        ]

    @pytest.mark.parametrize('weight', [0.049, -0.049])
    def test_margin_with_positive_weight(self, weight):
        rebalancing = Rebalancing(
            contracts=[ETF('SPY'), ETF('IEF'), ES(2020, 3)],
            allocation=[0.218, -0.259, weight],
            absolute=False,
            measure='weight',
            time=NOW,
            margin=0.05,
        )
        broker = make_broker()
        actual = rebalancing.make_trades(broker)
        expected = [
            Trade(NOW, ETF('SPY'), 743.3930093776642, 293.24, 293.25),
            Trade(NOW, ETF('IEF'), -2298.748557734978, 112.67, 112.68),
            # Note ES is missing because abs(weight) is < margin.
        ]
        assert actual == expected

    @pytest.mark.parametrize('sign', [-1, 1])
    def test_margin_with_liquidation_ignores_margin(self, sign):
        rebalancing = Rebalancing(
            time=NOW,
            margin=0.05,
        )
        broker = make_broker(initial_holdings={ES(2020, 3): 0.001 * sign})
        current_weight = broker.holdings_weights()[ES(2020, 3)]
        current_quantity = broker.holdings_quantity[ES(2020, 3)]
        assert abs(current_weight) < 0.05
        actual = rebalancing.make_trades(broker)
        expected = [
            Trade(NOW, ES(2020, 3), -current_quantity, 2941.75, 2942),
        ]
        assert actual == expected
