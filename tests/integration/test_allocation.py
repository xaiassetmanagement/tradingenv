from tradingenv.broker.allocation import NrContracts, Weights
from tradingenv.broker.broker import Broker
from tradingenv.events import EventNBBO
from tradingenv.exchange import Exchange
from tradingenv.contracts import Cash, ETF, VX, ES, Rate, FutureChain
from collections import namedtuple
from datetime import datetime
import pytest
Pair = namedtuple('Pair', ['allocation', 'expected'])
Imbalance = namedtuple('Pair', ['holdings', 'target', 'expected'])
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


class TestWeights:
    def test_to_nr_contracts_returns_nr_contracts_instance(self):
        broker = make_broker()
        weights = Weights({})
        assert isinstance(weights._to_nr_contracts(broker), NrContracts)

    def test_to_weights_returns_weights_instance(self):
        broker = make_broker()
        weights = Weights({})
        assert isinstance(weights._to_weights(broker), Weights)

    def test_to_weights_returns_copy(self):
        broker = make_broker()
        weights = Weights({})
        assert id(weights._to_weights(broker)) != id(weights)

    def test_to_weights_is_nr_contracts(self):
        """Allocation is already expressed as quantity, to self is returned."""
        broker = make_broker(
            initial_holdings={
                Cash(): -20,
                ETF('SPY'): 80,
                ETF('IEF'): -10,
                ES(2020, 9): 0.001,
            }
        )
        allocation = Weights(
            mapping={
                Cash(): -10,
                ETF('SPY'): 20,
                VX(2020, 3): -5,
                FutureChain(contracts=[ES(2020, 3), ES(2020, 6), ES(2020, 9)]): 2.2,
            },
        )
        actual = allocation._to_weights(broker)
        expected = {ETF('SPY'): 20, VX(2020, 3): -5, ES(2020, 3): 2.2}
        assert actual == expected

    def test_equality(self):
        broker = make_broker()
        weights = Weights({ETF('SPY'): 1})
        assert weights._to_weights(broker) == weights

    @pytest.mark.parametrize(
        argnames=['holdings'],
        argvalues=[
            [{Cash(): 20, ETF('SPY'): 80}],
            [{Cash(): -20, ETF('SPY'): 80}],
            [{Cash(): -20, ETF('SPY'): 80, ETF('IEF'): 10}],
            [{Cash(): -20, ETF('IEF'): 10}],
        ],
    )
    def test_to_nr_contracts_does_not_depend_on_broker_holdings(self, holdings):
        """Note that .as_quantity is not responsible for applying any filter,
        but just to convert the unit measure to whatever. Here we test only for
        what .as_quantity is responsible for."""
        broker = make_broker(initial_holdings=holdings)
        weights = Weights({ETF('SPY'): 1})
        actual = weights._to_nr_contracts(broker)
        acq_price = broker.exchange[ETF('SPY')].acq_price(1)
        expected = {ETF('SPY'): broker.net_liquidation_value() / acq_price}
        assert actual == expected

    def test_to_nr_contracts_round_trip(self):
        broker = make_broker()
        weights_start = Weights(
            mapping={
                ETF('SPY'): 0.8,
                VX(2020, 3): 0.5,
                FutureChain(contracts=[ES(2020, 3), ES(2020, 6), ES(2020, 9)]): -0.3,
            },
        )
        nr_contracts = weights_start._to_nr_contracts(broker)
        weights_end = nr_contracts._to_weights(broker)
        assert weights_start == weights_end

    @pytest.mark.parametrize(
        argnames=['allocation', 'expected'],
        argvalues=[
            Pair(
                allocation={
                    Cash(): -0.1,
                    ETF('SPY'): 0.2,
                    VX(2020, 3): -0.5,
                    FutureChain(contracts=[ES(2020, 3), ES(2020, 6), ES(2020, 9)]): 2.2,
                },
                expected={
                    ETF('SPY'): 0.6820119352088662,
                    VX(2020, 3): -0.026041666666666668,
                    ES(2020, 3): 0.01495581237253569,
                },
            ),
            Pair(
                allocation={
                    Cash(): 0.1,
                    ETF('SPY'): 0.2,
                    VX(2020, 3): -0.5,
                    FutureChain(contracts=[ES(2020, 3), ES(2020, 6), ES(2020, 9)]): 2.2,
                },
                expected={
                    ETF('SPY'): 0.6820119352088662,
                    VX(2020, 3): -0.026041666666666668,
                    ES(2020, 3): 0.01495581237253569,
                },
            ),
            Pair(
                allocation={
                    Cash(): 0.1,
                    ETF('SPY'): 0.2,
                    VX(2020, 3): 0.5,
                    FutureChain(contracts=[ES(2020, 3), ES(2020, 6), ES(2020, 9)]): 2.2,
                },
                expected={
                    ETF('SPY'): 0.6820119352088662,
                    VX(2020, 3): 0.025974025974025972,
                    ES(2020, 3): 0.01495581237253569,
                },
            ),
            Pair(
                allocation={
                    Cash(): -0.1,
                    ETF('SPY'): -0.2,
                    VX(2020, 3): -0.5,
                    FutureChain(contracts=[ES(2020, 3), ES(2020, 6), ES(2020, 9)]): -2.2,
                },
                expected={
                    ETF('SPY'): -0.6820351930159596,
                    VX(2020, 3): -0.026041666666666668,
                    ES(2020, 3): -0.014957083368743096,
                },
            ),
            Pair(
                allocation={
                    Cash(): -0.1,
                    ETF('SPY'): -0.2,
                    VX(2020, 3): 0,
                    FutureChain(contracts=[ES(2020, 3), ES(2020, 6), ES(2020, 9)]): -2.2,
                },
                expected={
                    ETF('SPY'): -0.6820351930159596,
                    ES(2020, 3): -0.014957083368743096,
                },
            ),
            Pair(
                allocation={
                    Cash(): 0,
                    ETF('SPY'): -0.2,
                    VX(2020, 3): 0,
                    FutureChain(contracts=[ES(2020, 3), ES(2020, 6), ES(2020, 9)]): -2.2,
                },
                expected={
                    ETF('SPY'): -0.6820351930159596,
                    ES(2020, 3): -0.014957083368743096,
                },
            ),
            Pair(
                allocation={
                    Cash(): 0,
                    ETF('SPY'): 0,
                    VX(2020, 3): 0,
                    FutureChain(contracts=[ES(2020, 3), ES(2020, 6), ES(2020, 9)]): 0,
                },
                expected={},
            ),
            Pair(
                allocation={
                    Cash(): 2,
                    ETF('SPY'): 0,
                    VX(2020, 3): 0,
                    FutureChain(contracts=[ES(2020, 3), ES(2020, 6), ES(2020, 9)]): 0,
                },
                expected={},
            ),
        ]
    )
    def test_to_nr_contracts_conversion(self, allocation, expected):
        """Note that .as_quantity is not responsible for applying any filter,
        but just to convert the unit measure to whatever. Here we test only for
        what .as_quantity is responsible for."""
        nlv = 1000
        broker = make_broker(deposit=nlv)
        allocation = Weights(allocation)
        actual = allocation._to_nr_contracts(broker)
        assert actual == expected

        # For documentation, it follows the logic to calculate expected.
        expected = {}
        for contract in [ETF('SPY'), VX(2020, 3), ES(2020, 3)]:
            weight = allocation.get(contract, 0)
            acq_price = broker.exchange[contract].acq_price(weight)
            multiplier = contract.multiplier
            if weight != 0:
                expected[contract] = weight * nlv / acq_price / multiplier
        assert actual == expected


class TestNrContracts:
    def test_to_nr_contracts_returns_nr_contracts_instance(self):
        broker = make_broker()
        nr_contracts = NrContracts({})
        assert isinstance(nr_contracts._to_nr_contracts(broker), NrContracts)

    def test_to_weights_returns_weights_instance(self):
        broker = make_broker()
        nr_contracts = NrContracts({})
        assert isinstance(nr_contracts._to_weights(broker), Weights)

    def test_to_nr_contracts_returns_copy(self):
        broker = make_broker()
        nr_contracts = NrContracts({})
        assert id(nr_contracts._to_nr_contracts(broker)) != id(nr_contracts)

    def test_to_nr_contracts_is_nr_contracts(self):
        """Allocation is already expressed as quantity, to self is returned."""
        broker = make_broker(
            initial_holdings={
                Cash(): -20,
                ETF('SPY'): 80,
                ETF('IEF'): -10,
                ES(2020, 9): 0.001,
            }
        )
        allocation = NrContracts(
            mapping={
                Cash(): -10,
                ETF('SPY'): 20,
                VX(2020, 3): -5,
                FutureChain(contracts=[ES(2020, 3), ES(2020, 6), ES(2020, 9)]): 2.2,
            },
        )
        actual = allocation._to_nr_contracts(broker)
        expected = {ETF('SPY'): 20, VX(2020, 3): -5, ES(2020, 3): 2.2}
        assert actual == expected

    def test_equality(self):
        broker = make_broker()
        nr_contracts = NrContracts({ETF('SPY'): 1})
        assert nr_contracts._to_nr_contracts(broker) == nr_contracts

    def test_to_weights_round_trip(self):
        broker = make_broker()
        nr_contracts_start = NrContracts(
            mapping={
                ETF('SPY'): 2,
                VX(2020, 3): 5.1,
                FutureChain(contracts=[ES(2020, 3), ES(2020, 6), ES(2020, 9)]): -0.3,
            },
        )
        weights = nr_contracts_start._to_weights(broker)
        nr_contracts_end = weights._to_nr_contracts(broker)
        assert nr_contracts_start == nr_contracts_end
