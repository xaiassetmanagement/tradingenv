from tradingenv.broker.allocation import _Allocation, NrContracts, Weights
from tradingenv.contracts import Cash, ETF, VX, ES, FutureChain
import pytest
from collections import namedtuple
_TestSubtraction = namedtuple('TestSubtraction', ['right', 'expected'])


class TestAllocation:
    def test_subclasses_dict(self):
        assert issubclass(_Allocation, dict)

    def test_initialization_with_mapping(self):
        allocation = _Allocation({ETF('SPY'): 1, ETF('IEF'): 2})
        assert allocation == {ETF('SPY'): 1, ETF('IEF'): 2}

    def test_initialization_with_keys_and_values(self):
        allocation = _Allocation(keys=[ETF('SPY'), ETF('IEF')], values=[1, 2])
        assert allocation == {ETF('SPY'): 1, ETF('IEF'): 2}

    def test_initialization_raises_if_more_values_than_keys(self):
        with pytest.raises(ValueError):
            _Allocation(keys=[ETF('SPY'), ETF('IEF')], values=[1, 2, 3])

    def test_initialization_raises_if_more_keys_than_values(self):
        with pytest.raises(ValueError):
            _Allocation(keys=[Cash(), ETF('SPY'), ETF('IEF')], values=[1, 2])

    def test_keys_have_static_hashing(self):
        future_chain = FutureChain(contracts=[ES(2020, 3), ES(2020, 6)])
        allocation = _Allocation({future_chain: 0.2})
        assert allocation == {ES(2020, 3): 0.2}
        key = list(allocation.keys())[0]
        assert not isinstance(key, FutureChain)
        assert isinstance(key, ES)

    def test_keys_whose_value_is_zero_are_dropped(self):
        allocation = _Allocation({ETF('SPY'): 1, ETF('IEF'): 0})
        assert allocation == {ETF('SPY'): 1}

    def test_keys_with_cash_are_dropped(self):
        allocation = _Allocation({ETF('SPY'): 1, Cash(): 2})
        assert allocation == {ETF('SPY'): 1}

    def test_subtraction_raises_if_different_measure(self):
        left = Weights({ETF('SPY'): -0.4})
        right = NrContracts({ETF('SPY'): -0.4})
        with pytest.raises(TypeError):
            left - right

    def test_subtraction_does_not_mutate_objects(self):
        left = _Allocation({ETF('SPY'): 1})
        right = _Allocation({ETF('SPY'): 1})
        actual = left - right
        assert id(actual) != id(left)
        assert id(actual) != id(right)
        assert left == {ETF('SPY'): 1}
        assert right == {ETF('SPY'): 1}

    @pytest.mark.parametrize('measure', ['nr-contracts', 'weight'])
    def test_subtraction_when_weights_returns_weights(self, measure):
        actual = Weights() - Weights()
        assert isinstance(actual, Weights)

    @pytest.mark.parametrize('measure', ['nr-contracts', 'weight'])
    def test_subtraction_when_nr_contracts_returns_nr_contracts(self, measure):
        actual = NrContracts() - NrContracts()
        assert isinstance(actual, NrContracts)

    @pytest.mark.parametrize(
        argnames=['right', 'expected'],
        argvalues=[
            _TestSubtraction(
                right={
                    Cash(): 0.5,
                    ETF('SPY'): -0.4,
                    ETF('IEF'): 0.1,
                    VX(2019, 3): 0.02,
                    VX(2019, 4): -0.01,
                    FutureChain(contracts=[ES(2020, 3), ES(2020, 6)]): 0.2,
                },
                expected={},
            ),
            _TestSubtraction(
                right={},
                expected={
                    Cash(): 0.5,
                    ETF('SPY'): -0.4,
                    ETF('IEF'): 0.1,
                    VX(2019, 3): 0.02,
                    VX(2019, 4): -0.01,
                    FutureChain(contracts=[ES(2020, 3), ES(2020, 6)]): 0.2,
                },
            ),
            _TestSubtraction(
                right={
                    ETF('SPY'): -0.4,
                    ETF('EEM'): 0.2,
                    VX(2019, 3): 0.015,
                },
                expected={
                    ETF('IEF'): 0.1,
                    ETF('EEM'): -0.2,
                    VX(2019, 3): 0.02 - 0.015,
                    VX(2019, 4): -0.01,
                    ES(2020, 3): 0.2,
                },
            ),
        ]
    )
    def test_subtraction(self, right, expected):
        future_chain = FutureChain(contracts=[ES(2020, 3), ES(2020, 6)])
        left = _Allocation({
            Cash(): 0.5,
            ETF('SPY'): -0.4,
            ETF('IEF'): 0.1,
            VX(2019, 3): 0.02,
            VX(2019, 4): -0.01,
            future_chain: 0.2,
        })
        actual = left - _Allocation(right)
        assert actual == _Allocation(expected)
        assert isinstance(actual, _Allocation)
