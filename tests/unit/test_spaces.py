from tradingenv.spaces import (
    Set,
    Float,
    PortfolioSpace,
    DiscretePortfolio,
    BoxPortfolio,
)
from tradingenv.broker.rebalancing import Weights
from tradingenv.broker.rebalancing import Rebalancing
from tradingenv.contracts import Cash, ETF, ES
from tradingenv.contracts import Future, FutureChain
from tradingenv.contracts import FutureChain, ES
from gym.spaces import Space, Discrete, Box
from datetime import datetime
import pytest
import numpy as np
import gym


try:
    from ray.rllib.models.extra_spaces import Simplex
    RAY_NOT_INSTALLED = False
except ModuleNotFoundError:
    RAY_NOT_INSTALLED = True


class TestSet:
    def test_parent_is_gym_space(self):
        assert issubclass(Set, gym.Space)

    def test_initialisation(self):
        space = Set(-1, 1)
        assert space.items == (-1, 1)

    def test_sample(self):
        space = Set(-1, 1)
        assert space.sample() in (-1, 1)

    def test_sample_distribution_is_uniform(self):
        space = Set(-1, 1)
        samples = np.array([space.sample() for _ in range(10000)])
        assert -0.05 < samples.mean() < 0.05

    def test_sample_mixed_dtypes(self):
        space = Set('a', 1)
        samples = [space.sample() for _ in range(20)]
        assert 'a' in samples
        assert 1 in samples
        assert '1' not in samples

    def test_contains(self):
        space = Set(-1, 1)
        assert space.contains(-1)
        assert space.contains(1)
        assert not space.contains(0)


class TestFloat:
    def test_parent_is_gym_space(self):
        assert issubclass(Float, gym.Space)

    def test_initialisation(self):
        low = -1
        high = 1
        space = Float(low, high)
        assert space.low == low
        assert space.high == high

    def test_default_initialization(self):
        space = Float()
        assert space.low == -1e16
        assert space.high == 1e16

    def test_sample_distribution_is_unbiased(self):
        low = -1
        high = 1
        space = Float(low=low, high=high)
        samples = np.array([space.sample() for _ in range(10000)])
        assert -0.05 < samples.mean() < 0.05

    def test_sample_is_within_bounds(self):
        low = -1
        high = 1
        space = Float(low=low, high=high)
        samples = [space.sample() for _ in range(10000)]
        assert min(samples) > -1
        assert max(samples) < 1

    def test_contains(self):
        low = -1
        high = 1
        space = Float(low=low, high=high)
        assert not space.contains(-1.01)
        assert space.contains(-1)
        assert space.contains(1)
        assert not space.contains(1.01)

    def test_sample_from_default_bounds(self):
        # Make sure that the following error is not raised:
        #  OverflowError: Range exceeds valid bounds
        space = Float()
        space.sample()


class TestPortfolioSpace:
    def test_make_allocation_is_abstract_method(self):
        space = PortfolioSpace([Cash(), ETF("IEF")])
        with pytest.raises(NotImplementedError):
            space._make_allocation(0)

    def test_default_initialization(self):
        class MockPortfolioSpace(PortfolioSpace, Space):
            def _make_allocation(self, action: np.ndarray, broker=None):
                return np.ones((len(self.contracts),))

        space = MockPortfolioSpace([Cash(), ETF("IEF")])
        assert space.contracts == [Cash(), ETF("IEF")]
        assert space._as_weights is True
        assert space._fractional is True
        assert space._margin == 0
        assert space.base_currency == Cash()

    def test_custom_initialization(self):
        class MockPortfolioSpace(PortfolioSpace, Space):
            def _make_allocation(self, action: np.ndarray, broker=None):
                return np.ones((len(self.contracts),))

        space = MockPortfolioSpace([Cash('EUR'), ETF("IEF")], True, False, 0.02)
        assert space.contracts == [Cash('EUR'), ETF("IEF")]
        assert space._as_weights is True
        assert space._fractional is False
        assert space._margin == 0.02
        assert space.base_currency == Cash('EUR')

    def test_raises_if_contracts_are_not_all_abstract_contracts(self):
        class MockPortfolioSpace(PortfolioSpace, Space):
            def _make_allocation(self, action: np.ndarray, broker=None):
                return np.ones((len(self.underlyings),))

        match = "'contracts' must be a sequence of AbstractContract instances."
        with pytest.raises(TypeError, match=match):
            MockPortfolioSpace(["SPY", ETF("IEF")])

    def test_raises_if_all_contracts_are_not_unique(self):
        class MockPortfolioSpace(PortfolioSpace, Space):
            def _make_allocation(self, action: np.ndarray, broker=None):
                return np.ones((len(self.underlyings),))

        match = "All items in 'contracts' must be unique."
        with pytest.raises(ValueError, match=match):
            MockPortfolioSpace([Cash(), Cash()])

    def test_make_rebalancing_request(self):
        class MockPortfolioSpace(PortfolioSpace, Discrete):
            def __init__(self, contracts, allocations):
                PortfolioSpace.__init__(self, contracts)
                Discrete.__init__(self, n=len(allocations))

            def _make_allocation(self, action: np.ndarray, broker=None):
                return np.ones((len(self.contracts),))

        space = MockPortfolioSpace([Cash(), ETF("IEF")], [[1, 0], [0, 1]])
        action = 0

        actual = space.make_rebalancing_request(action)
        expected = Rebalancing(
            space.contracts, np.ones((2,)), 'weight', True, True, 0, datetime.now()
        )
        assert isinstance(actual.allocation, Weights)
        assert actual.allocation == expected.allocation
        assert actual.absolute is expected.absolute
        assert actual.fractional is expected.fractional
        assert actual.margin == expected.margin

    def test_make_rebalancing_request_if_shortselling_shortable(self):
        class MockPortfolioSpace(PortfolioSpace, Discrete):
            def __init__(self, contracts, allocations):
                self.allocations = allocations
                PortfolioSpace.__init__(self, contracts)
                Discrete.__init__(self, n=len(allocations))

            def _make_allocation(self, action: np.ndarray, broker=None):
                return np.array(self.allocations[action])

        space = MockPortfolioSpace([Cash(), ES(2019, 6)], [[1.1, -0.1]])
        action = 0
        space.make_rebalancing_request(action)  # does not raise

    def test_make_rebalancing_request_raises_if_invalid_action(self):
        class MockPortfolioSpace(PortfolioSpace, Discrete):
            def __init__(self, contracts, allocations):
                PortfolioSpace.__init__(self, contracts)
                Discrete.__init__(self, n=len(allocations))

            def _make_allocation(self, action: np.ndarray, broker=None):
                return np.ones((len(self.contracts),))

        contracts = [Cash(), ETF("IEF")]
        allocations = [[1, 0], [0, 1]]
        space = MockPortfolioSpace(contracts, allocations)
        action = 42
        with pytest.raises(ValueError):
            # Only 0 and 1 are valid (2 actions)
            space.make_rebalancing_request(action)

    def test_raises_if_margin_is_negative(self):
        class MockPortfolioSpace(PortfolioSpace, Space):
            def _make_allocation(self, action: np.ndarray, broker=None):
                return np.ones((len(self.contracts),))

        with pytest.raises(ValueError):
            MockPortfolioSpace([Cash(), ETF("IEF")], margin=-0.01)

    def test_underlyings(self):
        class MockPortfolioSpace(PortfolioSpace, Space):
            def _make_allocation(self, action: np.ndarray, broker=None):
                return np.ones((len(self.contracts),))

            def __contains__(self, item):
                # NotImplementedError otherwise.
                return True


        contracts = [Cash(), ETF("IEF"), FutureChain(ES, "2018-12", "2019-07")]
        space = MockPortfolioSpace(contracts)
        np.testing.assert_equal(
            actual=space.contracts,
            desired=np.array(
                [Cash(), ETF("IEF"), FutureChain(ES, "2018-12", "2019-07")]
            ),
        )

    def test_with_leading_future(self):
        # This is actually an integration test.
        class MockPortfolioSpace(PortfolioSpace, Space):
            def _make_allocation(self, action: np.ndarray, broker=None):
                return action

            def __contains__(self, item):
                # NotImplementedError otherwise.
                return True

        class ES(Future):
            multiplier = 50.0
            freq = "Q-DEC"

            def _get_expiry_date(self, year: int, month: int) -> datetime:
                return datetime(year, month, 18)

            def _get_last_trading_date(self, expiry: datetime) -> datetime:
                return datetime(expiry.year, expiry.month, 8)

            def margin_requirement(self) -> float:
                return 0.1

        cash = Cash()
        leading_future = FutureChain(ES, "2019-01-01", "2019-12-31")
        contracts = [cash, leading_future]
        space = MockPortfolioSpace(contracts)
        action = np.array([0.25, 0.75])
        request = space.make_rebalancing_request(action)
        assert request.allocation == {
            leading_future.static_hashing(): 0.75
        }

    def test_cash_is_not_added_to_contracts_if_missing(self):
        class MockPortfolioSpace(PortfolioSpace, Space):
            def _make_allocation(self, action: np.ndarray, broker=None):
                return action

        contracts = [ETF("SPY"), ETF("IEF")]
        space = MockPortfolioSpace(contracts)
        assert space.contracts == [ETF("SPY"), ETF("IEF")]

    def test_make_rebalancing_request_when_cash_missing_from_contracts(self):
        class MockPortfolioSpace(PortfolioSpace, Space):
            def _make_allocation(self, action: np.ndarray, broker=None):
                return action

            def __contains__(self, item):
                # NotImplementedError otherwise.
                return True

        contracts = [ETF("SPY"), ETF("IEF")]
        space = MockPortfolioSpace(contracts)
        action = np.array([0.25, 0.75])
        rebalancing_request = space.make_rebalancing_request(action)
        assert rebalancing_request.allocation == {
            ETF("SPY"): 0.25,
            ETF("IEF"): 0.75,
        }

    def test_make_rebalancing_request_when_cash_not_first_item_of_contracts(self):
        class MockPortfolioSpace(PortfolioSpace, Space):
            def _make_allocation(self, action: np.ndarray, broker=None):
                return action

            def __contains__(self, item):
                # NotImplementedError otherwise.
                return True

        contracts = [ETF("SPY"), Cash(), ETF("IEF")]
        space = MockPortfolioSpace(contracts)
        action = np.array([0.25, 0.0, 0.75])
        rebalancing_request = space.make_rebalancing_request(action)
        assert rebalancing_request.allocation == {
            ETF("SPY"): 0.25,
            ETF("IEF"): 0.75,
        }


class TestDiscretePortfolio:
    def test_parent_classes(self):
        assert issubclass(DiscretePortfolio, PortfolioSpace)
        assert issubclass(DiscretePortfolio, Discrete)

    def test_initialization(self, mocker):
        mocker.patch.object(PortfolioSpace, "__init__", return_value=None)
        mocker.patch.object(Discrete, "__init__", return_value=None)

        contracts = [Cash(), ETF("IEF")]
        allocations = [[1, 0], [0.5, 0.5], [0, 1]]
        space = DiscretePortfolio(contracts, allocations)

        PortfolioSpace.__init__.assert_called_once_with(space, contracts, True, True)
        Discrete.__init__.assert_called_once_with(space, n=len(allocations))

    def test_custom_initialization(self):
        contracts = [Cash(), ETF("IEF")]
        allocations = [[1, 0], [0.5, 0.5], [0, 1]]
        space = DiscretePortfolio(
            contracts, allocations, as_weights=False, fractional=False
        )
        assert space._as_weights is False
        assert space._fractional is False

    def test_allocations_with_different_lengths_raise(self):
        contracts = [Cash(), ETF("IEF")]
        allocations = [[1, 0], [0.5, 0.5, 0.5], [0, 1]]
        with pytest.raises(ValueError):
            DiscretePortfolio(contracts, allocations)

    def test_contains_when_false(self):
        contracts = [Cash(), ETF("IEF")]
        allocations = [[1, 0], [0.5, 0.5], [0, 1]]
        space = DiscretePortfolio(contracts, allocations)
        invalid_item = 42
        assert not space.contains(invalid_item)

    def test_contains_when_true(self):
        contracts = [Cash(), ETF("IEF")]
        allocations = [[1, 0], [0.5, 0.5], [0, 1]]
        space = DiscretePortfolio(contracts, allocations)
        valid_item = 1
        assert space.contains(valid_item)

    def test_sample_returns_an_integer(self):
        contracts = [Cash(), ETF("IEF")]
        allocations = [[1, 0], [0.5, 0.5], [0, 1]]
        space = DiscretePortfolio(contracts, allocations)
        assert isinstance(space.sample(), int)

    def test_sample_items_from_space(self):
        contracts = [Cash(), ETF("IEF")]
        allocations = [[1, 0], [0.5, 0.5], [0, 1]]
        space = DiscretePortfolio(contracts, allocations)
        nr_samples = 1000
        for action in [space.sample() for _ in range(nr_samples)]:
            assert action in space

    def test_sample_distribution_is_unbiased(self):
        contracts = [Cash(), ETF("IEF")]
        allocations = [[1, 0], [0.5, 0.5], [0, 1]]
        space = DiscretePortfolio(contracts, allocations)
        nr_samples = 100000
        samples = [space.sample() for _ in range(nr_samples)]
        _, counts = np.unique(samples, return_counts=True, axis=0)
        rel_freq = counts / nr_samples
        assert max(rel_freq) - min(rel_freq) < 0.01

    def test_make_rebalancing_request(self):
        contracts = [Cash(), ETF("IEF"), ETF("GSG")]
        allocations = [[1, 0, 0], [0.7, 0.1, 0.2], [0.3, 0.6, 0.1]]
        space = DiscretePortfolio(contracts, allocations)

        actual = space.make_rebalancing_request(0).allocation
        assert actual == {}

        actual = space.make_rebalancing_request(1).allocation
        assert actual == {ETF('IEF'): 0.1, ETF('GSG'): 0.2}

        actual = space.make_rebalancing_request(2).allocation
        assert actual == {ETF('IEF'): 0.6, ETF('GSG'): 0.1}

    def test_make_rebalancing_request_when_invalid_index(self):
        contracts = [Cash(), ETF("IEF")]
        allocations = [[0, 0], [0.1, 0.2], [0.6, 0.1]]
        space = DiscretePortfolio(contracts, allocations)
        with pytest.raises(ValueError):
            space.make_rebalancing_request(-1)
        with pytest.raises(ValueError):
            space.make_rebalancing_request(3)

    def test_make_rebalancing_request_expands_weights_for_composite_futures(self):
        allocations = [[1, 0, 0], [0.7, 0.1, 0.2], [0.3, 0.6, 0.1]]
        space = DiscretePortfolio(
            contracts=[Cash(), ETF("IEF"), FutureChain(ES, "2018-12", "2019-07")],
            allocations=allocations,
        )
        rebalancing = space.make_rebalancing_request(2)
        assert rebalancing.allocation == {
            ETF("IEF"): 0.6,
            ES(2018, 12): 0.1,
        }

    def test_make_allocation(self):
        contracts = [Cash(), ETF("IEF"), ETF("GSG")]
        allocations = [[1, 0, 0], [0.7, 0.1, 0.2], [0.3, 0.6, 0.1]]
        space = DiscretePortfolio(contracts, allocations)
        assert space._make_allocation(0) == [1, 0, 0]
        assert space._make_allocation(1) == [0.7, 0.1, 0.2]
        assert space._make_allocation(2) == [0.3, 0.6, 0.1]

    def test_make_allocation_with_leading_future(self):
        """It's responsibility of observation_space._make_rebalancing_request to unpack
        the allocation vector."""
        contracts = [Cash(), ETF("IEF"), FutureChain(ES, "2018-12", "2019-07")]
        allocations = [[1, 0, 0], [0.7, 0.1, 0.2], [0.3, 0.6, 0.1]]
        space = DiscretePortfolio(contracts, allocations)
        assert space._make_allocation(0) == [1, 0, 0]
        assert space._make_allocation(1) == [0.7, 0.1, 0.2]
        assert space._make_allocation(2) == [0.3, 0.6, 0.1]


class TestBoxPortfolio:
    def test_parent_classes(self):
        assert issubclass(BoxPortfolio, PortfolioSpace)
        assert issubclass(BoxPortfolio, Box)

    def test_initialization(self, mocker):
        mocker.patch.object(PortfolioSpace, "__init__")
        mocker.patch.object(Box, "__init__")
        mocker.patch.object(Box, "__repr__")

        # To avoid TypeError: __init__() should return None, not 'MagicMock'
        PortfolioSpace.__init__.return_value = None
        Box.__init__.return_value = None
        Box.__repr__.return_value = 'Box'

        contracts = [Cash(), ETF("IEF")]
        space = BoxPortfolio(contracts)

        PortfolioSpace.__init__.assert_called_once_with(
            space, contracts, True, True, 0.0
        )
        Box.__init__.assert_called_once_with(
            space, 0., 1., (len(contracts), ), np.float64
        )

    def test_custom_initialization(self, mocker):
        mocker.patch.object(PortfolioSpace, "__init__")
        mocker.patch.object(Box, "__init__")

        # To avoid TypeError: __init__() should return None, not 'MagicMock'
        PortfolioSpace.__init__.return_value = None
        Box.__init__.return_value = None

        contracts = [Cash(), ETF("IEF")]
        space = BoxPortfolio(
            contracts=contracts,
            low=-1.1,
            high=2.2,
            as_weights=False,
            fractional=False,
            margin=0.02,
        )

        PortfolioSpace.__init__.assert_called_once_with(
            space, contracts, False, False, 0.02
        )
        Box.__init__.assert_called_once_with(space, -1.1, 2.2, (2, ), np.float64)

    def test_contains_is_true_even_when_sum_is_not_one(self):
        contracts = [Cash(), ETF("IEF")]
        space = BoxPortfolio(contracts)
        valid_item = np.array([0.25, 0.5])
        assert space.contains(valid_item)
        assert valid_item in space

    def test_contains_when_true_with_array(self):
        contracts = [Cash(), ETF("IEF")]
        space = BoxPortfolio(contracts)
        valid_item = np.array([0.5, 0.5])
        assert space.contains(valid_item)
        assert valid_item in space

    def test_sample_belongs_to_space(self):
        contracts = [Cash(), ETF("IEF")]
        space = BoxPortfolio(contracts)
        nr_samples = 1000
        for action in [space.sample() for _ in range(nr_samples)]:
            assert action in space

    def test_sample_distribution_is_unbiased(self):
        contracts = [Cash(), ETF("IEF")]
        space = BoxPortfolio(contracts)
        nr_samples = 100000
        samples = [space.sample() for _ in range(nr_samples)]
        mean = np.mean(samples, axis=0)
        np.testing.assert_allclose(mean, np.array([0.5, 0.5]), rtol=0.025)

    def test_make_rebalancing_request(self):
        contracts = [Cash(), ETF("IEF"), ETF("GSG")]
        space = BoxPortfolio(contracts)
        action = np.array([0.7, 0.1, 0.2])
        rebalancing_request = space.make_rebalancing_request(action)
        actual = rebalancing_request.allocation
        expected = {
            ETF("IEF"): 0.1,
            ETF("GSG"): 0.2,
        }
        assert actual == expected

    def test_make_rebalancing_request_action_does_not_have_to_be_array(self):
        contracts = [Cash(), ETF("IEF"), ETF("GSG")]
        space = BoxPortfolio(contracts)
        # Action as list instead of np.ndarray. Does not raise.
        space.make_rebalancing_request([0.25, 0.25, 0.5])

    def test_make_rebalancing_request_expands_weights_for_composite_futures(self):
        contracts = [Cash(), ETF("IEF"), FutureChain(ES, "2018-12", "2019-07")]
        space = BoxPortfolio(contracts)
        action = np.array([0.25, 0.15, 0.6])
        rebalancing = space.make_rebalancing_request(action)
        actual = rebalancing.allocation
        expected = {
            ETF("IEF"): 0.15,
            ES(2018, 12): 0.6,
        }
        assert actual == expected
        assert isinstance(list(actual.keys())[1], ES)
