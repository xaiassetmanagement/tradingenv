"""To create custom action spaces inherit PortfolioSpace and Space (or one of
its subclasses, e.g. Discrete, Box, ...) and implement
PortfolioSpace._make_rebalancing_request. See examples below."""
import tradingenv
from tradingenv.broker.rebalancing import Rebalancing
from tradingenv.contracts import AbstractContract, Cash
from gym.spaces import Space, Discrete, Box
from typing import Sequence, Union
from datetime import datetime
import numpy as np
import random


class Set(Space):
    """Similar to gym.spaces.Discrete. However, Set is more flexible because
    it allows to define a set of arbitrary values instead of counting from
    zero."""

    def __init__(self, *args):
        """
        Parameters
        ----------
        *args
            A sequence of items which must belong to the set being constructed.

        Examples
        --------
        >>> from tradingenv.spaces import Set
        >>> set_ = Set('a', 1, 2)
        >>> set_.contains(set_.sample())
        True
        """
        super().__init__(shape=None, dtype=None)
        self.items = args

    def __repr__(self) -> str:
        return str(self.items)

    def sample(self) -> float:
        """Uniformly randomly sample a random element of this observation_space."""
        return random.choice(self.items)

    def contains(self, item: float) -> bool:
        """Return boolean specifying if x is a valid member of this observation_space."""
        return item in self.items


class Float(Space):
    """Similar to gym.spaces.Box. However, Set is more flexible because
    it allows to define a set of arbitrary values instead of counting from
    zero."""

    def __init__(self, low: float=-1e16, high: float=1e16):
        """
        Parameters
        ----------
        low : float
            Lower bound of the interval of real numbers.
        high : float
            Upper bound of the interval of real numbers.

        Examples
        --------
        >>> from tradingenv.spaces import Float
        >>> real_interval = Float(-1, 1)
        >>> real_interval.contains(real_interval.sample())
        True
        """
        super().__init__(shape=(), dtype=np.float32)
        self.low = low
        self.high = high

    def __repr__(self) -> str:  # pragma: no cover
        return '{}({}, {})'.format(self.__class__.__name__, self.low, self.high)

    def sample(self) -> float:
        """Uniformly randomly sample a random element of this observation_space."""
        return np.random.uniform(low=self.low, high=self.high)

    def contains(self, item: float) -> bool:
        """Return boolean specifying if item is a valid member of this observation_space."""
        return self.low <= item <= self.high


class PortfolioSpace(Space):
    """All action spaces in TradingEnv must be an implementation of this
    abstract class. Actions are parsed to portfolio RebalancingRequest before
    being executed by Broker."""

    def __init__(
        self,
        contracts: Sequence[AbstractContract],
        as_weights: bool = True,
        fractional: bool = True,
        margin: float = 0.0,
    ):
        """
        Parameters
        ----------
        contracts : Sequence[AbstractContract]
            A sequence of AbstractContract implementations.
        as_weights : bool
            True by default. If True, the target allocation will be expressed
            in terms of target weights of the portfolio in terms of
            proportion of the notional value of the contract over the net
            liquidation value of the current positions. If False, the target
            allocation will be expressed in terms of target number of shares to
            be traded.
        fractional : bool
            True by default. If False, decimals from target positions will be
            dropped (e.g. 100.72 target shares for ETF(SPY) will become 100
            before being traded).
        margin : float
            If the target weight is not at least different by this amount
            from the current drift weight in absolute value, then the asset
            will not be rebalanced.
        """
        Space.__init__(self)
        if not isinstance(self, Space):
            raise TypeError(
                "{} must be subclassed along with gym.spaces.Space or one "
                "of its subclasses.".format(self.__class__.__name__)
            )
        if not all(isinstance(c, AbstractContract) for c in contracts):
            raise TypeError(
                "'contracts' must be a sequence of AbstractContract instances."
            )
        if margin < 0:
            raise ValueError("Parameter '_margin' must be non-negative.")
        if len(contracts) != len(set(contracts)):
            # Note: this condition might be bypassed when using FutureChain in
            # conjunction with a Future with same underlying due to the dynamic
            # hashing of FutureChain.
            raise ValueError("All items in 'contracts' must be unique.")

        self.contracts = contracts
        self._as_weights = as_weights
        self._fractional = fractional
        self._margin = margin

        # Find base currency.
        self.base_currency = Cash()
        for currency in self.contracts:
            if isinstance(currency, Cash):
                self.base_currency = currency

    def null_action(self):
        """Used to fill the deque of actions when delay_steps > 0 during the
        initialisation of TradingEnv."""
        # TODO: test.
        return self.sample() * 0.

    def make_rebalancing_request(
        self, action, time: datetime = None, broker: 'tradingenv.broker.Broker' = None
    ) -> Rebalancing:
        """Returns an instance of RebalancingRequest. TradingEnv will pass
        this instance to Broker.rebalance in order to perform the portfolio
        _rebalancing. Broker can be useful to implement custom environment. For
        example, if action is intended to be a portfolio weight change, you
        will need broker to check current allocations."""
        if action not in self:
            raise ValueError(
                "This action does not belong to the action observation_space {}: {}"
                "".format(self.__class__.__name__, action)
            )
        return Rebalancing(
            time=time,
            contracts=self.contracts,
            allocation=self._make_allocation(action, broker),
            measure='weight' if self._as_weights else 'nr-contracts',
            fractional=self._fractional,
            margin=self._margin,
        )

    def _make_allocation(self, action, broker: 'tradingenv.broker.Broker' = None) -> Sequence[float]:
        """The returned allocation will become the allocation attribute of
        RebalancingRequest instanced by
        PortfolioSpace.make_rebalancing_request."""
        raise NotImplementedError()


class DiscretePortfolio(PortfolioSpace, Discrete):
    """Use this class to allow a predefined finite set of discrete portfolio
    allocations."""

    def __init__(
        self,
        contracts: Sequence[AbstractContract],
        allocations: Sequence[Sequence],
        as_weights: bool = True,
        fractional: bool = True,
    ):
        """
        Parameters
        ----------
        contracts : Sequence[AbstractContract]
            A sequence of AbstractContract implementations.
        allocations : Sequence[Sequence]
            A sequence of target allocations. The action correspond to the
            index of the target allocation from that sequence.
        as_weights : bool
            True by default. If True, the target allocation will be expressed
            in terms of target weights of the portfolio in terms of
            proportion of the notional value of the contract over the net
            liquidation value of the current positions. If False, the target
            allocation will be expressed in terms of target number of shares to
            be traded.
        fractional : bool
            True by default. If False, decimals from target positions will be
            dropped (e.g. 100.72 target shares for ETF(SPY) will become 100
            before being traded).
        """
        lengths = set(len(seq) for seq in allocations)
        if len(lengths) != 1:
            raise ValueError("All allocations must have the same length. ")
        self._allocations = allocations
        PortfolioSpace.__init__(self, contracts, as_weights, fractional)
        Discrete.__init__(self, n=len(allocations))

    def _make_allocation(self, action, broker: 'tradingenv.broker.Broker' = None) -> Sequence[float]:
        """The action correspond to the  index of the target allocation from
        that sequence."""
        return self._allocations[action]


class BoxPortfolio(PortfolioSpace, Box):
    def __init__(
        self,
        contracts: Sequence[AbstractContract],
        low: float=0.,
        high: float=1.,
        as_weights: bool = True,
        fractional: bool = True,
        margin: float = 0.0,
    ):
        """
        Parameters
        ----------
        contracts : Sequence[AbstractContract]
            A sequence of AbstractContract implementations.
        low
            The minimum allowed quantity for each contract in the portfolio.
        high
            The maximum allowed quantity for each contract in the portfolio.
        as_weights : bool
            True by default. If True, the target allocation will be expressed
            in terms of target weights of the portfolio in terms of
            proportion of the notional value of the contract over the net
            liquidation value of the current positions. If False, the target
            allocation will be expressed in terms of target number of shares to
            be traded.
        fractional : bool
            True by default. If False, decimals from target positions will be
            dropped (e.g. 100.72 target shares for ETF(SPY) will become 100
            before being traded).
        margin : float
            If the target weight is not at least different by this amount
            from the current drift weight in absolute value, then the asset
            will not be rebalanced.
        """
        PortfolioSpace.__init__(self, contracts, as_weights, fractional, margin)
        Box.__init__(self, low, high, (len(contracts),), np.float64)

    def _make_allocation(self, action: np.ndarray, broker: 'tradingenv.broker.Broker' = None) -> Sequence[float]:
        """Action is already supposed to be the target allocation, so this
        method simply returns the input (action)."""
        return action

    def contains(self, x):
        # Removed np.can_cast from super as it return false when action is
        # float64 and self.dtype is float32.
        if not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=self.dtype)
        return (
            x.shape == self.shape
            and np.all(x >= self.low)
            and np.all(x <= self.high)
        )
