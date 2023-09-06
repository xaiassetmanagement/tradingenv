"""Note: all transformers expect 2D arrays in feature.transformer.transform.
    ValueError: Expected 2D array, got 1D array instead
Therefore, all space shapes should really be 2D and parsed history 3D. However,
            'vx-term-structure-basis': mid_prices[1:] - self.exchange[Contracts.vix].mid_price,
            'vx-term-structure-roll-yield': np.array([term_structure.roll_yield(0, t) for t in lifespan[:2]]),
            'vx-roll-yield-30days': roll_yield_30d,
            'vx-implied-sharpe-30days': implied_sharpe,
            'time-since-vix-update': np.array([hours_since_vix_update]),
"""
from tradingenv.features import Feature
from tradingenv.contracts import AbstractContract
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Union, Sequence
import gym.spaces
import numpy as np
import exchange_calendars as xcals


def listify(item):
    """If the item is not iterable, insert the item in a list."""
    try:
        item = list(item)
    except TypeError:  # contracts is a single contract, not in an iterable
        item = [item]
    return item


class FeaturePortfolioWeight(Feature):
    """Portfolio weights of holdings."""

    def __init__(
            self,
            contracts: Union[AbstractContract, Sequence[AbstractContract]],
            low: float,
            high: float,
            name: str = None,
            transformer=None,
            total: bool = False,  # TODO: test
    ):
        """
        Parameters
        ----------
        contracts
            A contract or sequence of contracts.
        low
            Minimum possible portfolio allocation to the contract. E.g. -1.5
            for -150% (i.e. 150% short).
        high
            Minimum possible portfolio allocation to the contract. E.g. -1.5
            for -150% (i.e. 150% short).
        name
            Feature name. Default is class name if omitted.
        transformer
            A class from sklearn.preprocessing.

        Notes
        -----
        Because gym.spaces.Box forces the same low and high bounds on all
        items, you'll have to instance a different PortfolioWeight for each
        contract with different 'low' and 'high' allocation.
        """
        if transformer is None:
            transformer = MinMaxScaler((-1., 1.))
        self.contracts = listify(contracts)
        self.total = total
        self._size = 1 if total else len(self.contracts)
        super().__init__(
            space=gym.spaces.Box(low, high, (1, self._size), float),
            name=name,
            transformer=transformer,
        )

    def parse(self):
        """Returns array of currently held weights of self.contracts in the
        portfolio."""
        holdings = self.broker.holdings_weights()
        w = [float(holdings.get(contract, 0.)) for contract in self.contracts]
        if self.total:
            w = [sum(w)]
        return np.array([w])

    def _manual_fit_transformer(self):
        """Whatever the max or min allocation is per asset, either long or
        short, rescale the current portfolio weights in the [-1, +1] range."""
        x = np.concatenate([self.space.high, self.space.low])
        self.transformer.fit(x)


class FeaturePrices(Feature):
    """Feature representing price data."""

    def __init__(
            self,
            contracts: Union[AbstractContract, Sequence[AbstractContract]],
            low: float = 0.,
            high: float = np.inf,
            name: str = None,
            transformer=None,
    ):
        """
        Parameters
        ----------
        contracts
            A contract or sequence of contracts.
        name
            Feature name. Default is class name if omitted.
        transformer
            A class from sklearn.preprocessing.
        """
        if transformer is None:
            transformer = StandardScaler()
        self.contracts = listify(contracts)
        super().__init__(
            space=gym.spaces.Box(low, high, (1, len(self.contracts)), float),
            name=name,
            transformer=transformer,
        )

    def parse(self):
        """Parse sequence of prices."""
        w = [self.exchange[contract].mid_price for contract in self.contracts]
        return np.array([w])


class FeatureSpread(Feature):
    """Feature representing prices spreads."""

    def __init__(
            self,
            contracts: Union[AbstractContract, Sequence[AbstractContract]],
            clip: float = 0.01,
            name: str = None,
            transformer=None,
            size: int = None
    ):
        """
        Parameters
        ----------
        contracts
            A contract or sequence of contracts.
        clip
            Spread (%) greater than this are clipped. By default this is
            0.01 (i.e. 1%).
        name
            Feature name. Default is class name if omitted.
        transformer
            A class from sklearn.preprocessing. Note that spread can be highly
            positively skewed unless clipped or log transformed if the market
            data include overnight sessions, illiquid or emerging assets, in
            which case you might want to log the spread or use a more
            sophisticated transformer to normalise the data.
        """
        if transformer is None:
            transformer = MinMaxScaler(feature_range=(-1, 1))
        self.contracts = listify(contracts)
        self._low = 0.
        self._high = clip
        self._size = size or len(self.contracts)
        super().__init__(
            space=gym.spaces.Box(self._low, self._high, (1, self._size), float),
            name=name,
            transformer=transformer,
        )

    def _make_weights(self):
        """Returned array must be of length len(self.contracts).
        For example, if you want to equal weights the spread of two contracts
        this method should return np.array([[0.5, 0.5]])"""
        return np.identity(self._size)
        # lifespan = np.array([c.lifespan() for c in self.contracts])
        # vxm0_weight = min(1, ((252 / 12) - lifespan[1]) / (lifespan[0] - lifespan[1]))
        # vxm1_weight = 1 - vxm0_weight
        # return np.array([vxm0_weight, vxm1_weight])

    def parse(self):
        """Parse sequence of spreads."""
        weights = self._make_weights()
        spreads = [
            self.exchange[c].bid_price / self.exchange[c].ask_price - 1
            for c in self.contracts
        ]
        spreads = np.abs([spreads])
        spreads = spreads.dot(weights.T)
        return spreads.clip(self._low, self._high)

    def _manual_fit_transformer(self):
        """Whatever the max or min allocation is per asset, either long or
        short, rescale the current portfolio weights in the [-1, +1] range."""
        x = np.concatenate([self.space.high, self.space.low])
        self.transformer.fit(x)


class _NullTransformer:
    def fit(self, *args, **kwargs):
        return self

    def transform(self, x):
        return x


class FeatureIsRTH(Feature):
    """Boolean flag indicating weather markets is trading during Regular
    Trading Hours."""

    def __init__(
            self,
            calendar: str = 'XCBF',
            tz: str = 'America/Chicago',
            name: str = None,
            transformer=None,
            kind: str = 'binary',
    ):
        if transformer is None:
            transformer = _NullTransformer()
            self._tz = tz
        if kind == 'binary':
            space = gym.spaces.MultiBinary(1)
        else:
            space = gym.spaces.Box(0., 1., (1, 1), float)
        self.calendar = xcals.get_calendar(calendar, side='neither')
        self.kind = kind
        super().__init__(space=space, name=name, transformer=transformer)

    def parse(self):
        """Parse progress bar during RTH, ETH or just a dummy indicating
        weather """
        now = self._now().tz_localize(self._tz).tz_convert('UTC')
        is_rth = self.calendar.is_open_on_minute(now)
        if self.kind == 'rth':
            left = self.calendar.previous_open(now)
            right = self.calendar.next_close(now)
            progress = (now - left) / (right - left)
            progress = np.array([[progress]]) * int(is_rth)
            #progress = np.array([1 - progress], dtype=np.float64) * int(is_rth)
        elif self.kind == 'eth':
            left = self.calendar.previous_close(now)
            right = self.calendar.next_open(now)
            progress = (now - left) / (right - left)
            progress = np.array([[progress]]) * int(not is_rth)
            #progress = np.array([1 - progress], dtype=np.float64) * int(not is_rth)
        else:
            progress = np.array([int(is_rth)])
        return progress

