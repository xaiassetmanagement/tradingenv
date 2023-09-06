"""The user is expected to instance these classes when instancing the action
observation_space within TradingEnv. For examples, see spaces.py and env.py."""
from tradingenv.events import IEvent, EventContractDiscontinued
from abc import ABC, abstractmethod
from typing import List, Type, Union, Sequence
from datetime import datetime, timedelta
from bisect import bisect_right
from collections import defaultdict
from pandas.tseries.offsets import BDay
import numpy as np
import pandas as pd
import calendar


class AbstractContract(ABC):
    """Trading contract must be represented by implementation of this abstract
    class, then passed to the action observation_space within the trading environment.

    Attributes
    ----------
    now : datetime
        Current time. This is handy for a subset of contracts such as
        FutureChain or calculate nr days to expiry for derivatives. This is a
        class attribute shared across all AbstractContract instances.
    """

    # TODO: global attribute does not reset across tests and could lead to
    #   information leakage across runs. E.g. run a regression test of env
    #   without resetting. Then run a unit test using this now attribute, it
    #   will show terminal now of the regression test despite being in a unit
    #   test. Not serious problem for normal pkg usage, but not perfect design
    #   for sure. E.g. this global attr could be a Clock class to centralise
    #   time.
    now: datetime = datetime.min

    def __hash__(self) -> int:
        """Contracts are mapped by this hash number in the Exchange. Therefore,
        different contracts have to have different hash number."""
        return hash(self.symbol)

    def __eq__(self, other: Union["AbstractContract", str]) -> bool:
        """Allows using AbstractContract.symbol in addition to the class
        instance when selecting items from a dictionary."""
        try:
            return self.symbol == other.symbol
        except AttributeError:
            # AttributeError: 'other' object has no attribute 'symbol'
            return hash(self.symbol) == hash(other)

    def __repr__(self) -> str:
        return "{}({})".format(self.__class__.__name__, self.symbol)

    def __lt__(self, other: 'Future'):
        return self.symbol < other.symbol

    @property
    @abstractmethod
    def symbol(self) -> str:
        """Unique symbol associated with the concrete implementation.
        Non-unique symbols might lead to silent bugs as symbols are used
        to hash AbstractClass instances."""

    @property
    @abstractmethod
    def multiplier(self) -> float:
        """Contract's multiplier. This is generally 1 for non-derivatives
        (e.g. ETFs, Indexes, Stocks) and > 1 for derivatives (e.g. Future)."""

    @property
    @abstractmethod
    def margin_requirement(self) -> float:
        """A number ranging from 0 to 1 defining what (constant) proportion of
        the notional value should be deposited as initial _margin when buying
        or selling the contract. Note that by design we are making the
        simplifying assumption that initial _margin == maintenance _margin.
        This is generally 1 for non-derivatives (e.g. ETFs, Indexes, Stocks)
        and > 1 for derivatives (e.g. Future)."""

    @property
    @abstractmethod
    def cash_requirement(self) -> float:
        """A number ranging from 0 to 1 defining what (constant) proportion of
        the notional value should be paid when buying the contract. This is
        generally 1 for non-derivatives (e.g. ETFs, Indexes, Stocks)
        and 0 for derivatives (e.g. Future)."""

    @property
    def underlyings(self) -> List["AbstractContract"]:
        """Returns list of all subcontracts. This is needed when implementing
        composite contract such as LeadingContract. For example for
        FutureChain(ES) this would return the list of ES contracts
        across all expiry dates. Note that this is not to be confused with
        the underlying index e.g. Index('S&P 500') for eMini S&P 500."""
        return [self]

    def verify(self, mid_price: float):
        if mid_price < 0:
            raise ValueError(
                "Found negative price {} for contract {}."
                "".format(mid_price, self)
            )

    def make_events(self) -> List[IEvent]:
        """Returns a list of events (IEvent instances) to be added the
         Transmitter when initializing TradingEnv. For example,
         EventContractDiscontinued associated with futures contracts."""
        return list()

    def size(self, price: float) -> float:
        """
        Parameters
        ----------
        price : float
            Value of one unit of the underlying asset.

        Returns
        -------
        Value of the underlying asset multiplier by the future's multiplier.
        """
        return self.multiplier * price

    def static_hashing(self) -> 'AbstractContract':
        """Subclasses of AbstractContract are not guaranteed to have static
        hashing, as it may vary with time. This method
        returns an instance of AbstractContract with the same hash of the
        class instance calling this method, which is guaranteed to be static.
        See FutureChain for more info as it is an example of AbstractContract
        with dynamic hashing."""
        return self

    @property
    def symbol_short(self) -> str:
        """Symbol is guaranteed to be unique across all contracts, and in fact
        it is used to hash the instance. The short symbol is not guaranteed
        to be unique. Useful e.g. for futures where the same future across
        different expiry dates share the same short symbol but not the same
        symbol."""
        return self.symbol


class Cash(AbstractContract):
    """A cash contract (e.g. USD).

    Examples
    --------
    >>> cash = Cash('EUR')
    >>> cash
    Cash(EUR)
    """

    multiplier = 1.0
    cash_requirement = 1.0
    margin_requirement = 0.0

    def __init__(self, currency: str = "USD"):
        self.currency = currency

    @property
    def symbol(self) -> str:
        return self.currency


class ETF(AbstractContract):
    """An ETF contract (e.g. SPY).

    Examples
    --------
    >>> etf = ETF('SPY')
    >>> etf
    ETF(SPY)
    """

    multiplier = 1.0
    cash_requirement = 1.0
    margin_requirement = 0.0

    def __init__(self, symbol: str, exchange: str = "SMART", currency: str = "USD"):
        self._symbol = symbol
        self.exchange = exchange
        self.currency = currency

    @property
    def symbol(self) -> str:
        return self._symbol


class Index(AbstractContract):
    """An index (e.g. S&P 500).

    Examples
    --------
    >>> index = Index('S&P 500')
    >>> index
    Index(S&P 500)
    """

    multiplier = 1.0
    cash_requirement = 1.0
    margin_requirement = 0.0

    def __init__(self, symbol: str):
        self._symbol = symbol

    @property
    def symbol(self) -> str:
        return self._symbol


class Rate(AbstractContract):
    """An rate (e.g. FED Funds rate).

    Examples
    --------
    >>> rate = Rate('FED funds rate')
    >>> rate
    Rate(FED funds rate)
    """

    multiplier = 1.0
    cash_requirement = 1.0
    margin_requirement = 0.0

    def __init__(self, symbol: str):
        self._symbol = symbol

    @property
    def symbol(self) -> str:
        return self._symbol

    def verify(self, mid_price: float):
        if mid_price >= 0.25:
            raise ValueError(
                '{} is unexpectedly high ({:.2%}). Perhaps you forgot to '
                'divide your interest rates data by 100 (0.01 reads 1%, 1 '
                'reads 100%)?'
                ''.format(self, mid_price)
            )


class Future(AbstractContract):
    """A future contract (e.g. ES, ZN)"""
    exists_since = datetime(2000, 1, 1)
    exists_until = datetime.now().date() + timedelta(days=260 * 2)
    cash_requirement = 0.0
    month_codes = {
        1: "F",
        2: "G",
        3: "H",
        4: "J",
        5: "K",
        6: "M",
        7: "N",
        8: "Q",
        9: "U",
        10: "V",
        11: "X",
        12: "Z",
    }

    def __init__(self, year: int, month: int):
        """
        Parameters
        ----------
        year : int
            Expiry year of the future contract.
        month : int
            Expiry month of the future contract.
        """
        self.expiry = self._get_expiry_date(year, month)
        self.last_trading_date = self._get_last_trading_date(self.expiry)
        self._symbol_short = self.__class__.__name__
        self._symbol = "{symbol_short}{month_code}{year_code}".format(
            symbol_short=self._symbol_short,
            month_code=self.month_codes[self.expiry.month],
            year_code=self.expiry.strftime("%y"),
        )

    def __lt__(self, other: 'Future'):
        return self.last_trading_date < other.last_trading_date

    @abstractmethod
    def _get_expiry_date(self, year: int, month: int) -> datetime:
        """Given the expiry year (int) and month (int), this method returns
         the expiry date of the contract. Note that the expiry date does not
         necessarily corresponds to the last trading date, which generally
         precedes the expiry month. The last trading date is implemented in
         Future._get_last_trading_date."""

    @abstractmethod
    def _get_last_trading_date(self, expiry: datetime) -> datetime:
        """Given the expiry date, returns the last trading date of the
        contract. Futures contract generally stops trading at least a few
        days before the expiry."""

    @property
    @abstractmethod
    def freq(self) -> str:
        """Returns frequency of the contracts, e.g. Q-DEC for ES, M for VX."""

    @property
    @abstractmethod
    def margin_requirement(self) -> float:
        """A number ranging from 0 to 1 defining what (constant) proportion of
        the notional value should be deposited as initial _margin when buying
        or selling the contract. Note that by design we are making the
        simplifying assumption that initial _margin == maintenance _margin."""

    @property
    def symbol(self) -> str:
        """Long symbol of the future contract, e.g. 'ESU19' for the ES contract
        expiring in """
        return self._symbol

    @property
    def symbol_short(self) -> str:
        return self._symbol_short

    def make_events(self) -> List[IEvent]:
        """Returns an event run when the contract will expire thus becoming
        no longer investable."""
        return [EventContractDiscontinued(time=self.expiry, contract=self)]

    def lifespan(self) -> float:
        """Returns the number of trading days left between now and the expiry
        date."""
        return np.busday_count(self.now.date(), self.expiry.date())


class FutureChain(AbstractContract):
    """Composition of Future contracts, where the only investable contract is
    the one associated with the leading month. You must have a very good
    reason for NOT using this class when running futures trading strategies.
    Thanks to this class, you don't have to work with continuous futures data,
    you can just use the chain of uncombined futures contracts across all
    expires dates of interest."""

    cash_requirement = 0.0

    def __init__(
        self,
        future_cls: Type[Future]=None,
        start: Union[str, datetime]=None,
        end: Union[str, datetime]=None,
        contracts: Sequence[Future]=None,
        month: int=0,
    ):
        """
        Parameters
        ----------
        future_cls
            A subclass of Feature (e.g. ES, ZN, VX). Full list of available
            futures can be found in futures.py or provide your own
            implementation.
        start
            Start period of the first available futures contract, passed to
            pandas.date_range.
        end
            Last period of the first available futures contract, passed to
            pandas.date_range.
        contracts
            A sequence of Future instances which will be represented by this
            single class. If you pass this argument, there is no need to also
            pass 'future_cls', 'start' or 'end' as they will be deduced.
        month
            0 by default, meaning that the preferred feature in the term
            structure is the one with the shorted expiry date (i.e. M1). If
            1, then the second expiration date will be preferred etc.

        Examples
        --------
        >>> from tradingenv.contracts import ES
        >>> future = FutureChain(ES, '2018-03', '2018-12')
        >>> future.lead_contract(datetime(2018, 4, 1))
        ES(ESM18)
        >>> future = FutureChain(ES, '2018-03', '2018-12', month=1)
        >>> future.lead_contract(datetime(2018, 4, 1))
        ES(ESU18)
        """
        self._month = month
        if future_cls is None and contracts is None:
            raise ValueError(
                "At least one argument between 'future_cls' and 'futures_"
                "chain' must be provided."
            )
        if contracts is None:
            start = start or future_cls.exists_since
            end = end or future_cls.exists_until
            months = pd.date_range(start, end, freq=future_cls.freq)
            self.contracts = [future_cls(q.year, q.month) for q in months]
        else:
            self.contracts = sorted(contracts)
        self._last_trading_dates = [
            future.last_trading_date for future in self.contracts
        ]

    @property
    def symbol(self) -> str:
        """Long symbol of the current leading contract from the current time
        (i.e. FutureChain.now). For example, in 2019 Oct, the leading ES
        contract symbol is 'ESZ19'."""
        return self.lead_contract().symbol

    @property
    def symbol_short(self) -> str:
        return self.lead_contract().symbol_short

    @property
    def multiplier(self) -> float:
        """Multiplier of the future contract."""
        return self.contracts[0].multiplier  # all the same, same cls

    @property
    def underlyings(self) -> List["AbstractContract"]:
        return self.contracts

    @property
    def margin_requirement(self) -> float:
        return self.contracts[0].margin_requirement

    def _lead_contract_idx(self, now: datetime = None) -> int:
        """Returns the index of the leading contract from self.contracts."""
        now = self.now if now is None else now
        idx = bisect_right(self._last_trading_dates, now)
        idx += self._month
        return idx

    def lead_contract(self, now: datetime = None, month: int=0) -> Future:
        """Use self.now to identify the lead future contract"""
        if now is None:
            now = self.now
        # TODO: test month.
        idx = self._lead_contract_idx(now)
        idx += month
        return self.contracts[idx]

    def make_events(self) -> List[IEvent]:
        """Returns list of EventContractDiscontinued associated with every
        future contract belonging to the specified period when instancing this
        class."""
        events = list()
        for future in self.contracts:
            events.extend(future.make_events())
        return events

    def lifespan(self) -> float:
        """Returns the number of trading days left between now and the expiry
        date of the leading future contract."""
        return self.lead_contract().lifespan()

    def static_hashing(self) -> 'AbstractContract':
        """The hash of this class instance varies with the lead contract, which
        varies with time. This method returns the future instance associated
        with the current time and static hashing."""
        return self.lead_contract()


class ES(Future):
    """
    Notes
    -----
    First trading date is 1997-09-07, see [1].

    References
    ----------
    [1] https://www.cmegroup.com/media-room/historical-first-trade-dates.html#equityIndices
    https://www.interactivebrokers.com/en/index.php?f=26662
    https://www.cmegroup.com/clearing/risk-management/historical-margins.html
    https://www.interactivebrokers.com/en/index.php?f=1567&p=physical
    https://www.tradingacademy.com/lessons/article/futures-contract-rollover
    """
    exists_since = datetime(1997, 9, 7)  # see reference [1]
    freq = "Q-DEC"
    multiplier = 50.0

    def _get_expiry_date(self, year: int, month: int) -> datetime:
        """Returns a datetime object representing the third Friday of the
        month."""
        # https://www.barchart.com/futures/quotes/ES*0/profile
        _, nr_days = calendar.monthrange(year, month)
        dates = defaultdict(list)
        for day in range(1, nr_days + 1):
            date = datetime(year, month, day)
            weekday = date.strftime("%A")
            dates[weekday].append(date)
        return dates["Friday"][2]

    def _get_last_trading_date(self, expiry: datetime) -> datetime:
        return expiry - timedelta(days=8)

    @property
    def margin_requirement(self) -> float:
        return 0.1


class VX(Future):
    """
    References
    ----------
    [1] https://www.cmegroup.com/media-room/historical-first-trade-dates.html#equityIndices
    https://www.interactivebrokers.com/en/index.php?f=2222&exch=cfe&showcategories=FUTGRP#productbuffer
    file:///home/federico/Downloads/Trading_VIX_Futures_Options.pdf
    http://www.cboe.com/micro/vix/pdf/VIX%20fact%20sheet%202019.pdf
    http://cfe.cboe.com/cfe-products/vx-cboe-volatility-index-vix-futures/contract-specifications
    https://www.cboe.com/ms/vix-futures-specsheet.pdf
    """
    exists_since = datetime(2004, 3, 26)  # see reference [1]
    freq = "M"
    multiplier = 1000.0

    def _get_expiry_date(self, year: int, month: int) -> datetime:
        """
        Returns
        -------
        Final Settlement Date. The final settlement date for a contract with
        the "VX" ticker symbol is on the Wednesday that is 30 days prior to
        the third Friday of the calendar month immediately following the
        month in which the contract expires. The final settlement date for
        a futures contract with the "VX" ticker symbol followed by a number
        denoting the specific week of a calendar year is on the Wednesday of
        the week specifically denoted in the ticker symbol.
        If that Wednesday or the Friday that is 30 days following that
        Wednesday is a Cboe Options holiday, the final settlement date for
        the contract shall be on the business day immediately preceding that
        Wednesday.

        Reference
        ---------
        https://www.cboe.com/tradable_products/vix/vix_futures/specifications/
        """
        this_month = datetime(year, month, 1)
        next_month = this_month + timedelta(days=32)
        next_month_third_friday_day = 21 - (
                    calendar.weekday(next_month.year, next_month.month,
                                     1) + 2) % 7
        next_month_third_friday = next_month.replace(
            day=next_month_third_friday_day)
        expiration = next_month_third_friday - timedelta(days=30)
        return expiration

    def _get_last_trading_date(self, expiry: datetime) -> datetime:
        return expiry - BDay(2)

    @property
    def margin_requirement(self) -> float:
        return 0.5


class _Treasury(Future):
    """
    References
    ----------
    [1] https://www.cmegroup.com/media-room/historical-first-trade-dates.html#equityIndices
    https://www.interactivebrokers.com/en/index.php?f=26662
    https://www.cmegroup.com/clearing/risk-management/historical-margins.html
    https://www.interactivebrokers.com/en/index.php?f=1567&p=physical
    https://www.tradingacademy.com/lessons/article/futures-contract-rollover
    https://www.barchart.com/futures/quotes/ES*0/profile
    """
    exists_since = datetime(1970, 1, 1)
    freq = "Q-DEC"

    def _get_expiry_date(self, year: int, month: int) -> datetime:
        """Last business day of the delivery month."""
        # https://www.barchart.com/futures/quotes/ZN*0/profile
        dates = pd.date_range(datetime(year, month, 1), periods=31, freq="B")
        dates = [date for date in dates if date.month == month]
        return dates[-1]

    def _get_last_trading_date(self, expiry: datetime) -> datetime:
        # It's like ES, but physical delivery. See page 18:
        # https://www.cmegroup.com/education/files/understanding-treasury-futures.pdf
        # https://www.barchart.com/futures/quotes/ZN*0/profile
        return (expiry - timedelta(days=30)).replace(day=24)

    @property
    def margin_requirement(self) -> float:
        return 0.05


class ZQ(_Treasury):
    multiplier = 4167
    margin_requirement = 0.004


class ZT(_Treasury):
    multiplier = 2000
    margin_requirement = 0.02


class ZF(_Treasury):
    multiplier = 1000
    margin_requirement = 0.02


class ZN(_Treasury):
    multiplier = 1000
    margin_requirement = 0.03


class ZB(_Treasury):
    multiplier = 1000
    margin_requirement = 0.05


class NK(Future):
    """
    References
    ----------
    https://www.cmegroup.com/trading/equity-index/international-index/nikkei-225-dollar_contract_specifications.html
    https://misc.interactivebrokers.com/cstools/contract_info/v3.10/index.php?action=Conid%20Info&wlId=IB&conid=345561869&lang=en
    https://www.cmegroup.com/trading/equity-index/nikkei-225-futures-and-options.html
    https://www.cmegroup.com/trading/equity-index/international-index/nikkei-225-dollar_contract_specifications.html
    """

    exists_since = datetime(1990, 11, 8)  # see reference [1]
    freq = "Q-DEC"
    multiplier = 5.

    def _get_expiry_date(self, year: int, month: int) -> datetime:
        """Returns a datetime object representing the third Friday of the
        month."""
        # https://www.barchart.com/futures/quotes/ES*0/profile
        _, nr_days = calendar.monthrange(year, month)
        dates = defaultdict(list)
        for day in range(1, nr_days + 1):
            date = datetime(year, month, day)
            weekday = date.strftime("%A")
            dates[weekday].append(date)
        return dates["Friday"][1]

    def _get_last_trading_date(self, expiry: datetime) -> datetime:
        """
        Termination of trading here:
            https://www.cmegroup.com/trading/equity-index/nikkei-225-futures-and-options.html
        https://misc.interactivebrokers.com/cstools/contract_info/v3.10/index.php?action=Conid%20Info&wlId=IB&conid=345561869&lang=en
        """
        # It's like ES, but physical delivery. See page 18:
        # https://www.cmegroup.com/education/files/understanding-treasury-futures.pdf
        # https://www.barchart.com/futures/quotes/ZN*0/profile
        return expiry - timedelta(days=14)  # NOTE: IB allows to trade 1 day before the expiry

    @property
    def margin_requirement(self) -> float:
        """Trading terminates on business day before the 2nd friday of the
        contract month."""
        return 0.3
