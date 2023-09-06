"""Classes to store the state of the exchange, used to check buy and sell
prices. Exchange is an aggregation of LimitOrderBook."""
from tradingenv.events import Observer, EventNBBO, EventContractDiscontinued
from tradingenv.contracts import AbstractContract
from datetime import datetime
from typing import Sequence, Dict, Union
from collections import defaultdict
import numpy as np
import pandas as pd


class LimitOrderBook:
    """Limit order book of a single asset."""

    def __init__(
        self,
        bid_price: float = np.nan,
        ask_price: float = np.nan,
        bid_size: float = np.nan,
        ask_size: float = np.nan,
        time: datetime = None,
    ):
        """
        Parameters
        ----------
        bid_price : float
            Latest on-screen bid transaction_price.
        ask_price : float
            Latest on-screen ask transaction_price.
        bid_size : float
            Latest on-screen bid size.
        ask_size : float
            Latest on-screen ask size.

        Examples
        --------
        >>> from tradingenv.exchange import LimitOrderBook
        >>> lob = LimitOrderBook(
        ...     bid_price=99.9,
        ...     ask_price=100.1,
        ...     bid_size=100,
        ...     ask_size=200,
        ... )
        >>> lob
        99.9 : 100.1
        """
        # Inputs.
        self.bid_price = bid_price
        self.ask_price = ask_price
        self.bid_size = bid_size
        self.ask_size = ask_size
        self.time = time

        # More attributes.
        self.history = defaultdict(list)
        self.is_alive = True

    def __repr__(self) -> str:
        """Class string representation."""
        return "{bid_price} : {ask_price}" "".format(
            bid_price=self.bid_price,
            ask_price=self.ask_price,
        )

    @property
    def spread(self) -> float:
        """Returns bid-ask spread."""
        return self.ask_price - self.bid_price

    @property
    def mid_price(self) -> float:
        """Returns the middle transaction_price between the bid and the ask."""
        return (self.ask_price + self.bid_price) / 2

    def liq_price(self, quantity: float) -> float:
        """Returns the execution price of selling 'quantity' now."""
        return self.acq_price(-quantity)

    def acq_price(self, quantity: float) -> float:
        """Returns the execution price of buying 'quantity' now."""
        if quantity < 0:
            return self.bid_price
        elif quantity > 0:
            return self.ask_price
        elif quantity == 0:
            return self.mid_price
        else:
            raise ValueError("Unexpected sign: {}".format(quantity))

    def update(self, event: EventNBBO):
        # NOTE: this Exchange is responsible to call this method rather than
        # making LOB an Observe for lecagy reasons.
        self.bid_price = event.bid_price
        self.ask_price = event.ask_price
        self.bid_size = event.bid_size
        self.ask_size = event.ask_size
        self.time = event.time

        # Save historical data.
        self.history["time"].append(self.time)
        self.history["bid_price"].append(self.bid_price)
        self.history["ask_price"].append(self.ask_price)
        self.history["mid_price"].append(self.mid_price)
        self.history["bid_size"].append(self.bid_size)
        self.history["ask_size"].append(self.ask_size)

    def terminate(self, event: EventContractDiscontinued):
        """Reset and stop quotes in the lob but keep history."""
        history = self.history
        self.__init__()
        self.history = history
        self.time = event.time
        self.is_alive = False

    def to_frame(self, what: str='mid_price') -> pd.Series:
        return pd.Series(
            data=self.history[what],
            index=self.history['time'],
            name=what,
        )


class Exchange(Observer):
    """An aggregation of limit order books.

    Attributes
    ----------
    _books : Dict[AbstractContract, LimitOrderBook]
        A dictionary mapping contract IDs to the corresponding instance of
        LimitOrderBook.
    """

    def __init__(self):
        """
        Examples
        --------
        >>> from tradingenv.exchange import Exchange
        >>> from tradingenv.events import EventNBBO
        >>> from tradingenv.contracts import ETF
        >>> exchange = Exchange()
        >>>
        >>> # Assume that there is a NBBO update for S&P 500 and NASDAQ.
        >>> nbbo_sp500 = EventNBBO(
        ...     time=datetime.now(),
        ...     bid_price=9,
        ...     ask_price=11,
        ...     bid_size=100,
        ...     ask_size=200,
        ...     contract=ETF('SPY'),
        ... )
        >>> nbbo_nasdaq = EventNBBO(
        ...     time=datetime.now(),
        ...     bid_price=99,
        ...     ask_price=101,
        ...     bid_size=10,
        ...     ask_size=20,
        ...     contract=ETF('IEF'),
        ... )
        >>> exchange.process_EventNBBO(nbbo_sp500)
        >>> exchange.process_EventNBBO(nbbo_nasdaq)
        >>> exchange
        {ETF(SPY): '9 : 11', ETF(IEF): '99 : 101'}
        """
        self._books = defaultdict(LimitOrderBook)
        self.last_update = None

    def __repr__(self) -> str:
        """Class string representation."""
        return str(
            {contract_id: repr(book)
             for contract_id, book in self._books.items()
             if not np.isnan(book.mid_price)
             }
        )

    def __len__(self) -> int:
        """Returns number of limit order books in the exchange."""
        return len(self._books)

    def __getitem__(self, key: Union[str, AbstractContract]) -> LimitOrderBook:
        """Returns the limit order book associated with contract_id."""
        if isinstance(key, AbstractContract):
            key = key.static_hashing()  # TODO: Test
        return self._books[key]

    def bid_prices(self, keys: Sequence[AbstractContract]) -> np.array:
        """Return a np.array with mid prices of the provided contract IDs."""
        return np.array([self[key].bid_price for key in keys])

    def ask_prices(self, keys: Sequence[AbstractContract]) -> np.array:
        """Return a np.array with mid prices of the provided contract IDs."""
        return np.array([self[key].ask_price for key in keys])

    def mid_prices(self, keys: Sequence[AbstractContract]) -> np.array:
        """Return a np.array with mid prices of the provided contract IDs."""
        return np.array([self[key].mid_price for key in keys])

    def acq_prices(
        self, keys: Sequence[AbstractContract], signs: np.ndarray
    ) -> np.ndarray:
        return np.array([self[key].acq_price(sign) for key, sign in zip(keys, signs)])

    def liq_prices(
        self, keys: Sequence[AbstractContract], signs: np.ndarray
    ) -> np.ndarray:
        return self.acq_prices(keys, -signs)

    def spreads(self, keys: Sequence[AbstractContract]) -> np.array:
        """Return a np.array with spreads of the provided contract IDs."""
        return np.array([self[key].spread for key in keys])

    def process_EventNBBO(self, event: EventNBBO):
        """
        Process an EventNBBO, representing an update of market prices and
        sizes in the limit order book.

        Parameters
        ----------
        event : EventNBBO
            An event instance storing prices, sizes and contract_id to be
            updated.
        """
        book = self[event.contract]
        if book.is_alive:
            book.update(event)
        self.last_update = event.time

    def process_EventContractDiscontinued(self, event: EventContractDiscontinued):
        self[event.contract].terminate(event)

    def to_frame(self, contracts: Sequence[AbstractContract], join: str = 'outer'):
        # TODO: Test.
        # An alternative implementation, but not faster due to slow .from_dict
        #contracts = result.policy._returns.columns
        #raw = dict()
        #for contract in contracts:
        #    history = result.env.exchange[contract].history
        #    raw[contract] = dict(zip(history['time'], history['mid_price']))
        #pd.DataFrame.from_dict(raw).sort_index()
        data = list()
        for contract in contracts:
            history = self[contract].history
            series = pd.Series(
                data=history['mid_price'],
                index=history['time'],
                name=contract,
                dtype=np.float32,
            )
            data.append(series)
        return pd.concat(data, axis='columns', join=join, sort=True, copy=False)
