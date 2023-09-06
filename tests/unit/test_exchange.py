from tradingenv.exchange import LimitOrderBook, Exchange
from tradingenv.events import Observer, EventNBBO, EventContractDiscontinued
from tradingenv.contracts import FutureChain, ES, Index
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np


class TestLimitOrderBook:
    def test_lazy_initialization(self):
        lob = LimitOrderBook()
        assert np.isnan(lob.bid_price)
        assert np.isnan(lob.ask_price)
        assert np.isnan(lob.bid_size)
        assert np.isnan(lob.ask_size)
        assert lob.time is None

        # Properties.
        assert np.isnan(lob.mid_price)
        assert np.isnan(lob.spread)

    def test_verbose_initialization(self):
        now = datetime.utcnow().replace(tzinfo=timezone.utc)
        lob = LimitOrderBook(1, 2, 3, 4, now)
        assert lob.bid_price == 1
        assert lob.ask_price == 2
        assert lob.bid_size == 3
        assert lob.ask_size == 4
        assert lob.time == now

        # Properties.
        assert lob.mid_price == 1.5
        assert lob.spread == 1


class TestExchange:
    def test_is_subclass_of_Observer(self):
        assert issubclass(Exchange, Observer)

    def test_initialization(self):
        exchange = Exchange()
        assert isinstance(exchange._books, defaultdict)
        assert exchange._books.default_factory == LimitOrderBook
        assert exchange.last_update is None
        assert exchange._observed_events == {
            "EventNBBO": "process_EventNBBO",
            "EventContractDiscontinued": "process_EventContractDiscontinued",
        }

    def test_getitem(self):
        exchange = Exchange()
        assert isinstance(exchange[Index("S&P 500")], LimitOrderBook)

    def test_len(self):
        exchange = Exchange()
        book1 = exchange._books[Index("S&P 500")]
        book2 = exchange._books["Bonds"]
        assert len(exchange) == 2

    def test_get_bid_prices(self):
        exchange = Exchange()
        exchange._books[Index("S&P 500")].bid_price = 9
        exchange._books[Index("S&P 500")].ask_price = 11
        exchange._books["Bonds"].bid_price = 99
        exchange._books["Bonds"].ask_price = 101
        expected = np.array([9, 99])
        actual = exchange.bid_prices([Index("S&P 500"), "Bonds"])
        assert all(actual == expected)

    def test_get_ask_prices(self):
        exchange = Exchange()
        exchange._books[Index("S&P 500")].bid_price = 9
        exchange._books[Index("S&P 500")].ask_price = 11
        exchange._books["Bonds"].bid_price = 99
        exchange._books["Bonds"].ask_price = 101
        expected = np.array([11, 101])
        actual = exchange.ask_prices([Index("S&P 500"), "Bonds"])
        assert all(actual == expected)

    def test_get_mid_prices(self):
        exchange = Exchange()
        exchange._books[Index("S&P 500")].bid_price = 9
        exchange._books[Index("S&P 500")].ask_price = 11
        exchange._books["Bonds"].bid_price = 99
        exchange._books["Bonds"].ask_price = 101
        expected = np.array([10, 100])
        actual = exchange.mid_prices([Index("S&P 500"), "Bonds"])
        assert all(actual == expected)

    def test_get_spreads(self):
        exchange = Exchange()
        exchange._books[Index("S&P 500")].bid_price = 9
        exchange._books[Index("S&P 500")].ask_price = 11
        exchange._books["Bonds"].bid_price = 99
        exchange._books["Bonds"].ask_price = 102
        expected = np.array([2, 3])
        actual = exchange.spreads([Index("S&P 500"), "Bonds"])
        assert all(actual == expected)

    def test_get_history(self):
        exchange = Exchange()
        exchange.process_EventNBBO(
            EventNBBO(datetime(2019, 1, 1), Index("S&P 500"), 1, 2, 3, 4)
        )
        exchange.process_EventNBBO(
            EventNBBO(datetime(2019, 1, 2), Index("T-Notes"), 5, 6, 7, 8)
        )
        exchange.process_EventNBBO(
            EventNBBO(datetime(2019, 1, 3), Index("T-Notes"), 9, 10, 11, 12)
        )
        exchange.process_EventNBBO(
            EventNBBO(datetime(2019, 1, 4), Index("S&P 500"), 13, 14, 15, 16)
        )

        assert exchange[Index("S&P 500")].history["time"] == [
            datetime(2019, 1, 1),
            datetime(2019, 1, 4),
        ]
        assert exchange[Index("S&P 500")].history["bid_price"] == [1, 13]
        assert exchange[Index("S&P 500")].history["ask_price"] == [2, 14]
        assert exchange[Index("S&P 500")].history["mid_price"] == [1.5, 13.5]
        assert exchange[Index("S&P 500")].history["bid_size"] == [3, 15]
        assert exchange[Index("S&P 500")].history["ask_size"] == [4, 16]

        assert exchange[Index("T-Notes")].history["time"] == [
            datetime(2019, 1, 2),
            datetime(2019, 1, 3),
        ]
        assert exchange[Index("T-Notes")].history["bid_price"] == [5, 9]
        assert exchange[Index("T-Notes")].history["ask_price"] == [6, 10]
        assert exchange[Index("T-Notes")].history["mid_price"] == [5.5, 9.5]
        assert exchange[Index("T-Notes")].history["bid_size"] == [7, 11]
        assert exchange[Index("T-Notes")].history["ask_size"] == [8, 12]

    def test_process_NBBO(self):
        now = datetime.utcnow().replace(tzinfo=timezone.utc)
        exchange = Exchange()
        exchange.process_EventNBBO(
            event=EventNBBO(
                time=now,
                bid_price=9,
                ask_price=11,
                bid_size=100,
                ask_size=200,
                contract=Index("S&P 500"),
            )
        )
        assert exchange._books[Index("S&P 500")].bid_price == 9
        assert exchange._books[Index("S&P 500")].ask_price == 11
        assert exchange._books[Index("S&P 500")].bid_size == 100
        assert exchange._books[Index("S&P 500")].ask_size == 200
        assert exchange._books[Index("S&P 500")].time == now
        assert exchange.last_update == now

    def test_process_EventContractDiscontinued(self):
        exchange = Exchange()
        exchange.process_EventNBBO(
            event=EventNBBO(
                time=datetime.now(),
                contract=Index("S&P 500"),
                bid_price=9,
                ask_price=11,
                bid_size=100,
                ask_size=200,
            )
        )
        assert exchange[Index("S&P 500")].bid_price == 9
        history_before_termination = exchange[Index("S&P 500")].history
        exchange.process_EventContractDiscontinued(
            event=EventContractDiscontinued(datetime.now(), Index("S&P 500"))
        )
        history_after_termination = exchange[Index("S&P 500")].history
        assert exchange[Index("S&P 500")].bid_price is np.nan
        assert history_before_termination == history_after_termination

    def test_process_NBBO_doesnt_update_after_discontinuation_of_book(self):
        event_discontinued = EventContractDiscontinued(datetime.now(), Index("S&P 500"))
        event_nbbo = EventNBBO(
            time=datetime.now(),
            contract=Index("S&P 500"),
            bid_price=9,
            ask_price=11,
            bid_size=100,
            ask_size=200,
        )

        exchange = Exchange()

        exchange.process_EventNBBO(event_nbbo)
        assert exchange[Index("S&P 500")].bid_price == 9

        exchange.process_EventContractDiscontinued(event_discontinued)
        assert exchange[Index("S&P 500")].bid_price is np.nan

        exchange.process_EventNBBO(event_nbbo)
        assert exchange[Index("S&P 500")].bid_price is np.nan  # nan despite nbbo

    def test_getitem_changes_key_for_leading_contracts(self):
        lead_contract = FutureChain(ES, "2018-12", "2019-03")
        lead_contract.now = datetime.min
        exchange = Exchange()
        assert exchange[lead_contract] is exchange[ES(2018, 12)]
        assert exchange[lead_contract] is not exchange[ES(2019, 3)]
