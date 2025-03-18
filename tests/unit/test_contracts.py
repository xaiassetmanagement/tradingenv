from tradingenv.contracts import (
    AbstractContract,
    Cash,
    ETF,
    Index,
    Rate,
    Future,
    FutureChain,
    VX,
)
from tradingenv.events import EventContractDiscontinued
from datetime import datetime, timedelta
import numpy as np
import pytest
import pandas as pd


class TestAbstractContract:
    def test_multiplier_is_abstractproperty(self):
        assert AbstractContract.multiplier.__isabstractmethod__

    def test_symbol_is_abstractproperty(self):
        assert AbstractContract.symbol.__isabstractmethod__

    def test_margin_requirement_is_abstractproperty(self):
        assert AbstractContract.margin_requirement.__isabstractmethod__

    def test_cash_requirement_is_abstractproperty(self):
        assert AbstractContract.cash_requirement.__isabstractmethod__

    def test_now_is_datetime_global_attribute(self):
        assert isinstance(AbstractContract.now, datetime)

    def test_make_events_returns_no_events_by_default(self):
        class MockAbstractContract(AbstractContract):
            cash_requirement = 1.0
            margin_requirement = 0.0

            @property
            def multiplier(self):
                return 42.0

            @property
            def symbol(self):
                return "TheAnswer"

        contract = MockAbstractContract()
        events = contract.make_events()
        assert events == list()

    def test_contract_in_dictionary_can_be_hashed_by_symbol(self):
        class ContractSPY(AbstractContract):
            cash_requirement = 1.0
            margin_requirement = 0.0

            @property
            def multiplier(self):
                return 18.0

            @property
            def symbol(self):
                return "SPY"

        class ContractIEF(AbstractContract):
            cash_requirement = 1.0
            margin_requirement = 0.0

            @property
            def multiplier(self):
                return 42.0

            @property
            def symbol(self):
                return "IEF"

        spy = ContractSPY()
        ief = ContractIEF()
        d = {spy: spy, ief: ief}
        assert d[spy] is spy
        assert d["SPY"] is spy
        assert d[ief] is ief
        assert d["IEF"] is ief

    def test_two_contracts_in_dictionary_with_same_hash(self):
        """This is equivalent to {'a': 1, 'a': 2} but with contracts as keys."""

        class ContractSPY1(AbstractContract):
            cash_requirement = 1.0
            margin_requirement = 0.0

            @property
            def multiplier(self):
                return 18.0

            @property
            def symbol(self):
                return "SPY"

        class ContractSPY2(AbstractContract):
            cash_requirement = 1.0
            margin_requirement = 0.0

            @property
            def multiplier(self):
                return 42.0

            @property
            def symbol(self):
                return "SPY"

        spy1 = ContractSPY1()
        spy2 = ContractSPY2()
        d = {spy1: spy1, spy2: spy2}
        assert d == {spy1: spy1} or d == {spy2: spy2}

    def test_size(self):
        class MockAbstractContract(AbstractContract):
            cash_requirement = 1.0
            margin_requirement = 0.0

            @property
            def multiplier(self):
                return 42.0

            @property
            def symbol(self):
                return "TheAnswer"

        contract = MockAbstractContract()
        assert contract.size(price=2845.25) == 2845.25 * 42

    def test_include(self):
        class SPY(AbstractContract):
            cash_requirement = 1.0
            margin_requirement = 0.0

            @property
            def multiplier(self):
                return 1.0

            @property
            def symbol(self):
                return "SPY"

        class IEF(AbstractContract):
            cash_requirement = 1.0
            margin_requirement = 0.0

            @property
            def multiplier(self):
                return 1.0

            @property
            def symbol(self):
                return "IEF"

        assert SPY() in [SPY()]
        assert SPY() not in [IEF()]

    def test_set(self):
        class SPY(AbstractContract):
            cash_requirement = 1.0
            margin_requirement = 0.0

            @property
            def multiplier(self):
                return 1.0

            @property
            def symbol(self):
                return "SPY"

        assert {SPY(), SPY()} == {SPY()}

    def test_underlyings(self):
        class SPY(AbstractContract):
            cash_requirement = 1.0
            margin_requirement = 0.0

            @property
            def multiplier(self):
                return 1.0

            @property
            def symbol(self):
                return "SPY"

        spy = SPY()
        assert spy.underlyings == [spy]

    def test_static_hashing(self):
        class ContractSPY(AbstractContract):
            cash_requirement = 1.0
            margin_requirement = 0.0

            @property
            def multiplier(self):
                return 18.0

            @property
            def symbol(self):
                return "SPY"

        spy = ContractSPY()
        assert id(spy.static_hashing()) == id(spy)

    def test_symbol_short(self):
        class ContractSPY(AbstractContract):
            cash_requirement = 1.0
            margin_requirement = 0.0

            @property
            def multiplier(self):
                return 18.0

            @property
            def symbol(self):
                return "SPY"

        spy = ContractSPY()
        assert spy.symbol_short == 'SPY'


class TestCash:
    def test_parent_class_is_abstract(self):
        assert issubclass(Cash, AbstractContract)

    def test_multiplier(self):
        cash = Cash()
        assert cash.multiplier == 1

    def test_symbol(self):
        cash = Cash()
        assert cash.symbol == "USD"

    def test_custom_symbol(self):
        cash = Cash("EUR")
        assert cash.symbol == "EUR"

    def test_margin_requirement(self):
        cash = Cash()
        assert cash.margin_requirement == 0

    def test_cash_requirement(self):
        cash = Cash()
        assert cash.cash_requirement == 1


class TestETF:
    def test_parent_class_is_abstract(self):
        assert issubclass(ETF, AbstractContract)

    def test_multiplier(self):
        cash = ETF("SPY")
        assert cash.multiplier == 1

    def test_symbol(self):
        cash = ETF("SPY")
        assert cash.symbol == "SPY"

    def test_margin_requirement(self):
        cash = ETF("SPY")
        assert cash.margin_requirement == 0

    def test_cash_requirement(self):
        cash = ETF("SPY")
        assert cash.cash_requirement == 1


class TestIndex:
    def test_parent_class_is_abstract(self):
        assert issubclass(Index, AbstractContract)

    def test_multiplier(self):
        cash = Index("S&P 500")
        assert cash.multiplier == 1

    def test_symbol(self):
        cash = Index("S&P 500")
        assert cash.symbol == "S&P 500"

    def test_margin_requirement(self):
        cash = Index("S&P 500")
        assert cash.margin_requirement == 0

    def test_cash_requirement(self):
        cash = Index("S&P 500")
        assert cash.cash_requirement == 1


class TestRate:
    def test_parent_class_is_abstract(self):
        assert issubclass(Rate, AbstractContract)

    def test_multiplier(self):
        cash = Rate("FED Funds Rate")
        assert cash.multiplier == 1

    def test_symbol(self):
        cash = Rate("FED Funds Rate")
        assert cash.symbol == "FED Funds Rate"

    def test_margin_requirement(self):
        cash = Rate("FED Funds Rate")
        assert cash.margin_requirement == 0

    def test_cash_requirement(self):
        cash = Rate("FED Funds Rate")
        assert cash.cash_requirement == 1


class TestFuture:
    def make_es_cls(self):
        class ES(Future):
            cash_requirement = 0.0
            margin_requirement = 0.1
            multiplier = 50.

            @classmethod
            def freq(cls) -> str:
                return "QE-DEC"

            def _get_expiry_date(self, year: int, month: int) -> datetime:
                return datetime(2019, 9, 17)

            def _get_last_trading_date(self, expiry: datetime) -> datetime:
                return datetime(2019, 9, 5)

        return ES(2019, 6)

    def make_zn_cls(self):
        class ZN(Future):
            cash_requirement = 0.0
            margin_requirement = 0.2
            multiplier = 100.

            @classmethod
            def freq(cls) -> str:
                return "QE-DEC"

            def _get_expiry_date(self, year: int, month: int) -> datetime:
                return datetime(2019, 9, 15)

            def _get_last_trading_date(self, expiry: datetime) -> datetime:
                return datetime(2019, 9, 4)

        return ZN(2019, 6)

    def test_parent_class_is_abstract(self):
        assert issubclass(Future, AbstractContract)

    def test_type_is_abstract(self):
        with pytest.raises(TypeError):
            Future(2019, 6)

    def test_freq_is_abstract(self):
        assert Future.freq.__isabstractmethod__

    def test_multiplier_is_abstract(self):
        assert Future.multiplier.__isabstractmethod__

    def test_get_expiry_date_is_abstract(self):
        assert Future._get_expiry_date.__isabstractmethod__

    def test_get_cutoff_is_abstract(self):
        assert Future._get_last_trading_date.__isabstractmethod__

    def test_margin_requirement_is_abstractproperty(self):
        assert Future.margin_requirement.__isabstractmethod__

    def test_month_codes(self):
        assert Future.month_codes == {
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

    def test_expiry(self):
        future = self.make_es_cls()
        assert future.expiry == datetime(2019, 9, 17)

    def test_cutoff(self):
        future = self.make_es_cls()
        assert future.last_trading_date == datetime(2019, 9, 5)

    def test_symbol(self):
        future = self.make_es_cls()
        assert future.symbol == "ESU19"

    def test_make_events(self):
        future = self.make_es_cls()
        actual = future.make_events()
        expected = EventContractDiscontinued(
            contract=future, time=datetime(2019, 9, 17)
        )
        assert actual == [expected]

    def test_size(self):
        future = self.make_es_cls()
        assert future.size(price=2842) == 2842 * future.multiplier

    def test_short_symbol(self):
        future = self.make_es_cls()
        assert future.symbol_short == 'ES'

    def test_margin_requirement(self):
        future = self.make_es_cls()
        assert future.margin_requirement == 0.1

    def test_cash_requirement(self):
        future = self.make_es_cls()
        assert future.cash_requirement == 0.0

    def test_less_then_raises_if_contracts_of_different_types(self):
        es = self.make_es_cls()
        zn = self.make_zn_cls()
        assert zn.last_trading_date < es.last_trading_date
        assert zn < es

    def test_exists_since(self):
        assert Future.exists_since == datetime(2000, 1, 1)

    def test_exists_until(self):
        expected = datetime.now().date() + timedelta(days=260 * 2)
        assert Future.exists_until == expected

    def test_lifespan(self):
        now_default = Future.now
        es = self.make_es_cls()
        Future.now = es.expiry
        assert es.lifespan() == 0
        Future.now = es.expiry - timedelta(days=14)
        assert es.lifespan() == 10
        Future.now = now_default


class TestFutureChain:
    def make_es_cls(self):
        class ES(Future):
            multiplier = 50.0
            margin_requirement = 0.1
            freq = "QE-DEC"

            def _get_expiry_date(self, year: int, month: int) -> datetime:
                return datetime(year, month, 18)

            def _get_last_trading_date(self, expiry: datetime) -> datetime:
                return datetime(expiry.year, expiry.month, 8)

        return ES

    def test_subclasses_abstract_contract(self):
        assert issubclass(FutureChain, AbstractContract)

    def test_raise_if_minimal_args_are_not_provided(self):
        with pytest.raises(ValueError):
            FutureChain()

    def test_initialization_with_futures_chain(self):
        ES = self.make_es_cls()
        future_chain = FutureChain(
            contracts=[ES(2019, 6), ES(2019, 3), ES(2019, 9)]
        )
        assert future_chain.contracts == [ES(2019, 3), ES(2019, 6), ES(2019, 9)]

    def test_initialization_without_futures_chain_start_end_provided(self):
        ES = self.make_es_cls()
        future_chain = FutureChain(
            future_cls=ES, start=datetime(2019, 1, 1), end=datetime(2020, 1, 1)
        )
        assert future_chain.contracts == [ES(2019, 3), ES(2019, 6), ES(2019, 9), ES(2019, 12)]

    def test_initialization_without_futures_chain_start_end_not_provided(self):
        ES = self.make_es_cls()
        future_chain = FutureChain(future_cls=ES)
        expected = [ES(t.year, t.month) for t in pd.date_range(start=ES.exists_since, end=ES.exists_until, freq=ES.freq)]
        assert future_chain.contracts == expected

    def test_multiplier(self):
        ES = self.make_es_cls()
        future_chain = FutureChain(ES, "2019-01-01", "2019-12-31")
        assert future_chain.multiplier == 50

    def test_short_symbol(self):
        ES = self.make_es_cls()
        future_chain = FutureChain(ES, "2019-01-01", "2019-12-31")
        assert future_chain.symbol_short == 'ES'

    def test_symbol(self):
        ES = self.make_es_cls()
        future_chain = FutureChain(ES, "2019-01-01", "2019-12-31")
        assert future_chain.symbol == "ESH19"
        future_chain.now = datetime(2019, 11, 1)
        assert future_chain.symbol == "ESZ19"
        future_chain.now = datetime(2019, 5, 1)
        assert future_chain.symbol == "ESM19"
        future_chain.now = datetime.min
        assert future_chain.symbol == "ESH19"
    
    def test_lead_contract(self):
        ES = self.make_es_cls()
        future_chain = FutureChain(ES, "2019-01-01", "2020-12-31")
        assert future_chain.lead_contract() == ES(2019, 3)
        future_chain.now = datetime(2019, 11, 1)
        assert future_chain.lead_contract() == ES(2019, 12)
        future_chain.now = datetime(2019, 5, 1)
        assert future_chain.lead_contract() == ES(2019, 6)
        future_chain.now = datetime.min
        assert future_chain.lead_contract() == ES(2019, 3)

    def test_lead_contract_when_month_is_specified(self):
        ES = self.make_es_cls()
        future_chain = FutureChain(ES, "2019-01-01", "2020-12-31", month=1)
        assert future_chain.lead_contract() == ES(2019, 6)
        future_chain.now = datetime(2019, 11, 1)
        assert future_chain.lead_contract() == ES(2020, 3)
        future_chain.now = datetime(2019, 5, 1)
        assert future_chain.lead_contract() == ES(2019, 9)
        future_chain.now = datetime.min
        assert future_chain.lead_contract() == ES(2019, 6)

    def test_lead_contract_now_is_provided(self):
        ES = self.make_es_cls()
        future_chain = FutureChain(ES, "2019-01-01", "2019-12-31")
        assert future_chain.lead_contract(future_chain.now) == ES(2019, 3)
        assert future_chain.lead_contract(datetime(2019, 11, 1)) == ES(2019, 12)
        assert future_chain.lead_contract(datetime(2019, 5, 1)) == ES(2019, 6)
        assert future_chain.lead_contract(datetime.min) == ES(2019, 3)

    def test_make_events(self):
        ES = self.make_es_cls()
        future_chain = FutureChain(ES, "2019-01-01", "2019-12-31")
        actual = future_chain.make_events()
        expected = [
            EventContractDiscontinued(contract=ES(2019, 3), time=ES(2019, 3).expiry),
            EventContractDiscontinued(contract=ES(2019, 6), time=ES(2019, 6).expiry),
            EventContractDiscontinued(contract=ES(2019, 9), time=ES(2019, 9).expiry),
            EventContractDiscontinued(contract=ES(2019, 12), time=ES(2019, 12).expiry),
        ]
        assert actual == expected

    def test_size(self):
        ES = self.make_es_cls()
        future_chain = FutureChain(ES, "2019-01-01", "2019-12-31")
        assert future_chain.size(2847.25) == 2847.25 * ES.multiplier

    def test_underlyings(self):
        ES = self.make_es_cls()
        future_chain = FutureChain(ES, "2019-01-01", "2019-12-31")
        assert future_chain.underlyings == [
            ES(2019, month) for month in [3, 6, 9, 12]
        ]

    def test_margin_requirement(self):
        ES = self.make_es_cls()
        future_chain = FutureChain(ES, "2019-01-01", "2019-12-31")
        assert future_chain.margin_requirement == 0.1

    def test_cash_requirement(self):
        ES = self.make_es_cls()
        future_chain = FutureChain(ES, "2019-01-01", "2019-12-31")
        assert future_chain.cash_requirement == 0.0

    def test_lifespan(self):
        now_default = Future.now
        ES = self.make_es_cls()
        future_chain = FutureChain(ES, "2019-01-01", "2019-12-31")
        Future.now = future_chain.lead_contract().expiry
        assert future_chain.lifespan() == 0
        Future.now = future_chain.lead_contract().expiry - timedelta(days=14)
        assert future_chain.lifespan() == 10
        Future.now = now_default

    def test_static_hashing(self):
        now_default = Future.now
        ES = self.make_es_cls()
        future_chain = FutureChain(ES, "2019-01-01", "2019-12-31")
        assert future_chain.static_hashing() == ES(2019, 3)
        future_chain.now = datetime(2019, 7, 3)
        assert future_chain.static_hashing() == ES(2019, 9)
        Future.now = now_default


@pytest.mark.parametrize('year, month, expected', [
    (2020, 10, datetime(2020, 10, 21)),
    (2020, 11, datetime(2020, 11, 18)),
    (2020, 12, datetime(2020, 12, 16)),
    (2021, 1, datetime(2021, 1, 20)),
    (2021, 2, datetime(2021, 2, 17)),
    (2021, 3, datetime(2021, 3, 17)),
    (2021, 4, datetime(2021, 4, 21)),
    (2021, 5, datetime(2021, 5, 19)),
])
def test_vx_expiration(year, month, expected):
    # https://cdn.cboe.com/resources/aboutcboe/Cboe2020FUTURESCalendar.pdf
    # https://cdn.cboe.com/resources/aboutcboe/Cboe2021FUTURESCalendar.pdf
    vx = VX(year, month)
    assert vx.expiry == expected
