from tradingenv.broker.rebalancing import Rebalancing
from tradingenv.broker.broker import Broker, Exchange
from tradingenv.events import EventNBBO
from tradingenv.contracts import Cash, Rate, ETF
from typing import List
import pandas as pd
from datetime import datetime
import pytest


class TestTrackRecord:
    def make_exchange(self, nbbo_events: List[EventNBBO] = None):
        if nbbo_events is None:
            nbbo_events = list()
        nbbo_events += [
            EventNBBO(datetime.now(), Cash(), 1, 1),
            EventNBBO(datetime.now(), Rate("FED funds rate"), 0.0, 0.0),
            EventNBBO(datetime.now(), ETF("SPY"), 145, 146),
            EventNBBO(datetime.now(), ETF("IEF"), 45, 46),
        ]
        exchange = Exchange()
        for nbbo in nbbo_events:
            exchange.process_EventNBBO(nbbo)
        return exchange

    def make_broker(self):
        return Broker(exchange=self.make_exchange())

    def test_initialization(self):
        broker = self.make_broker()
        assert broker.track_record._time == list()
        assert broker.track_record._rebalancing == dict()
        assert broker.track_record._nr_steps_to_burn == 0

    def test_checkpoint(self):
        broker = self.make_broker()
        rebalancing1 = Rebalancing(time=datetime(2019, 1, 1))
        broker.rebalance(rebalancing1)
        assert broker.track_record._time == [datetime(2019, 1, 1)]
        assert broker.track_record._rebalancing == {datetime(2019, 1, 1): rebalancing1}

        rebalancing2 = Rebalancing(time=datetime(2019, 1, 2))
        broker.rebalance(rebalancing2)
        assert broker.track_record._time == [datetime(2019, 1, 1), datetime(2019, 1, 2)]
        assert broker.track_record._rebalancing == {
            datetime(2019, 1, 1): rebalancing1,
            datetime(2019, 1, 2): rebalancing2,
        }

    def test_checkpoint_counts_burn(self):
        broker = self.make_broker()
        broker.track_record.burn = True
        assert broker.track_record._nr_steps_to_burn == 0

        # Burn is increased because the first _rebalancing trades nothing.
        rebalancing = Rebalancing(
            time=datetime(2019, 1, 1),
            contracts=[Cash(), ETF('SPY')],
            allocation=[1, 0],
        )
        broker.rebalance(rebalancing)
        assert broker.track_record._nr_steps_to_burn == 1

        # Burn is not increased because the second _rebalancing trades something.
        rebalancing = Rebalancing(
            time=datetime(2019, 1, 2),
            contracts=[Cash(), ETF('SPY')],
            allocation=[0.9, 0.1],
        )
        broker.rebalance(rebalancing)
        assert broker.track_record._nr_steps_to_burn == 1

        # Burn is not increased because we have already started trading at the
        # previous step.
        rebalancing = Rebalancing(
                time=datetime(2019, 1, 3),
                contracts=[Cash(), ETF('SPY')],
                allocation=[1, 0],
            )
        broker.rebalance(rebalancing)
        assert broker.track_record._nr_steps_to_burn == 1

    def test_checkpoint_raises_if_duplicate_time(self):
        broker = self.make_broker()
        rebalancing = Rebalancing(time=datetime(2019, 1, 1))
        broker.rebalance(rebalancing)
        with pytest.raises(ValueError):
            broker.rebalance(rebalancing)

    def test_checkpoint_converts_pandas_timestamp_to_datetime(self):
        broker = self.make_broker()
        rebalancing = Rebalancing(time=pd.Timestamp(2019, 1, 1))
        broker.rebalance(rebalancing)
        assert broker.track_record._time == [datetime(2019, 1, 1)]
        assert broker.track_record._rebalancing == {datetime(2019, 1, 1): rebalancing}

    def test_getitem_when_index_is_int(self):
        broker = self.make_broker()
        for i in range(10):
            rebalancing = Rebalancing(time=datetime(2019, 1, i + 1))
            broker.rebalance(rebalancing)
        for i in range(10):
            time = datetime(2019, 1, i + 1)
            rebalancing = broker.track_record[i]
            assert rebalancing.time == time

    def test_getitem_when_index_is_time(self):
        exchange = self.make_exchange()
        broker = Broker(exchange)
        for i in range(10):
            rebalancing = Rebalancing(time=datetime(2019, 1, i + 1))
            broker.rebalance(rebalancing)
        for i in range(10):
            time = datetime(2019, 1, i + 1)
            rebalancing = broker.track_record[time]
            assert rebalancing.time == time

    def test_len(self):
        exchange = self.make_exchange()
        broker = Broker(exchange)
        for i in range(1, 11):
            rebalancing = Rebalancing(time=datetime(2019, 1, i))
            broker.rebalance(rebalancing)
            assert len(broker.track_record) == i
