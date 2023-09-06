from tradingenv.events import IEvent, EventNBBO, EventReset, EventStep
from tradingenv.contracts import Index
from datetime import datetime, timezone
import pytest
import numpy as np
import time
import pickle


class TestEvent:
    def test_subclasses_Event(self):
        assert issubclass(IEvent, IEvent)

    def test_initialization(self):
        event = IEvent()
        assert event.time is None  # set during initialization of subclass

    def test_construction(self):
        """We want the attributes to be set in __new__ and not __init__ so
        the user will not have to remember to call super().__init__() when
        overriding IEvent.__init__."""

        class EventChild(IEvent):
            def __init__(self, time):
                self.time = time

        event_child = EventChild(datetime(2019, 1, 1))
        assert event_child.time == datetime(2019, 1, 1)

    def test_eq_is_false_when_time_is_different(self):
        event1 = IEvent()
        event1.time = datetime(2019, 1, 1)
        event2 = IEvent()
        event2.time = datetime(2019, 1, 2)
        assert event1 != event2

    def test_eq_is_true_when_all_attributes_are_equal(self):
        event1 = IEvent()
        event1.time = datetime(2019, 1, 1)
        event2 = IEvent()
        event2.time = datetime(2019, 1, 1)
        assert event1 == event2

    def test_eq_is_false_when_at_least_one_attribute_is_different(self):
        event1 = IEvent()
        event1.time = datetime(2019, 1, 1)
        event1.attr = "attr1"
        event2 = IEvent()
        event2.time = datetime(2019, 1, 1)
        event2.attr = "attr2"
        assert event1 != event2

    def test_events_are_sorted_by_time(self):
        # Events are sorted by time, so let's sleep a bit to make sure that
        # their attribute 'time' is strictly increasing over time.
        event1 = IEvent()
        event1.time = datetime.now()
        time.sleep(0.001)
        event2 = IEvent()
        event2.time = datetime.now()
        time.sleep(0.001)
        event3 = IEvent()
        event3.time = datetime.now()
        actual = sorted([event2, event3, event1])
        expected = [event1, event2, event3]
        assert actual == expected

    def test_event_is_pickable(self):
        event = IEvent()
        event_pickled = pickle.dumps(event)
        event_unpickled = pickle.loads(event_pickled)
        assert event.__dict__ == event_unpickled.__dict__


class TestEventNBBO:
    def test_is_subclassing_Event(self):
        assert issubclass(EventNBBO, IEvent)

    def test_lazy_initialization(self):
        now = datetime.utcnow()
        event = EventNBBO(now, Index("S&P 500"), 1, 2)
        assert event.contract == Index("S&P 500")
        assert event.bid_price == 1
        assert event.ask_price == 2
        assert event.mid_price == 1.5
        assert event.bid_size == np.inf
        assert event.ask_size == np.inf
        assert event.time == now

    def test_verbose_initialization(self):
        now = datetime.utcnow()
        event = EventNBBO(now, Index("S&P 500"), 1, 2, 3, 4)
        assert event.contract == Index("S&P 500")
        assert event.bid_price == 1
        assert event.ask_price == 2
        assert event.bid_size == 3
        assert event.ask_size == 4
        assert event.time == now

    def test_eq_when_true(self):
        event1 = EventNBBO(datetime(2019, 1, 1), Index("S&P 500"), 1, 2, 3, 4)
        event2 = EventNBBO(datetime(2019, 1, 1), Index("S&P 500"), 1, 2, 3, 4)
        assert event1 == event2

    @pytest.mark.parametrize(
        "event",
        [
            EventNBBO(datetime(2019, 1, 1), Index("T-Bills"), 1, 2, 3, 4),
            EventNBBO(datetime(2019, 1, 1), Index("S&P 500"), 42, 2, 3, 4),
            EventNBBO(datetime(2019, 1, 1), Index("S&P 500"), 1, 42, 3, 4),
            EventNBBO(datetime(2019, 1, 1), Index("S&P 500"), 1, 2, 42, 4),
            EventNBBO(datetime(2019, 1, 1), Index("S&P 500"), 1, 2, 3, 42),
        ],
    )
    def test_eq_when_false(self, event):
        event_base = EventNBBO(datetime(2019, 1, 1), Index("S&P 500"), 1, 2, 3, 4)
        assert event_base != event


class TestEventReset:
    def test_is_subclassing_Event(self):
        assert issubclass(EventReset, IEvent)


class TestEventAction:
    def test_is_subclassing_Event(self):
        assert issubclass(EventStep, IEvent)
