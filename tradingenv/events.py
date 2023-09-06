"""TradingEnv is an even-driven market simulator. All custom events must
inherit the class IEvent."""
import tradingenv
from tradingenv.broker import broker
from typing import Sequence
import numpy as np
from datetime import datetime
from abc import ABC
from typing import Dict
import inspect


class Observer(ABC):
    """
    Attributes
    ----------
    _observed_events : Sequence[Type['tradingenv.events.IEvent']]
        A sequence of class Events.  The state of a Feature is updated
        automatically by TradingEnv whenever an observed event occurs
        (i.e. it is passed to TradingEnv.add_event). By default, no event
        are observed.

    Notes
    -----
    Generally in the observer patterns we store the functions to be run
    whenever an observed event is processed. Here we register the method id
    instead, which is slightly slower and certainly less elegant. The reason
    is that deepcopies of the env will introduce bucks because features will
    be copied but not their callback methods from the observed_events, which
    might introduce nasty bugs (e.g. when using ray).

    Ideas for alternative implementations of the observer pattern:
        https://stackoverflow.com/questions/1092531/event-system-in-python
    """

    def __new__(cls, *args, **kwargs) -> 'Observer':
        observer = super().__new__(cls)
        observer.last_update: datetime = None
        observer._nr_callbacks = 0
        observer.name = cls.__name__
        observer._observed_events = observer._get_observed_events()
        observer._init_args = args
        observer._init_kwargs = kwargs
        return observer

    def __call__(self, *args, **kwargs):
        """Called as soon as the Observer has finished processing a whatever
        event. Implement if you need to add extra logic (e.g. checkpoint in
        history current state of the observer/feature/state."""

    def reset(self):
        """Re-initialise from scratch, discard attributes that don't come
        with the initialisation."""
        # TODO: test
        self.__init__(*self._init_args, **self._init_kwargs)
        self.last_update = None
        self._nr_callbacks = 0

    def _get_observed_events(self) -> Dict[str, str]:
        """Infer observed events from class methods starting with 'process_'."""
        observed_events = dict()
        for attr_name in dir(self.__class__):
            if attr_name.startswith("process_"):
                event_name = attr_name.replace("process_", "")
                self._verify_callback_signature(event_name)
                observed_events[event_name] = attr_name
        return observed_events

    def _verify_is_observing(self):
        if len(self._observed_events) == 0:
            raise AttributeError(
                "You must implement at least one {}.process_<EventName>"
                "".format(self.name)
            )

    def _verify_callback_signature(self, event_name: str):
        """Verify that callback methods accept argument 'event' only."""
        callback_name = "process_{}".format(event_name)
        callback = getattr(self, callback_name)
        if list(inspect.signature(callback).parameters) != ["event"]:
            raise TypeError(
                "{}.{} must accept the single argument 'event'."
                "".format(self.__class__.__name__, callback_name)
            )


class IEvent:
    """TradingEnv is an even-driven simulator where events are passed using
    TradingEnv.add_event. All events must inherit this class.

    Attributes
    ----------
    time : datetime
        Timestamp associated with the event, i.e. when the event occurred.
    """
    time: datetime = None  # class attribute set during instance initialization

    def __repr__(self) -> str:
        return "{}({})".format(self.__class__.__name__, self.time)

    def __le__(self, other: "IEvent") -> bool:
        """Events are sorted by time."""
        return self.time <= other.time

    def __lt__(self, other: "IEvent") -> bool:
        """Events are sorted by time."""
        return self.time < other.time

    def __eq__(self, other: "IEvent") -> bool:
        """Two events are equal if and only if all attributes are equal."""
        return self._public_attr(self) == self._public_attr(other)

    def _public_attr(self, obj: object) -> dict:
        """Returns vars(obj) dropping attributes starting with '_' this
        representing private attributes."""
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}

    def notify(self, observers: Sequence['tradingenv.events.Observer']):
        for observer in observers:
            if type(self).__name__ in observer._observed_events:
                callback_name = observer._observed_events[type(self).__name__]
                callback = getattr(observer, callback_name)
                callback(self)
                observer.last_update = self.time
                observer._nr_callbacks += 1
                observer()


class EventNBBO(IEvent):
    """(Synchronous) National Best Bid and Offer."""

    __slots__ = ("contract", "bid_price", "ask_price", "bid_size", "ask_size", "time")

    def __init__(
        self,
        time: datetime,
        contract: "tradingenv.contracts.AbstractContract",
        bid_price: float,
        ask_price: float,
        bid_size: float = np.inf,
        ask_size: float = np.inf,
    ):
        """
        Parameters
        ----------
        contract : 'tradingenv.contracts.AbstractContract'
            Contract ID. Generally a string (e.g. 'S&P 500') but could be a
            whatever hashable object.
        bid_price : float
            Bid transaction_price now.
        ask_price : float
            Ask transaction_price now.
        bid_size : float
            Bid size now (top of the book).
        ask_size : float
            Ask size now (top of the book).
        time : datetime
            Timestamp associated with the NBBO. If not provided, the time will
            be assumed to be the current UTC time.
        """
        self.time = time
        self.contract = contract
        self.bid_price = bid_price
        self.ask_price = ask_price
        self.mid_price = (bid_price + ask_price) / 2
        self.bid_size = bid_size
        self.ask_size = ask_size
        contract.verify(self.mid_price)

    def __repr__(self) -> str:
        return repr({attr: getattr(self, attr) for attr in self.__slots__})

    def __eq__(self, other: "EventNBBO") -> bool:
        return all(
            getattr(self, attr) == getattr(other, attr)
            for attr in self.__slots__
        )


class EventContractDiscontinued(IEvent):
    def __init__(self, time: datetime, contract: "tradingenv.contracts.AbstractContract"):
        self.time = time
        self.contract = contract


class EventReset(IEvent):
    """IEvent signaling that TradingEnv.reset has finished to run."""

    def __init__(self, time: datetime, track_record: "broker.TrackRecord"):
        self.time = time
        self.track_record = track_record


class EventStep(IEvent):
    """IEvent signaling that TradingEnv.step has finished to run."""

    def __init__(self, time: datetime, track_record: "broker.TrackRecord", action):
        self.time = time
        self.track_record = track_record
        self.action = action


class EventDone(IEvent):
    """IEvent signaling that TradingEnv.step has finished to run."""
    def __init__(self, time: datetime, broker: "broker.Broker"):
        self.time = time
        self.broker = broker


class EventNewDate(IEvent):
    """Triggered just before the first event of the date is processed."""
    def __init__(self, time: datetime, broker: "broker.Broker"):
        self.time = time
        self.broker = broker
