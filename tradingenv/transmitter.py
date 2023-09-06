"""TradingEnv must receive events in order to update its state during a
simulation. This could either be done by the client with TradingEnv.add_event
(low-level) OR by providing a AbstractTransmitter when instancing TradingEnv (
high-level, recommended)."""
from tradingenv.events import IEvent, EventNBBO
from typing import Iterable, List, Dict, Type, Sequence, Tuple
from datetime import datetime, timezone
from collections import defaultdict, OrderedDict
from abc import ABC, abstractmethod
from datetime import timedelta
import numpy as np
import pandas as pd
import itertools
import bisect
import inspect
import random

TRAINING_SET = "training-set"
TEST_SET = "test-set"


class AbstractTransmitter(ABC):
    """Implementations of this interface take care of delivering events to
    the trading environment during the interaction of the agent with the
    environment at pre-specified timesteps."""

    @abstractmethod
    def _reset(self, fold_name: str = TRAINING_SET) -> None:
        """Iterator protocol."""

    @abstractmethod
    def _next(self) -> Tuple[List[IEvent], List[IEvent]]:
        """Iterator protocol."""

    @abstractmethod
    def _create_partitions(self, latency: float=0.) -> None:
        """Create partitions of Events, sorted chronologically."""

    @abstractmethod
    def _now(self) -> datetime:
        """Returns current time."""

    @abstractmethod
    def add_events(self, events: List[IEvent]):
        """"""


class AsynchronousTransmitter(AbstractTransmitter):
    """This AbstractTransmitter does not schedule any event to the TradingEnv.
    Suitable for when deploying to production your code."""

    def __len__(self) -> float:
        """This iterator never ends (it keeps returning empty lists), so len
        returns np.inf."""
        return np.inf

    def __repr__(self) -> str:
        return "{}(events: 0, timesteps: inf)".format(self.__class__.__name__)

    def _reset(self, *args, **kwargs):
        """Iterator protocol."""

    def _next(self) -> Tuple[List[IEvent], List[IEvent]]:
        """Iterator protocol. Returns an empty list."""
        return list(), list()

    def _create_partitions(self, latency: float=0.):
        """No need to create partitions."""

    def _now(self) -> datetime:
        """Returns current UTC time as a datetime object with timezone."""
        return datetime.utcnow()#.replace(tzinfo=timezone.utc)

    def add_events(self, events: List[IEvent]):
        return list()


class PartitionTimeRanges:
    def __init__(self, folds: dict = None):
        if folds is None:
            folds = {TRAINING_SET: [datetime.min, datetime.max]}

        # Parse split to ordered dictionary sorted by start date.
        self.folds = OrderedDict(sorted(folds.items(), key=lambda x: x[1]))
        self.verify_start_before_end()
        # I'm allowing partitions to overlap as some use cases such as
        # walk-forward partitions need support for this.
        # self.verify_overlaps()

    def __getitem__(self, item: str):
        return self.folds[item]

    def verify_start_before_end(self):
        """Verify that start < end across all datasets."""
        for key, (start, end) in self.folds.items():
            if end < start:
                raise ValueError(
                    "Dataset '{}' end date {} must be more recent than the "
                    "start date {}.".format(key, start, end)
                )

    def verify_overlaps(self):
        """Verify absence of any overlap across all time windows."""
        for key1, (start1, end1) in self.folds.items():
            for key2, (start2, end2) in self.folds.items():
                if key1 != key2 and start1 <= start2 <= end1:
                    raise ValueError(
                        "Dataset '{}' ({}:{})overlaps with '{}' ({}:{})"
                        "".format(key1, start1, end1, key2, start2, end2)
                    )


class Folds:
    def __init__(
        self,
        timesteps: Sequence[datetime],
        train_start: Sequence,
        train_end: Sequence,
        test_start: Sequence,
        test_end: Sequence,
    ):
        self.timesteps = timesteps
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end

    def as_time(self):
        """Returns Folds instance storing timesteps associated with the end
        and start of each fold."""
        timesteps = np.array(self.timesteps)
        return Folds(
            timesteps=self.timesteps,
            train_start=timesteps[self.train_start],
            train_end=timesteps[self.train_end],
            test_start=timesteps[self.test_start],
            test_end=timesteps[self.test_end],
        )


class Transmitter(AbstractTransmitter):
    """AbstractTransmitter takes care of sending Event(s) to TradingEnv while
    interacting with the environment at specified timesteps. You want to use
    this class to run a backtest in order to schedule at what time the Event(s)
    (e.g. market prices or alternative data) should trigger.

    Attributes
    ----------
    events : List[Event]
        A list accumulating events to be delivered to TradingEnv during the
        interaction.
    timesteps : List[IEvent]
        A list accumulating timesteps at which to deliver events to
        TradingEnv during the interaction.
    """

    def __init__(
            self,
            timesteps: Sequence,
            folds: Dict[str, Sequence[datetime]] = None,
            markov_reset: bool = False,
            warmup: timedelta = None,
    ):
        """
        Parameters
        ----------
        timesteps : Sequence
            A sequence of timestamps (e.g. datetime, date, timestamp). Starting
            from the oldest timestamp, Transmitter next will send events in
            batches stopping at the next timestamp (chronological order) until
            when the last timestep is reached.
        folds : Dict[str, Sequence[datetime]]
            A dictionary mapping fold _names (keys) to a pair (Sequence) of
            timesteps defining the start and end time of the fold. If not
            provided, all timesteps will belong to a single fold named
            'training-set'.
        markov_reset : bool
            If False (default), all events occurring before the date of the
            reset will be processed to allow calculating derived data from past
            information. If True, past events will not be processed. Setting
            this to True will speed up the time required to reset the
            environment.

        Examples
        --------
        >>> from tradingenv.contracts import Index
        >>> returns = pd.DataFrame(
        ...     data=np.random.normal(0, 0.001, (100, 2)),
        ...     columns=[Index('S&P 500'), Index('T-Bond')],
        ...     index=pd.date_range('2019-01-01', periods=100, freq='B'),
        ... )
        >>> prices = (1 + returns).cumprod()
        >>>
        >>> # All prices belong to the training-set.
        >>> transmitter = Transmitter(timesteps=prices.index)
        >>> transmitter.add_prices(prices)

        >>> # Partition prices between training and test set.
        >>> transmitter = Transmitter(
        ...     timesteps=prices.index,
        ...     folds={
        ...         'training-set': [datetime(2019, 1, 1), datetime(2019, 3, 1)],
        ...         'test-set': [datetime(2019, 3, 2), datetime(2019, 5, 1)],
        ...     },
        ... )
        >>> transmitter.add_prices(prices)
        >>> transmitter._create_partitions()
        >>> transmitter._reset(fold_name='test-set')

        >>> # Retrieve batches of events, then processed by TradingEnv.add_event
        >>> events_batch_0 = transmitter._next()
        >>> events_batch_1 = transmitter._next()
        """
        # TODO: add optional argument events.
        # TODO: make timesteps optional.
        # TODO: is it an issue in case of mixed pd.Timestamp and datetime?
        # Inputs.
        self.timesteps: List[datetime] = list(timesteps)
        self._folds = PartitionTimeRanges(folds)
        self._markov_reset = markov_reset
        self._warmup = warmup

        # Attributes.
        self.events: List[IEvent] = list()
        self._fold_name: str = None
        self._start_date: datetime = None
        self._end_date: datetime = None
        self._partition_nonlatent: Dict[datetime, List[IEvent]] = None

    def __len__(self) -> int:
        """Length of the iterator, corresponding to the number of timesteps."""
        return len(self._partition_nonlatent.keys())

    def __repr__(self) -> str:
        nr_timesteps = "?" if self._partition_nonlatent is None else len(self._partition_nonlatent.keys())
        return "{cls}(events: {nr_events}, timesteps: {nr_timesteps})".format(
            cls=self.__class__.__name__,
            nr_events=len(self.events),
            nr_timesteps=nr_timesteps,
        )

    def _reset(self, fold_name: str = TRAINING_SET, episode_length: int = None, sampling_span: int = None):
        """
        Parameters
        ----------
        fold_name : str
            Fold id must have been specified in Transmitter.folds, which is
            'training-set' by default.
        episode_length : int
            Transmitter.next will stop returning events episode_length is
            reached. By default, there is no maximum length.
            If you are training reinforcement learning agents, setting this
            parameter might be good practice when episodes are very long
            (continuing tasks) or you want you use a RNN policy as training
            from long sequences might be harder for back-propagation through
            time.
        """
        if self._partition_nonlatent is None:
            self._create_partitions()
        start_date, end_date = self._folds[fold_name]
        timesteps = set(self._partition_nonlatent) | set(self._partition_latent)
        steps = np.sort(list(timesteps))
        steps = steps[start_date <= steps]
        steps = steps[steps <= end_date]
        if episode_length is not None:
            start_dates = steps[: -(episode_length - 1)]
            if sampling_span is not None:
                gamma = 1 - (1 / sampling_span)
                p = gamma ** np.arange(len(start_dates))[::-1]
                p /= p.sum()
            else:
                p = None
            start_date_idx = np.random.choice(range(len(start_dates)), p=p)
            end_date_idx = start_date_idx + episode_length - 1
            start_date = steps[start_date_idx]
            end_date = steps[end_date_idx]
            steps = steps[start_date_idx : end_date_idx + 1]

        # Done. Save to attributes.
        self._step_nr = 0
        self._current_time = np.nan
        self._steps = steps
        self._start_date = start_date
        self._end_date = end_date
        self._fold_name = fold_name

    def _next(self) -> Tuple[List[IEvent], List[IEvent]]:
        """Latent events front run the action to simulate a latency. Then the
        action if reflected in the trading environment, then non-latent events
        are run, then the new state is returned."""
        try:
            self._current_time = self._steps[self._step_nr]
        except IndexError:
            raise StopIteration()
        self._step_nr += 1
        if (self._step_nr == 1) and (not self._markov_reset):
            origin = (self._current_time - self._warmup) if self._warmup else datetime.min
            #start_date, end_date = self._folds[self._fold_name]
            events_latent = [
                e for t, e in self._partition_latent.items()
                if origin <= t <= self._current_time
            ]
            events_latent = list(itertools.chain(*events_latent))
            events_nonlatent = [
                events for t, events in self._partition_nonlatent.items()
                if origin <= t <= self._current_time
            ]
            events_nonlatent = list(itertools.chain(*events_nonlatent))
        else:
            events_latent = self._partition_latent[self._current_time]
            events_nonlatent = self._partition_nonlatent[self._current_time]
        return events_latent, events_nonlatent

    def _now(self) -> datetime:
        """Returns current UTC time as a datetime object with timezone."""
        return self._current_time

    def add_events(self, events: List[IEvent]) -> None:
        """
        Parameters
        ----------
        events : Iterable[IEvent]
            Events to be send to the environment. The time at which the Event
            will trigger corresponds to the first timestep greater or equal
            to Event.time.
        """
        self.events.extend(events)

    def add_timesteps(self, timesteps: List[datetime]) -> None:
        """
        Parameters
        ----------
        timesteps : Iterable[datetime]
            Events will be sent at the intervals specified here.
        """
        self.timesteps.extend(timesteps)

    def add_prices(self, prices: pd.DataFrame, spread: float = 0) -> None:
        """
        Parameters
        ----------
        prices : pandas.DataFrame
            A pandas.DataFrame with the following characteristics:
            - index: datetime objects
            - columns: contract _names
            - values: mid prices.
            All mid prices will be mapped to EventNBBOs to be delivered while
            interacting with TradingEnv. No timesteps are added.
        spread : float
            Spread to be applied to mid_prices (zero by default). For example,
            if spread=0.001, then there will be a 0.1% bid-ask spread
            for all prices.
        """
        prices: pd.DataFrame = prices.astype(float)
        for contract in prices.columns:
            series: pd.Series = prices[contract]
            series.dropna(inplace=True)
            for time, price in series.items():
                half_spread = price * spread / 2
                event = EventNBBO(
                    time=time,
                    contract=contract,
                    bid_price=price - half_spread,
                    ask_price=price + half_spread,
                    bid_size=np.inf,
                    ask_size=np.inf,
                )
                self.events.append(event)

    def add_custom_events(
        self,
        data: pd.DataFrame,
        event_class: Type[IEvent],
        mapping: Dict[str, str] = None,
    ):
        """
        Parameters
        ----------
        data : pandas.DataFrame
            Every row of this pandas.DataFrame will be used to create
            instance of EventType passing columns-state pairs as kwargs. The
            index of the dataframe must be of type Datetime and will be used
            to set Event.time.
        event_class : Type[Event]
            Arguments in EventType.__init__ will be searched from the columns
            of the dataframe and set accordingly.
        mapping : Dict[str, str]
            A dictionary mapping df column _names to attribute _names of
            EventType. There is no need to provide a mapping if the columns
            _names are exactly the same as the attribute _names of EventType.

        Examples
        --------
        >>> from tradingenv.events import IEvent
        >>> from tradingenv.transmitter import Transmitter
        >>> from datetime import datetime
        >>>
        >>> class NewsSentiment(IEvent):
        ...     def __init__(self, headline: str, sentiment: float):
        ...         self.headline = headline
        ...         self.sentiment = sentiment
        >>>
        >>> dataset = pd.DataFrame(
        ...     data=[
        ...         ['This is a very good news!', +0.7],
        ...         ['A kind of bad stuff has happened', -0.5],
        ...         ['Neutral news', 0],
        ...     ],
        ...     columns=['headline', 'sentiment'],
        ...     index=[
        ...         datetime(2019, 1, 1),
        ...         datetime(2019, 1, 1),
        ...         datetime(2019, 1, 2),
        ... ])
        >>> transmitter = Transmitter(timesteps=dataset.index)
        >>> transmitter.add_custom_events(dataset, NewsSentiment)
        """
        if mapping is None:
            mapping = dict()
        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError("data.index must be of type pandas.DatetimeIndex.")
        parameters = dict(inspect.signature(event_class.__init__).parameters)
        parameters.pop("self")
        for index, row in data.iterrows():
            # If EventType requires one of the following missing keys during
            # its initialization, we use the index as a workaround.
            kwargs = row.to_dict()
            for key in ["Date", "date", "Time", "time"]:
                if key not in row.index:
                    kwargs[key] = index

            # Change attribute id according to mapping.
            for col_name, attr_name in mapping.items():
                kwargs[attr_name] = kwargs.pop(col_name)

            # Keep only relevant keys.
            kwargs = {k: v for k, v in kwargs.items() if k in parameters}

            # Create custom event.
            event = event_class(**kwargs)
            event.time = index
            self.events.append(event)

    def _min_timesteps_diff(self) -> float:
        """Returns minimum time gap in seconds between all timesteps."""
        gaps = np.diff(self.timesteps)
        return min([gap.total_seconds() for gap in gaps], default=np.inf)

    def _create_partitions(self, latency: float=0):
        """Create batches of events with bisect according to the timesteps."""
        if len(self.timesteps) == 0:
            raise ValueError(
                "Transmitter delivers events to TradingEnv at "
                "timesteps indicated by Transmitter.timesteps, "
                "which is empty. Have you called "
                "Transmitter.add_timesteps at least once?"
            )
        self.timesteps = sorted(set(self.timesteps))
        if latency >= self._min_timesteps_diff():
            raise ValueError(
                "Parameter 'latency'={}s must be smaller than the minimum "
                "time gap between each timesteps, corresponding to {}s."
                "".format(latency, self._min_timesteps_diff())
            )
        self._partition_nonlatent = defaultdict(list)
        self._partition_latent = defaultdict(list)
        # TODO: Test sorted. It guarantees that all events associated with a
        #  timestamp are sorted. This avoids possible surprises when processing
        #  events by Feature.
        events = sorted(e for e in self.events if e.time <= self.timesteps[-1])
        if self._markov_reset:
            events = [e for e in events if e.time >= self.timesteps[0]]
        for event in sorted(events):
            # Events which occur after the last timestep are never run.
            if event.time <= self.timesteps[-1]:
                # IEvent is associated to first timestep that is >= event.time
                index = bisect.bisect_left(self.timesteps, event.time)
                index_previous = index - 1
                timestep = self.timesteps[index]
                timestep_previous = self.timesteps[index_previous] if index_previous >= 0 else datetime(1800, 1, 1)  # or pd.Timestamp.min?
                sec_since_timestep = (event.time - timestep_previous).total_seconds()
                if sec_since_timestep < 0:
                    raise ValueError(
                        'Unexpected sec_since_timestep={}'
                        ''.format(sec_since_timestep)
                    )
                if sec_since_timestep <= latency:
                    self._partition_latent[timestep].append(event)
                else:
                    self._partition_nonlatent[timestep].append(event)

    def walk_forward(
        self, train_size: int = None, test_size: int = None, sliding_window: bool = True
    ) -> Folds:
        """
        Parameters
        ----------
        train_size : int
            Number of observations in the training folds.
        test_size : int
            Number of observation in the validation folds.
        sliding_window : bool
            If True (default), training folds will be computed using a sliding
            window with the same number of observations. If False, an
            expanding window will be used instead, so each training fold will
            start from the first timestep which a monotonically increasing
            number of observations.

        Notes
        -----
        By design, the step size of the window is set to be 'test_size'.
        Reasons are:
        - if 'step'<'test_size': your validation set is not using all available
        information.
        - if 'step'>'test_size': there is the risk of look-ahead bias because
        the next training fold will be validated on a portion of previously
        seen validation data.

        Returns
        -------
        A Folds object.
        """
        count = np.arange(len(self.timesteps))
        train_start = count[: -train_size - test_size + 1 : test_size]
        return Folds(
            timesteps=self.timesteps,
            train_start=train_start * int(sliding_window),
            train_end=train_start + train_size - 1,
            test_start=train_start + train_size,
            test_end=train_start + train_size + test_size - 1,
        )
