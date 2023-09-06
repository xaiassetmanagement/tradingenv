from tradingenv.transmitter import AbstractTransmitter, Transmitter
from tradingenv.contracts import Index
from tradingenv.events import IEvent, EventNBBO
from datetime import datetime, timedelta
import pandas as pd
import pytest
import numpy as np
import itertools


class MarketEvent(IEvent):
    """Concrete event with timestamp, used for testing."""

    def __init__(self, time: datetime):
        self.time = time


class TestTransmitter:
    def test_parent_class_is_Transmitter(self):
        assert issubclass(Transmitter, AbstractTransmitter)

    def test_add_events(self):
        event1 = MarketEvent(datetime(2019, 1, 1))
        event2 = MarketEvent(datetime(2019, 1, 2))
        event3 = MarketEvent(datetime(2019, 1, 2))
        transmitter = Transmitter([datetime(2019, 1, 1, 12), datetime(2019, 1, 2)])
        transmitter.add_events([event1, event2, event3])
        transmitter._create_partitions()
        transmitter._reset()
        assert transmitter._now() is np.nan
        assert transmitter._next() == ([], [event1])
        assert transmitter._now() == datetime(2019, 1, 1, 12)
        assert transmitter._next() == ([], [event2, event3])
        assert transmitter._now() == datetime(2019, 1, 2)
        with pytest.raises(StopIteration):
            transmitter._next()

    def test_markov_walk_forward_partition(self):
        event1 = MarketEvent(datetime(2019, 1, 1))
        event2 = MarketEvent(datetime(2019, 1, 2))
        event3 = MarketEvent(datetime(2019, 1, 3))
        transmitter = Transmitter(
            timesteps=[datetime(2019, 1, 1), datetime(2019, 1, 2), datetime(2019, 1, 3)],
            folds={
                "2019": [datetime(2019, 1, 1), datetime(2019, 1, 2)],
                "2020": [datetime(2019, 1, 2), datetime(2019, 1, 3)],
            },
            markov_reset=True,
        )
        transmitter.add_events([event1, event2, event3])
        transmitter._reset('2020')
        _, events = transmitter._next()
        assert events == [event2]  #and not [event1]

    @pytest.mark.parametrize("markov_reset,expected", [
        [False, [datetime(2019, 1, 1), datetime(2019, 1, 2)]],
        [True, [datetime(2019, 1, 2)]]
    ])
    def test_markov_reset(self, markov_reset, expected):
        event1 = MarketEvent(datetime(2019, 1, 1))
        event2 = MarketEvent(datetime(2019, 1, 2))
        event3 = MarketEvent(datetime(2019, 1, 3))
        timesteps = [datetime(2019, 1, 2), datetime(2019, 1, 3)]
        transmitter = Transmitter(timesteps, markov_reset=markov_reset)
        transmitter.add_events([event1, event2, event3])
        transmitter._reset()
        [], events = transmitter._next()
        actual = [event.time for event in events]
        assert actual == expected

    def test_create_partitions_ignores_events_after_last_timestep(self):
        event = MarketEvent(datetime(2019, 1, 2))
        transmitter = Transmitter([datetime(2019, 1, 1)])
        transmitter.add_events([event])
        transmitter._create_partitions()  # does not raise
        assert transmitter._partition_nonlatent == {}

    def test_duplicate_timesteps_are_dropped(self):
        event1 = MarketEvent(datetime(2019, 1, 1))
        event2 = MarketEvent(datetime(2019, 1, 2))
        event3 = MarketEvent(datetime(2019, 1, 2))
        transmitter = Transmitter(
            [
                datetime(2019, 1, 1, 12),
                datetime(2019, 1, 2),
                datetime(2019, 1, 2),  # DUPLICATE
            ]
        )
        transmitter.add_events([event1, event2, event3])
        transmitter._create_partitions()
        transmitter._reset()
        assert transmitter._now() is np.nan
        assert transmitter._next() == ([], [event1])
        assert transmitter._now() == datetime(2019, 1, 1, 12)
        assert transmitter._next() == ([], [event2, event3])
        assert transmitter._now() == datetime(2019, 1, 2)
        with pytest.raises(StopIteration):
            transmitter._next()

    def test_add_disordered_events(self):
        event1 = MarketEvent(datetime(2019, 1, 1))
        event2 = MarketEvent(datetime(2019, 1, 2))
        event3 = MarketEvent(datetime(2019, 1, 2))
        transmitter = Transmitter([datetime(2019, 1, 1, 12), datetime(2019, 1, 2)])
        transmitter.add_events([event2, event1, event3])
        transmitter._create_partitions()
        transmitter._reset()
        assert transmitter._now() is np.nan
        assert transmitter._next() == ([], [event1])
        assert transmitter._now() == datetime(2019, 1, 1, 12)
        assert transmitter._next() == ([], [event2, event3])
        assert transmitter._now() == datetime(2019, 1, 2)
        with pytest.raises(StopIteration):
            transmitter._next()

    def test_add_events_without_timesteps_raises_error(self):
        event1 = MarketEvent(datetime(2019, 1, 1))
        event2 = MarketEvent(datetime(2019, 1, 2))
        event3 = MarketEvent(datetime(2019, 1, 2))
        transmitter = Transmitter(timesteps=list())
        transmitter.add_events([event2, event1, event3])
        with pytest.raises(ValueError):
            transmitter._create_partitions()

    def test_add_disordered_timesteps_and_disordered_events(self):
        # Events are sorted by time, so let's sleep a bit to make sure that
        # their attribute 'time' is strictly increasing over time.
        event1 = MarketEvent(datetime(2019, 1, 1))
        event2 = MarketEvent(datetime(2019, 1, 2))
        event3 = MarketEvent(datetime(2019, 1, 2))
        transmitter = Transmitter([datetime(2019, 1, 2), datetime(2019, 1, 1, 12)])
        transmitter.add_events([event3, event1, event2])
        transmitter._create_partitions()
        transmitter._reset()
        assert transmitter._now() is np.nan
        assert transmitter._next() == ([], [event1])
        assert transmitter._now() == datetime(2019, 1, 1, 12)
        [], events = transmitter._next()
        assert (events == [event2, event3]) or (events == [event3, event2])
        assert transmitter._now() == datetime(2019, 1, 2)
        with pytest.raises(StopIteration):
            transmitter._next()

    def test_sequence_resets_now_after_iteration_end(self):
        event1 = MarketEvent(datetime(2019, 1, 1))
        event2 = MarketEvent(datetime(2019, 1, 2))
        event3 = MarketEvent(datetime(2019, 1, 2))
        transmitter = Transmitter([datetime(2019, 1, 1, 12), datetime(2019, 1, 2)])
        transmitter.add_events([event1, event2, event3])
        transmitter._create_partitions()
        for _ in range(10):
            transmitter._reset()
            assert transmitter._now() is np.nan
            assert transmitter._next() == ([], [event1])
            assert transmitter._now() == datetime(2019, 1, 1, 12)
            assert transmitter._next() == ([], [event2, event3])
            assert transmitter._now() == datetime(2019, 1, 2)
            with pytest.raises(StopIteration):
                transmitter._next()

    def test_add_prices(self):
        prices = pd.DataFrame(
            data=[[1, 2], [np.nan, 4], [5, "6"], [None, 8]],
            index=[
                datetime(2019, 1, 5),
                datetime(2019, 1, 7),
                datetime(2019, 1, 9),
                datetime(2019, 1, 11),
            ],
            columns=[Index("S&P 500"), Index("T-bills")],
        )
        transmitter = Transmitter([datetime(2019, 1, 7, 12), datetime(2019, 1, 11)])
        transmitter.add_prices(prices=prices, spread=0.01)
        transmitter._create_partitions()

        # Test.
        transmitter._reset()
        assert transmitter._now() is np.nan
        assert transmitter._next() == ([], [
            EventNBBO(datetime(2019, 1, 5), Index("S&P 500"), 0.995, 1.005, np.inf, np.inf),
            EventNBBO(datetime(2019, 1, 5), Index("T-bills"), 1.990, 2.010, np.inf, np.inf),
            EventNBBO(datetime(2019, 1, 7), Index("T-bills"), 3.980, 4.020, np.inf, np.inf),
        ])
        assert transmitter._now() == datetime(2019, 1, 7, 12)
        assert transmitter._next() == ([], [
            EventNBBO(datetime(2019, 1, 9), Index("S&P 500"), 4.975, 5.025, np.inf, np.inf),
            EventNBBO(datetime(2019, 1, 9), Index("T-bills"), 5.970, 6.030, np.inf, np.inf),
            EventNBBO(datetime(2019, 1, 11), Index("T-bills"), 7.960, 8.040, np.inf, np.inf),
        ])
        assert transmitter._now() == datetime(2019, 1, 11)
        with pytest.raises(StopIteration):
            transmitter._next()

    def test_add_custom_events_with_time_in_init_args(self):
        class NewsSentiment(IEvent):
            def __init__(self, time: datetime, headline: str, sentiment: float):
                self.time = time
                self.headline = headline
                self.sentiment = sentiment

            def __eq__(self, other: "NewsSentiment"):
                return all(
                    [
                        self.time == other.time,
                        self.headline == other.headline,
                        self.sentiment == other.sentiment,
                    ]
                )

        dataset = pd.DataFrame(
            data=[
                ["This is a very good news!", +0.7],
                ["A kind of bad stuff has happened", -0.5],
                ["Neutral news", 0],
            ],
            columns=["headline", "sentiment"],
            index=[datetime(2019, 1, 1), datetime(2019, 1, 1), datetime(2019, 1, 2)],
        )
        transmitter = Transmitter(timesteps=dataset.index)
        transmitter.add_custom_events(dataset, NewsSentiment)
        transmitter._create_partitions()
        transmitter._reset()

        pre_latency, actual = transmitter._next()
        assert pre_latency == list()
        expected = [
            NewsSentiment(datetime(2019, 1, 1), "This is a very good news!", 0.7),
            NewsSentiment(
                datetime(2019, 1, 1), "A kind of bad stuff has happened", -0.5
            ),
        ]
        assert actual == expected

        pre_latency, actual = transmitter._next()
        assert pre_latency == list()
        expected = [NewsSentiment(datetime(2019, 1, 2), "Neutral news", 0)]
        assert actual == expected
        with pytest.raises(StopIteration):
            transmitter._next()

    def test_add_custom_events_without_time_in_init_args(self):
        class NewsSentiment(IEvent):
            def __init__(self, headline: str, sentiment: float):
                self.headline = headline
                self.sentiment = sentiment

            def __eq__(self, other: "NewsSentiment"):
                return all(
                    [
                        self.time == other.time,
                        self.headline == other.headline,
                        self.sentiment == other.sentiment,
                    ]
                )

        dataset = pd.DataFrame(
            data=[
                ["This is a very good news!", +0.7],
                ["A kind of bad stuff has happened", -0.5],
                ["Neutral news", 0],
            ],
            columns=["headline", "sentiment"],
            index=[datetime(2019, 1, 1), datetime(2019, 1, 1), datetime(2019, 1, 2)],
        )
        transmitter = Transmitter(timesteps=dataset.index)
        transmitter.add_custom_events(dataset, NewsSentiment)
        transmitter._create_partitions()
        transmitter._reset()

        event1 = NewsSentiment("This is a very good news!", 0.7)
        event1.time = datetime(2019, 1, 1)
        event2 = NewsSentiment("A kind of bad stuff has happened", -0.5)
        event2.time = datetime(2019, 1, 1)
        expected = [event1, event2]
        pre_latency, actual = transmitter._next()
        assert pre_latency == list()
        assert actual == expected

        event = NewsSentiment("Neutral news", 0)
        event.time = datetime(2019, 1, 2)
        pre_latency, actual = transmitter._next()
        assert pre_latency == list()
        expected = [event]
        assert actual == expected

        with pytest.raises(StopIteration):
            transmitter._next()

    def test_add_custom_events_raises_if_index_is_not_datetime(self):
        class NewsSentiment(IEvent):
            def __init__(self, time: datetime, headline: str, sentiment: float):
                self.time = time
                self.headline = headline
                self.sentiment = sentiment

            def __eq__(self, other: "NewsSentiment"):
                return all(
                    [
                        self.time == other.time,
                        self.headline == other.headline,
                        self.sentiment == other.sentiment,
                    ]
                )

        dataset = pd.DataFrame(
            data=[
                ["This is a very good news!", +0.7],
                ["A kind of bad stuff has happened", -0.5],
                ["Neutral news", 0],
            ],
            columns=["headline", "sentiment"],
            index=[1, 2, 3],  # should be datetime, so TypeError is expected
        )
        transmitter = Transmitter(timesteps=dataset.index)
        with pytest.raises(TypeError):
            # TypeError('df.index must be of type pandas.DatetimeIndex.')
            transmitter.add_custom_events(dataset, NewsSentiment)

    def test_add_custom_events_uses_provided_mapping(self):
        class NewsSentiment(IEvent):
            def __init__(self, time: datetime, headline: str, sentiment: float):
                self.time = time
                self.headline = headline
                self.sentiment = sentiment

            def __eq__(self, other: "NewsSentiment"):
                return all(
                    [
                        self.time == other.time,
                        self.headline == other.headline,
                        self.sentiment == other.sentiment,
                    ]
                )

        dataset = pd.DataFrame(
            data=[
                ["This is a very good news!", +0.7],
                ["A kind of bad stuff has happened", -0.5],
                ["Neutral news", 0],
            ],
            columns=["col1", "col2"],
            index=[datetime(2019, 1, 1), datetime(2019, 1, 1), datetime(2019, 1, 2)],
        )
        transmitter = Transmitter(timesteps=dataset.index)
        transmitter.add_custom_events(
            dataset, NewsSentiment, mapping={"col1": "headline", "col2": "sentiment"}
        )
        transmitter._create_partitions()
        transmitter._reset()

        pre_latency, actual = transmitter._next()
        assert pre_latency == list()
        expected = [
            NewsSentiment(datetime(2019, 1, 1), "This is a very good news!", 0.7),
            NewsSentiment(
                datetime(2019, 1, 1), "A kind of bad stuff has happened", -0.5
            ),
        ]
        assert actual == expected

        pre_latency, actual = transmitter._next()
        assert pre_latency == list()
        expected = [NewsSentiment(datetime(2019, 1, 2), "Neutral news", 0)]
        assert actual == expected

    def test_split_date(self):
        prices = pd.DataFrame(
            data=[[1, 2], [3, 4], [5, 6], [10, 11], [20, 21], [22, 23]],
            index=[
                datetime(2019, 1, 1),
                datetime(2019, 1, 2),
                datetime(2019, 1, 3),
                datetime(2019, 1, 4),
                datetime(2019, 1, 5),
                datetime(2019, 1, 6),
            ],
            columns=[Index("S&P 500"), Index("T-bills")],
        )
        transmitter = Transmitter(
            timesteps=prices.index,
            folds={
                "training-set": (datetime(2019, 1, 1), datetime(2019, 1, 3)),
                "validation-set": (datetime(2019, 1, 4), datetime(2019, 1, 4)),
                "test-set": (datetime(2019, 1, 5), datetime(2019, 1, 6)),
            },
        )
        transmitter.add_prices(prices)
        transmitter._create_partitions()

        # Reset uses training data by default..
        transmitter._reset()
        assert transmitter._now() is np.nan

        assert transmitter._next() == ([], [
            EventNBBO(datetime(2019, 1, 1), Index("S&P 500"), 1, 1),
            EventNBBO(datetime(2019, 1, 1), Index("T-bills"), 2, 2),
        ])
        assert transmitter._now() == datetime(2019, 1, 1)

        assert transmitter._next() == ([], [
            EventNBBO(datetime(2019, 1, 2), Index("S&P 500"), 3, 3),
            EventNBBO(datetime(2019, 1, 2), Index("T-bills"), 4, 4),
        ])
        assert transmitter._now() == datetime(2019, 1, 2)

        assert transmitter._next() == ([], [
            EventNBBO(datetime(2019, 1, 3), Index("S&P 500"), 5, 5),
            EventNBBO(datetime(2019, 1, 3), Index("T-bills"), 6, 6),
        ])
        assert transmitter._now() == datetime(2019, 1, 3)

        with pytest.raises(StopIteration):
            transmitter._next()

        # Test data.
        transmitter._reset("test-set")
        assert transmitter._now() is np.nan

        assert transmitter._next() == ([], [
            EventNBBO(datetime(2019, 1, 1), Index("S&P 500"), 1, 1),
            EventNBBO(datetime(2019, 1, 1), Index("T-bills"), 2, 2),
            EventNBBO(datetime(2019, 1, 2), Index("S&P 500"), 3, 3),
            EventNBBO(datetime(2019, 1, 2), Index("T-bills"), 4, 4),
            EventNBBO(datetime(2019, 1, 3), Index("S&P 500"), 5, 5),
            EventNBBO(datetime(2019, 1, 3), Index("T-bills"), 6, 6),
            EventNBBO(datetime(2019, 1, 4), Index("S&P 500"), 10, 10),
            EventNBBO(datetime(2019, 1, 4), Index("T-bills"), 11, 11),
            EventNBBO(datetime(2019, 1, 5), Index("S&P 500"), 20, 20),
            EventNBBO(datetime(2019, 1, 5), Index("T-bills"), 21, 21),
        ])
        assert transmitter._now() == datetime(2019, 1, 5)

        assert transmitter._next() == ([], [
            EventNBBO(datetime(2019, 1, 6), Index("S&P 500"), 22, 22),
            EventNBBO(datetime(2019, 1, 6), Index("T-bills"), 23, 23),
        ])
        assert transmitter._now() == datetime(2019, 1, 6)

        with pytest.raises(StopIteration):
            transmitter._next()

        # Validation data.
        transmitter._reset("validation-set")
        assert transmitter._now() is np.nan

        assert transmitter._next() == ([], [
            EventNBBO(datetime(2019, 1, 1), Index("S&P 500"), 1, 1),
            EventNBBO(datetime(2019, 1, 1), Index("T-bills"), 2, 2),
            EventNBBO(datetime(2019, 1, 2), Index("S&P 500"), 3, 3),
            EventNBBO(datetime(2019, 1, 2), Index("T-bills"), 4, 4),
            EventNBBO(datetime(2019, 1, 3), Index("S&P 500"), 5, 5),
            EventNBBO(datetime(2019, 1, 3), Index("T-bills"), 6, 6),
            EventNBBO(datetime(2019, 1, 4), Index("S&P 500"), 10, 10),
            EventNBBO(datetime(2019, 1, 4), Index("T-bills"), 11, 11),
        ])
        assert transmitter._now() == datetime(2019, 1, 4)

        with pytest.raises(StopIteration):
            transmitter._next()

    @pytest.mark.parametrize(
        argnames=["fold_name", "valid_origins"],
        argvalues=[
            [
                "training-set",
                [datetime(2019, 1, 1), datetime(2019, 1, 2), datetime(2019, 1, 3)],
            ],
            [
                "test-set",
                [datetime(2019, 1, 5), datetime(2019, 1, 6), datetime(2019, 1, 7)],
            ],
        ],
    )
    def test_episode_length_as_int(self, fold_name, valid_origins):
        prices = pd.DataFrame(
            data=[
                [1, 2],
                [3, 4],
                [5, 6],
                [10, 11],
                [20, 21],
                [22, 23],
                [24, 25],
                [27, 29],
            ],
            index=[
                datetime(2019, 1, 1),
                datetime(2019, 1, 2),
                datetime(2019, 1, 3),
                datetime(2019, 1, 4),
                datetime(2019, 1, 5),
                datetime(2019, 1, 6),
                datetime(2019, 1, 7),
                datetime(2019, 1, 8),
            ],
            columns=[Index("S&P 500"), Index("T-bills")],
        )
        transmitter = Transmitter(
            timesteps=prices.index,
            folds={
                "training-set": [datetime(2019, 1, 1), datetime(2019, 1, 4)],
                "test-set": [datetime(2019, 1, 5), datetime(2019, 1, 8)],
            },
        )
        transmitter.add_prices(prices)
        transmitter._create_partitions()

        origin_list = list()
        for _ in range(10000):
            transmitter._reset(fold_name, episode_length=2)
            events_latent, events_nonlatent = transmitter._next()

            # Assert start date is within the fold.
            origin = transmitter._now()
            assert origin in valid_origins

            # Assert latent events include all events of the fold up to _now
            start_date, end_date = transmitter._folds[transmitter._fold_name]
            start_fold_to_current_times = [
                t for t in transmitter._partition_latent
                if t <= transmitter._current_time
            ]
            expected_events_latent = [transmitter._partition_latent[t] for t in start_fold_to_current_times]
            expected_events_latent = list(itertools.chain(*expected_events_latent))
            assert expected_events_latent == events_latent

            # Assert nonlatent events include all events of the fold up to _now
            start_date, end_date = transmitter._folds[transmitter._fold_name]
            start_fold_to_current_times = [
                t for t in transmitter._partition_nonlatent
                if t <= transmitter._current_time
            ]
            expected_events_nonlatent = [transmitter._partition_nonlatent[t] for t in start_fold_to_current_times]
            expected_events_nonlatent = list(itertools.chain(*expected_events_nonlatent))
            assert expected_events_nonlatent == events_nonlatent

            # Episode length is 2, so at next iteration is expected to stop.
            transmitter._next()
            assert transmitter._now() == origin + timedelta(days=1)
            with pytest.raises(StopIteration):
                transmitter._next()
            origin_list.append(origin)

        # All relative frequencies of origin must be the same.
        relative_freq = pd.Series(origin_list).value_counts(normalize=True)
        np.testing.assert_allclose(
            actual=relative_freq.values,
            desired=np.full(
                shape=(len(valid_origins),), fill_value=1 / len(valid_origins)
            ),
            rtol=0,
            atol=0.02,
        )

    def test_create_partitions_without_latency(self):
        transmitter = Transmitter(
            timesteps=[
                # Day 1.
                datetime(2019, 1, 1, 15, 59),
                datetime(2019, 1, 1, 16),
                # Day 2.
                datetime(2019, 1, 2, 15, 59),
                datetime(2019, 1, 2, 16),
            ])
        events = list()
        for t in transmitter.timesteps:
            for delay in [-1, 0, 1, 29, 30, 31]:
                events.append(MarketEvent(t+timedelta(seconds=delay)))
        transmitter.add_events(events)
        transmitter._create_partitions(latency=0)
        assert transmitter._partition_nonlatent == {
            datetime(2019, 1, 1, 15, 59): [
                MarketEvent(datetime(2019, 1, 1, 15, 58, 59)),
                MarketEvent(datetime(2019, 1, 1, 15, 59))],
            datetime(2019, 1, 1, 16, 0): [
                MarketEvent(datetime(2019, 1, 1, 15, 59, 1)),
                MarketEvent(datetime(2019, 1, 1, 15, 59, 29)),
                MarketEvent(datetime(2019, 1, 1, 15, 59, 30)),
                MarketEvent(datetime(2019, 1, 1, 15, 59, 31)),
                MarketEvent(datetime(2019, 1, 1, 15, 59, 59)),
                MarketEvent(datetime(2019, 1, 1, 16, 0))
            ],
            datetime(2019, 1, 2, 15, 59): [
                MarketEvent(datetime(2019, 1, 1, 16, 0, 1)),
                MarketEvent(datetime(2019, 1, 1, 16, 0, 29)),
                MarketEvent(datetime(2019, 1, 1, 16, 0, 30)),
                MarketEvent(datetime(2019, 1, 1, 16, 0, 31)),
                MarketEvent(datetime(2019, 1, 2, 15, 58, 59)),
                MarketEvent(datetime(2019, 1, 2, 15, 59)),
            ],
            datetime(2019, 1, 2, 16, 0): [
                MarketEvent(datetime(2019, 1, 2, 15, 59, 1)),
                MarketEvent(datetime(2019, 1, 2, 15, 59, 29)),
                MarketEvent(datetime(2019, 1, 2, 15, 59, 30)),
                MarketEvent(datetime(2019, 1, 2, 15, 59, 31)),
                MarketEvent(datetime(2019, 1, 2, 15, 59, 59)),
                MarketEvent(datetime(2019, 1, 2, 16, 0)),
            ]}

    def test_create_partitions_with_latency(self):
        transmitter = Transmitter(
            timesteps=[
                # Day 1.
                datetime(2019, 1, 1, 15, 59),
                datetime(2019, 1, 1, 16),
                # Day 2.
                datetime(2019, 1, 2, 15, 59),
                datetime(2019, 1, 2, 16),
            ])
        events = list()
        for t in transmitter.timesteps:
            for delay in [-1, 0, 1, 29, 30, 31]:
                events.append(MarketEvent(t+timedelta(seconds=delay)))
        transmitter.add_events(events)
        transmitter._create_partitions(latency=30)
        assert transmitter._partition_nonlatent == {
            datetime(2019, 1, 1, 15, 59): [
                MarketEvent(datetime(2019, 1, 1, 15, 58, 59)),
                MarketEvent(datetime(2019, 1, 1, 15, 59))],
            datetime(2019, 1, 1, 16, 0): [
                #MarketEvent(datetime(2019, 1, 1, 15, 59, 1)),
                #MarketEvent(datetime(2019, 1, 1, 15, 59, 29)),
                #MarketEvent(datetime(2019, 1, 1, 15, 59, 30)),
                MarketEvent(datetime(2019, 1, 1, 15, 59, 31)),
                MarketEvent(datetime(2019, 1, 1, 15, 59, 59)),
                MarketEvent(datetime(2019, 1, 1, 16, 0))
            ],
            datetime(2019, 1, 2, 15, 59): [
                #MarketEvent(datetime(2019, 1, 1, 16, 0, 1)),
                #MarketEvent(datetime(2019, 1, 1, 16, 0, 29)),
                #MarketEvent(datetime(2019, 1, 1, 16, 0, 30)),
                MarketEvent(datetime(2019, 1, 1, 16, 0, 31)),
                MarketEvent(datetime(2019, 1, 2, 15, 58, 59)),
                MarketEvent(datetime(2019, 1, 2, 15, 59)),
            ],
            datetime(2019, 1, 2, 16, 0): [
                #MarketEvent(datetime(2019, 1, 2, 15, 59, 1)),
                #MarketEvent(datetime(2019, 1, 2, 15, 59, 29)),
                #MarketEvent(datetime(2019, 1, 2, 15, 59, 30)),
                MarketEvent(datetime(2019, 1, 2, 15, 59, 31)),
                MarketEvent(datetime(2019, 1, 2, 15, 59, 59)),
                MarketEvent(datetime(2019, 1, 2, 16, 0)),
            ]}
        assert transmitter._partition_latent == {
            #datetime(2019, 1, 1, 15, 59): [
            #    MarketEvent(datetime(2019, 1, 1, 15, 58, 59)),
            #    MarketEvent(datetime(2019, 1, 1, 15, 59))],
            datetime(2019, 1, 1, 16, 0): [
                MarketEvent(datetime(2019, 1, 1, 15, 59, 1)),
                MarketEvent(datetime(2019, 1, 1, 15, 59, 29)),
                MarketEvent(datetime(2019, 1, 1, 15, 59, 30)),
                #MarketEvent(datetime(2019, 1, 1, 15, 59, 31)),
                #MarketEvent(datetime(2019, 1, 1, 15, 59, 59)),
                #MarketEvent(datetime(2019, 1, 1, 16, 0))
            ],
            datetime(2019, 1, 2, 15, 59): [
                MarketEvent(datetime(2019, 1, 1, 16, 0, 1)),
                MarketEvent(datetime(2019, 1, 1, 16, 0, 29)),
                MarketEvent(datetime(2019, 1, 1, 16, 0, 30)),
                #MarketEvent(datetime(2019, 1, 1, 16, 0, 31)),
                #MarketEvent(datetime(2019, 1, 2, 15, 58, 59)),
                #MarketEvent(datetime(2019, 1, 2, 15, 59)),
            ],
            datetime(2019, 1, 2, 16, 0): [
                MarketEvent(datetime(2019, 1, 2, 15, 59, 1)),
                MarketEvent(datetime(2019, 1, 2, 15, 59, 29)),
                MarketEvent(datetime(2019, 1, 2, 15, 59, 30)),
                #MarketEvent(datetime(2019, 1, 2, 15, 59, 31)),
                #MarketEvent(datetime(2019, 1, 2, 15, 59, 59)),
                #MarketEvent(datetime(2019, 1, 2, 16, 0)),
            ]}

    @pytest.mark.parametrize('latency, expected_raise', [
        (59, False), (60, True), (61, True),
    ])
    def test_raises_if_latency_greater_than_min_timestep(self, latency, expected_raise):
        transmitter = Transmitter(
            timesteps=[
                # Day 1.
                datetime(2019, 1, 1, 15, 59),
                datetime(2019, 1, 1, 16),
                # Day 2.
                datetime(2019, 1, 2, 15, 59),
                datetime(2019, 1, 2, 16),
            ])
        events = list()
        for t in transmitter.timesteps:
            for delay in [-1, 0, 1, 29, 30, 31]:
                events.append(MarketEvent(t+timedelta(seconds=delay)))
        transmitter.add_events(events)

        if expected_raise:
            with pytest.raises(ValueError):
                transmitter._create_partitions(latency)
        else:
            # Does not raise.
            transmitter._create_partitions(latency)
