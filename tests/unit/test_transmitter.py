from tradingenv.transmitter import (
    AsynchronousTransmitter,
    AbstractTransmitter,
    Transmitter,
    Folds,
)
from tradingenv.events import IEvent
from datetime import datetime, timezone
import time as time_
import numpy as np
import pandas as pd
import pytest
from collections import Counter


class MarketEvent(IEvent):
    def __init__(self, time):
        self.time = time


class TestEmptyTransmitter:
    def test_parent_class_is_Transmitter(self):
        assert issubclass(AsynchronousTransmitter, AbstractTransmitter)

    def test_now(self):
        transmitter = AsynchronousTransmitter()
        time_.sleep(0.002)
        now = datetime.utcnow()#.replace(tzinfo=timezone.utc)
        assert (transmitter._now() - now).total_seconds() < 0.001

    def test_create_partitions(self):
        transmitter = AsynchronousTransmitter()
        transmitter._create_partitions()  # does not raise error


class TestTransmitter:
    def test_parent_class_is_Transmitter(self):
        assert issubclass(Transmitter, AbstractTransmitter)

    def test_reset_create_partitions_calling_reset_for_the_first_time(self):
        transmitter = Transmitter(timesteps=[datetime.now()])
        assert transmitter._partition_nonlatent is None
        transmitter._reset()
        assert transmitter._partition_nonlatent is not None

    def test_sampling_span(self):
        # There are 10 dates
        # ##########
        # We cannot sample the last 4 as episode_length is 5
        # ######
        # Gamma is 1 / (1 - sampling_span) = 0.5
        # Therefore the probabilities of the first six dates should be:
        #   p = 0.5 ** np.arange(6)
        #   p /= p.sum()
        # That is [0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.50]
        start_dates = list(pd.date_range('2023', periods=10, freq='D'))
        events = [MarketEvent(t) for t in start_dates]
        transmitter = Transmitter(timesteps=start_dates)
        transmitter.add_events(events)

        start_dates = list()
        np.random.seed(0)
        for i in range(10000):
            transmitter._reset(episode_length=5, sampling_span=2)
            start_dates.append(transmitter._start_date)
        c = Counter(start_dates)
        c = {k: c[k] / sum(c.values()) for k in sorted(c)}

        actual_dates = list(c.keys())
        expected_dates = list(pd.date_range('2023', periods=6, freq='D'))
        assert actual_dates == expected_dates

        actual_freq = list(c.values())
        expected_freq = [0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.50]
        np.testing.assert_allclose(actual_freq, expected_freq, rtol=0.2)

    def test_now_origin(self):
        # Origin is first date less avg diff
        transmitter = Transmitter(timesteps=[datetime.now()])
        transmitter._create_partitions()
        transmitter._reset()
        assert transmitter._now() is np.nan

    def test_split_raises_if_periods_have_more_then_two_timestamps(self):
        with pytest.raises(ValueError):
            Transmitter(
                timesteps=[datetime.now()],
                folds={"training-set": [datetime(2019, 1, 5), datetime(2019, 1, 1)]},
            )

    def test_split_raises_if_periods_have_less_then_two_timestamps(self):
        with pytest.raises(ValueError):
            Transmitter(
                timesteps=[datetime.now()],
                folds={"training-set": [datetime(2019, 1, 1)]},
            )

    def test_split_raises_if_start_is_older_than_end(self):
        with pytest.raises(ValueError):
            Transmitter(
                timesteps=[datetime.now()],
                folds={"training-set": [datetime(2019, 1, 5), datetime(2019, 1, 1)]},
            )

    @pytest.mark.skip
    def test_split_raises_if_overlapping_periods_edge(self):
        with pytest.raises(ValueError):
            Transmitter(
                timesteps=[datetime.now()],
                folds={
                    "training-set": [datetime(2019, 1, 1), datetime(2019, 1, 5)],
                    "test-set": [datetime(2019, 1, 5), datetime(2019, 1, 10)],
                },
            )

    @pytest.mark.skip
    def test_split_raises_if_overlapping_periods(self):
        with pytest.raises(ValueError):
            Transmitter(
                timesteps=[datetime.now()],
                folds={
                    "training-set": [datetime(2019, 1, 1), datetime(2019, 1, 5)],
                    "test-set": [datetime(2019, 1, 4), datetime(2019, 1, 10)],
                },
            )

    @pytest.mark.skip
    def test_split_does_not_raises_if_overlapping_datetime_min(self):
        with pytest.raises(ValueError):
            Transmitter(
                timesteps=[datetime.now()],
                folds={
                    "training-set": [datetime.min, datetime(2019, 1, 5)],
                    "test-set": [datetime(2019, 1, 4), datetime(2019, 1, 10)],
                },
            )

    def test_walk_forward_idx_have_same_length(self):
        for train_size in range(1, 5):
            for test_size in range(1, 5):
                transmitter = Transmitter(
                    timesteps=pd.date_range("2019-01-01", periods=20, freq="B")
                )
                folds = transmitter.walk_forward(
                    train_size=train_size, test_size=test_size
                )
                assert len(folds.train_start) == len(folds.train_end)
                assert len(folds.train_start) == len(folds.test_start)
                assert len(folds.train_start) == len(folds.test_end)

    @pytest.mark.parametrize(
        argnames=[
            "nr_obs",
            "expected_train_start",
            "expected_train_end",
            "expected_test_start",
            "expected_test_end",
        ],
        ids=[str(i) for i in range(1, 20)],  # nr_obs
        argvalues=[
            (1, [], [], [], []),
            (2, [], [], [], []),
            (3, [], [], [], []),
            (4, [], [], [], []),
            (5, [], [], [], []),
            (6, [], [], [], []),
            (7, [], [], [], []),
            (8, [0], [4], [5], [7]),
            (9, [0], [4], [5], [7]),
            (10, [0], [4], [5], [7]),
            (11, [0, 3], [4, 7], [5, 8], [7, 10]),
            (12, [0, 3], [4, 7], [5, 8], [7, 10]),
            (13, [0, 3], [4, 7], [5, 8], [7, 10]),
            (14, [0, 3, 6], [4, 7, 10], [5, 8, 11], [7, 10, 13]),
            (15, [0, 3, 6], [4, 7, 10], [5, 8, 11], [7, 10, 13]),
            (16, [0, 3, 6], [4, 7, 10], [5, 8, 11], [7, 10, 13]),
            (17, [0, 3, 6, 9], [4, 7, 10, 13], [5, 8, 11, 14], [7, 10, 13, 16]),
            (18, [0, 3, 6, 9], [4, 7, 10, 13], [5, 8, 11, 14], [7, 10, 13, 16]),
            (19, [0, 3, 6, 9], [4, 7, 10, 13], [5, 8, 11, 14], [7, 10, 13, 16]),
        ],
    )
    def test_split(
        self,
        nr_obs,
        expected_train_start,
        expected_train_end,
        expected_test_start,
        expected_test_end,
    ):
        transmitter = Transmitter(
            timesteps=pd.date_range("2019-01-01", periods=nr_obs, freq="B")
        )
        folds = transmitter.walk_forward(train_size=5, test_size=3)
        np.testing.assert_equal(folds.train_start, expected_train_start)
        np.testing.assert_equal(folds.train_end, expected_train_end)
        np.testing.assert_equal(folds.test_start, expected_test_start)
        np.testing.assert_equal(folds.test_end, expected_test_end)


class TestFolds:
    def test_as_time(self):
        folds = Folds(
            timesteps=[
                datetime(2019, 1, 1),
                datetime(2019, 1, 2),
                datetime(2019, 1, 3),
                datetime(2019, 1, 4),
                datetime(2019, 1, 7),
                datetime(2019, 1, 8),
                datetime(2019, 1, 9),
                datetime(2019, 1, 10),
                datetime(2019, 1, 11),
                datetime(2019, 1, 14),
                datetime(2019, 1, 15),
            ],
            train_start=[0, 3],
            train_end=[5, 8],
            test_start=[6, 9],
            test_end=[7, 10],
        )
        actual = folds.as_time()
        np.testing.assert_equal(
            actual=actual.train_start,
            desired=[datetime(2019, 1, 1), datetime(2019, 1, 4)],
        )
        np.testing.assert_equal(
            actual=actual.train_end,
            desired=[datetime(2019, 1, 8), datetime(2019, 1, 11)],
        )
        np.testing.assert_equal(
            actual=actual.test_start,
            desired=[datetime(2019, 1, 9), datetime(2019, 1, 14)],
        )
        np.testing.assert_equal(
            actual=actual.test_end,
            desired=[datetime(2019, 1, 10), datetime(2019, 1, 15)],
        )
