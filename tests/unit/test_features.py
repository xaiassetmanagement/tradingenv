import pytest
from tradingenv.features import Feature
from tradingenv.events import Observer
import gymnasium.spaces
from datetime import datetime
from tradingenv.events import IEvent
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.exceptions import NotFittedError
import numpy as np
import pandas as pd


class EventSentiment(IEvent):
    def __init__(self, time: datetime, sentiment: float):
        self.time = time
        self.sentiment = sentiment


class TestAbstractFeature:
    def test_feature_is_an_observer(self):
        assert issubclass(Feature, Observer)

    def test_init_default_params(self):
        feature = Feature()
        assert feature.space is None
        assert feature.name == type(feature).__name__
        assert feature.transformer is None
        assert feature.save is True
        assert feature.history == dict()

    def test_init_name(self):
        feature = Feature(name='NewName')
        assert feature.name == 'NewName'

    def test_init_save(self):
        feature = Feature(save=False)
        assert feature.save is False

    def test_init_space(self):
        space = gymnasium.spaces.Box(-100, +100, (1, ))
        feature = Feature(space)
        assert feature.space is space

    def test_call_when_parse_is_not_implemented(self):
        feature = Feature()
        assert feature() is feature

    def test_call_when_parse_is_implemented(self):
        class F(Feature):
            def parse(self):
                return 1

        space = gymnasium.spaces.Discrete(2)
        feature = F(space)
        assert feature() == 1

    def test_call_raises_if_feature_does_not_belong_to_space(self):
        class F(Feature):
            def parse(self):
                return 5

        space = gymnasium.spaces.Discrete(2)
        feature = F(space)
        with pytest.raises(ValueError):
            feature(verify=True)

    def test_call_does_not_verify_obs_by_default(self):
        """Here obs does not belong to space but calling feature() does not
        raise by default. This is by design for two reasons:
        (1) big features might not belong to the space until they warm up
        (2) state is verify=True by default, so avoid double verification
        """
        class F(Feature):
            def parse(self):
                return 5

        space = gymnasium.spaces.Discrete(2)
        feature = F(space)
        # with pytest.raises(ValueError):
        feature()

    def test_call_does_not_raises_if_verify_set_to_false(self):
        class F(Feature):
            def parse(self):
                return 5

        space = gymnasium.spaces.Discrete(2)
        feature = F(space)
        feature(verify=False)

    def test_call_raises_if_verify_set_to_true(self):
        class F(Feature):
            def parse(self):
                return 5

        space = gymnasium.spaces.Discrete(2)
        feature = F(space)
        with pytest.raises(ValueError):
            feature(verify=True)

    def test_fit_transformer_manual_if_provided(self):
        class F(Feature):
            def _manual_fit_transformer(self):
                self.transformer.n_samples_seen_ = 0
                self.transformer.scale_ = 1.
                self.transformer.min_ = 2.
                self.transformer.data_min_ = 3.
                self.transformer.data_max_ = 4.
                self.transformer.data_range_ = 1.

        f = F(
            space=gymnasium.spaces.Box(-1., 1., (1, 2), float),
            transformer=MinMaxScaler(),
        )
        f.fit_transformer()
        assert f.transformer.n_samples_seen_ == 0
        assert f.transformer.scale_ == 1.
        assert f.transformer.min_ == 2.
        assert f.transformer.data_min_ == 3.
        assert f.transformer.data_max_ == 4.
        assert f.transformer.data_range_ == 1.

    @pytest.mark.parametrize(
        ['shape', 'expected_shape'],
        (
                [(1, ), (10, 1)],
                [(5,), (10, 5)],
                [(1, 3), (10, 1, 3)],
                [(3, 1), (10, 3, 1)],
                [(2, 3), (10, 2, 3)],
                [(2, 3, 4), (10, 2, 3, 4)],
        )
    )
    def test_parse_history(self, shape, expected_shape):
        feature = Feature(space=gymnasium.spaces.Box(-1., +1., shape, float))
        nr_obs = 10
        for t in pd.date_range('2022-01-01', freq='B', periods=nr_obs):
            feature._save_observation(feature.space.sample())
            feature.last_update = t
        actual = feature._parse_history()
        assert isinstance(actual, np.ndarray)
        assert actual.shape == expected_shape

    def test_fit_transformer_box_raises_if_no_history(self):
        f = Feature(
            space=gymnasium.spaces.Box(-1., 1., (1, 2), float),
            transformer=MinMaxScaler(),
        )
        with pytest.raises(ValueError):
            f.fit_transformer()

    def test_fit_transformer_without_history(self):
        f = Feature(
            space=gymnasium.spaces.MultiDiscrete([2, 3]),
            transformer=MinMaxScaler(),
        )
        with pytest.raises(NotImplementedError):
            f.fit_transformer()

    def test_fit_transformer_with_unsupported_space_raises(self):
        f = Feature(
            space=gymnasium.spaces.MultiDiscrete([2, 3]),
            transformer=MinMaxScaler(),
        )
        obs = np.array([1, 1])
        f._save_observation(obs)
        with pytest.raises(NotImplementedError):
            f.fit_transformer()

    def test_reset_last_update(self):
        class F(Feature):
            def process_EventSentiment(self, event: EventSentiment):
                return event.sentiment

        feature = F()
        assert feature.last_update is None
        EventSentiment(datetime(2019, 1, 1), 0.42).notify([feature])
        assert feature.last_update == datetime(2019, 1, 1)
        feature.reset()
        assert feature.last_update is None

    def test_reset_of_custom_future_with_custom_initialisation(self):
        class F(Feature):
            def __init__(self, a, b='b'):
                super().__init__()
                self.a = a
                self.b = b

        feature = F('a')
        # Does not raise
        feature.reset()
        assert feature.a == 'a'
        assert feature.b == 'b'

    def test_save_parsed_feature_to_history_then_reset(self):
        class F(Feature):
            def process_EventSentiment(self, event: EventSentiment):
                self._last_event = event

            def parse(self):
                return self._last_event.sentiment

        feature = F()
        assert feature.history == dict()
        EventSentiment(datetime(2019, 1, 1), 0.42).notify([feature])
        assert feature.history == {datetime(2019, 1, 1): 0.42}
        feature.reset()
        assert feature.history == dict()

    def test_dont_save_history_when_save_is_false(self):
        class F(Feature):
            def process_EventSentiment(self, event: EventSentiment):
                self._last_event = event

            def parse(self):
                return self._last_event.sentiment

        feature = F(save=False)
        assert feature.history == dict()
        EventSentiment(datetime(2019, 1, 1), 0.42).notify([feature])
        assert feature.history == dict()
        feature()
        assert feature.history == dict()
        feature.reset()
        assert feature.history == dict()

    def test_a_deepcopy_is_saved_in_data(self):
        class F(Feature):
            def __init__(self):
                super().__init__()
                self.data = dict()

            def process_EventSentiment(self, event: EventSentiment):
                self.data['sentiment'] = event.sentiment

            def parse(self):
                return self.data

        feature = F()
        EventSentiment(datetime(2019, 1, 1), 0.42).notify([feature])
        EventSentiment(datetime(2019, 1, 2), 0.43).notify([feature])
        assert feature.history == {
            datetime(2019, 1, 1): {'sentiment': 0.42},
            datetime(2019, 1, 2): {'sentiment': 0.43},
        }
