import pandas as pd
import pytest
from tradingenv.features import Feature
from tradingenv.events import Observer
import gym.spaces
from datetime import datetime
from tradingenv.events import IEvent
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tradingenv.events import EventNBBO
from tradingenv.contracts import ETF
import numpy as np
# TODO: find an interface of features possibly based on sklearn
# TODO: feature should be optionally scaled using sklearn transformers, if
#  provided - no need to reinvent the wheel and use familiar API. There are
#  use cases where the user does not need to transform (e.g. backtesting) so
#  this should be a feature and not a necessary condition.
# TODO: AbstractFeature should implement reset no?


class TestFeaturesTransformers:
    def test_raises_if_transformer_is_passed_without_space(self):
        with pytest.raises(ValueError):
            Feature(transformer=StandardScaler())

    def test_reset_preserves_transformer_attributes(self):
        # https://stackoverflow.com/questions/55731933/initialise-standardscaler-from-scaling-parameters
        f = Feature(
            space=gym.spaces.Box(-100, +100, (1, 2)),
            transformer=StandardScaler(),
        )
        f.transformer.mean_ = np.array(([[1., 2.]]))
        f.reset()

        actual = f.transformer.mean_
        expected = np.array(([[1., 2.]]))
        np.testing.assert_equal(actual, expected)

    def test_transform_persist_across_reset(self):
        class FeaturePortfolioWeight(Feature):
            """Feature instancing the transformer in init, therefore not
            saved to _init_args/kwargs, make sure that transformer fit persists
            across resets."""

            def __init__(self, contracts):
                self.contracts = contracts
                super().__init__(
                    space=gym.spaces.Box(0., 1., (1, len(self.contracts))),
                    transformer=MinMaxScaler((-1, 1)),
                )

            def parse(self):
                return np.array([[0.5]])

            def _manual_fit_transformer(self):
                """Whatever the max or min allocation is per asset, either long or
                short, rescale the current portfolio weights in the [-1, +1] range."""
                x = np.concatenate([self.space.high, self.space.low])
                self.transformer.fit(x)

        feature = FeaturePortfolioWeight([ETF('SPY')])
        feature.reset()
        actual = feature()
        expected = np.array([[0.5]])
        np.testing.assert_almost_equal(actual, expected)
        feature.fit_transformer()

        actual = feature()
        expected = np.array([[0.]])
        np.testing.assert_almost_equal(actual, expected)

        feature.reset()

        actual = feature()
        expected = np.array([[0.]])
        np.testing.assert_almost_equal(actual, expected)

    @pytest.mark.skip(reason='TODO: Refactor')
    def test_can_override_parameters_of_StandardScaler(self):
        class MyFeature(Feature):
            def process_EventNBBO(self, event):
                self.event = event

            def parse(self):
                data = np.array([[self.event.bid_price, self.event.ask_price]])
                data *= np.random.normal()
                return data

        feature = MyFeature(
            space=gym.spaces.Box(-100, +100, (1, 2)),
            transformer=StandardScaler(),
        )
        feature.reset()
        for t in pd.date_range('2022-01-01', periods=10, freq='B'):
            event = EventNBBO(t, ETF('SPY'), 1., 1., 1., 1.)
            event.notify([feature])

        X = np.concatenate(list(feature.history.values()))
        scaler = StandardScaler()
        scaler.fit(X)
        mean, scale, var = scaler.mean_, scaler.scale_, scaler.var_
        expected = scaler.transform(X)

        scaler = StandardScaler()
        scaler.mean_, scaler.scale_, scaler.var_ = mean, scale, var
        actual = scaler.transform(X)
        np.testing.assert_equal(actual, expected)

    def test_(self):
        pass
        # scaler.min_
        # scaler.scale_
        # scaler.data_min_
        # scaler.data_max_
        # scaler.data_range_
