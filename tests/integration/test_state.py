from tradingenv.state import IState
from tradingenv.features import Feature
import gymnasium.spaces
import numpy as np
import unittest.mock


class TestState:
    def test_state_without_features(self):
        state = IState()
        assert state.features is None

    def test_state_with_feature_with_space_undefined(self):
        class F(Feature):
            pass

        features = [Feature()]
        state = IState(features)
        assert state.space is None

    def test_state_with_feature_with_space_defined(self):
        class F1(Feature):
            pass

        class F2(Feature):
            pass

        f1 = F1(gymnasium.spaces.Discrete(2))
        f2 = F2(gymnasium.spaces.Box(-1., +1., (1, ), np.float64))
        state = IState([f1, f2])
        actual = state.space
        expected = gymnasium.spaces.Dict({f1.name: f1.space, f2.name: f2.space})
        assert actual == expected

    def test_parse_when_features_are_provided(self):
        class F1(Feature):
            def parse(self, state=None):
                return 1

        class F2(Feature):
            def parse(self, state=None):
                return np.array([0.42])

        f1 = F1(gymnasium.spaces.Discrete(2))
        f2 = F2(gymnasium.spaces.Box(-1., +1., (1, ), float))
        state = IState([f1, f2])
        actual = state.parse()
        expected = {
            f1.name: 1,
            f2.name: 0.42,
        }
        assert actual == expected

    def test_reset_resets_features(self):
        class F1(Feature):
            pass

        f1 = F1()
        state = IState([f1])
        with unittest.mock.patch.object(F1, 'reset', return_value=None) as reset:
            state.reset()
            reset.assert_called_once()

