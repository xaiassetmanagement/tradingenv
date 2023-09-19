from tradingenv.events import Observer, EventNewObservation
from tradingenv.features import Feature
from gym.spaces import Space
from datetime import datetime
from typing import Dict, Any, Sequence, Callable, Union
from collections import deque
import tradingenv
import numpy as np
import functools
import copy
import gym


class IState(Observer):
    """An object storing the state of the environment. If you need
    OpenAI-gym compatibility, you are expected to implement IState.space and
    IState.parse in order to define the appropriate data structure to store
    the state."""
    # TODO: Add support for hierarchical features?
    space: Space = None

    def __new__(cls, *args, **kwargs):
        obj: 'Observer' = super().__new__(cls)
        obj._init_args = args
        obj._init_kwargs = kwargs
        obj._cache = dict()
        obj._nr_callbacks = -1
        obj._cache_enabled = False
        obj.history: Dict[datetime, Any] = None
        obj.exchange: 'tradingenv.exchange.Exchange' = None
        obj.features: Sequence[Feature] = None
        obj.save: bool = True
        obj.current_weights: Callable
        obj.reset()
        return obj

    def __init__(self, features: Sequence[Feature] = None, save: bool = True):
        """
        Parameters
        ----------
        features
            A sequence of classes implementing Feature. For small
            states, you probably don't need this. If your state class is
            becoming too big or if you need features transformation, a common
            requirement in machine learning applications, implementing
            Feature allows logic decoupling as IState will be a
            collection of features. Each feature is an observer of events and
            can define its own logic decoupled from the rest of the state or
            other features.
            Note: if you implement features, you don't have to implement any
            other method of this class. You are expected to implement only
            feature and their abstract methods and attributed. See tests for
            examples.
        """
        # TODO: should 'save' be passed to features to guarantee consistency?
        self.features = features
        self.save = save
        if self.features is not None:
            # TODO: test.
            self._verify_features = True
            self._transform_features = True
            names = [f.name for f in self.features]
            repeated_names = set([x for x in names if names.count(x) > 1])
            if repeated_names:
                raise ValueError(f"Duplicate features: {repeated_names}")
            mapping = {
                feature.name: feature.space for feature in self.features
            }
            if all(space is not None for space in mapping.values()):
                self.space = gym.spaces.Dict(mapping)

    def __call__(self, verify: bool = True):
        """Returns self is self.space is None. Otherwise, an object compatible
        with the space will be returned."""
        try:
            state = self.parse()
        except NotImplementedError:
            state = self
        else:
            if verify and self.space is not None:
                if state not in self.space:
                    raise ValueError(
                        'The following state does not belong to the '
                        'observation space.\n{}'.format(repr(state))
                    )
            if self.save:
                t = self._get_global_last_update()
                if self._cache_enabled:
                    if t not in self._cache:
                        self._cache[t] = state
                    self.history[t] = self._cache[t]
                else:
                    # Copy so that to avoid changing history if parse returns a
                    # mutable object (e.g. dict) that changes with new events.
                    self.history[t] = copy.deepcopy(state)
        return state

    def _get_global_last_update(self):
        """Returns time of the most recent update of the state or its features.
        """
        times = list()
        if self.last_update is not None:
            times.append(self.last_update)
        if self.exchange is not None:
            # TODO: test time is also taken from exchange.
            if self.exchange.last_update is not None:
                times.append(self.exchange.last_update)
        if self.features is not None:
            for feature in self.features:
                if feature.last_update is not None:
                    times.append(feature.last_update)
        # This will raise if mixed time zones.
        # TypeError: can't compare offset-naive and offset-aware datetimes
        now = max(times) if times else None
        return now

    def parse(self):
        """Returns an object compatible with IState.space. If IState.space is
        None, you don't have to worry about implementing this method as
        IState() will simply return self. If features are provided, no need
        to implement IState.parse."""
        if self.features is not None:
            # TODO: test feature() and not feature.parse() so that saved in history.
            return {
                feature.name: feature(
                    verify=self._verify_features,
                    transform=self._transform_features
                )
                for feature in self.features
            }
        raise NotImplementedError(
            "If State.space is an instance of gym.spaces.Space (i.e. not "
            "None), then you are supposed to implement IState.parse, which is "
            "expected to return the current state in a format such that "
            "'object in State.space' is True. Note that: specifying "
            "State.space is:\n"
            "- Optional by default.\n"
            "- Required if you want to validate if the state belongs to the "
            "observation space.\n"
            "- Required if you want compatibility with gym spaces "
            "(i.e. IState.space is not None).\n"
            "- Required if you want to use @cache.\n"
            "- Unnecessary if you have provided a list of features during init."
        )

    def reset(
            self,
            exchange: 'tradingenv.exchange.Exchange' = None,
            action_space: 'tradingenv.spaces.PortfolioSpace' = None,
            broker: 'tradingenv.broker.Broker' = None,
    ):
        """Reset state and features for next episode."""
        super().reset()
        self.history = dict()
        self.exchange = exchange
        self.action_space = action_space
        self.broker = broker
        if self.features is not None:
            for feature in self.features:
                feature.reset(exchange, action_space, broker)


class State(IState):
    """This State implementation allows to specify a window of past data. Useful
    when the observation is not Markovian."""

    def __init__(
            self,
            features: Union[int, Sequence[str]],
            window: int = 1,
            max_: float = 10.
    ):
        """
        Parameters
        ----------
        features
            Number of features or list of feature names. If int, feature names
            will be 0, 1, ..., n-1.
        window
            Window size of the observations. In other words, the observation
            returned by the environment in a given timestep is given by the
            concatenation of as many past observations. Using a window larger
            than 1 can be useful when past observation may have informational
            value that is observations are not Markovian.
        max_
            Maximum value of each feature. This is used to scale the state.
            The state is scaled to [-max_, +max_]. The default value is 100,
            arguably a large value. This is because we defer sanity checks to
            input features to data preprocessing.
        """
        try:
            list(features)
        except TypeError:
            n = features
            self.names = list(range(n))
        else:
            n = len(features)
            self.names = list(features)
        self.space = gym.spaces.Box(-max_, max_, (window, n), float)
        self.queue = deque(maxlen=window)
        self.last_event = None
        super().__init__()

    def process_EventNewObservation(self, event: EventNewObservation):
        if self.last_event is None:
            for _ in range(self.queue.maxlen):
                self.queue.append([event.to_list()])
        self.queue.append([event.to_list()])
        self.last_event = event

    def parse(self) -> np.array:
        """Returns a numpy array of shape (window, n_features)."""
        return np.concatenate(self.queue)

    def flatten(self) -> dict:
        """Flatten the history of past states. Can be useful to parse complex
        states in tabular form for further analysis or visualisations."""
        s_ = dict()
        for k, v in self.history.items():
            try:
                if v.ndim == 1:
                    # To make this work with 1d array
                    s_[k] = v.item()
                elif v.ndim == 2:
                    # To make this work with 2d array
                    s_[k] = v.squeeze()
            except ValueError:
                # Was not a np.array, either 1d or 2d.
                # ValueError: can only convert an array of size 1 to a Python scalar
                for i, j in enumerate(v[0]):
                    s_[f'{k}{i}'] = j
                pass
        return s_


def cache(callback=None, check_every: int = np.inf, check_on_start: bool = True):
    """Decorator which allows to use cached results from previous
    episodes/backtests of Feature.process_<Event> methods to save computations.

    Parameters
    ----------
    callback
        A convenience arg all allows cache being either callable or not.
    check_every : bool
        If True, every 'check_every' the cached result will not be used and
        compared with the cached result. If there is basis_naive_tm1 difference, an error
        will be raised, meaning that your features should probably not used
        cached results. See Notes.
    check_on_start : bool
        If True, cached and non-cached results will be compared when the
        feature will receive its first observed event. If they are different,
        an error will be raised, meaning that your features should probably
        not used cached results. See Notes.

    Notes
    -----
    By cached result we mean the state of the feature calculated at the same
    timestep from basis_naive_tm1 previous episode. By the nature of this caching mechanism,
    there is no way to use cached results when your feature is not deterministic
    and in such case an error will be raised presumably at the first check,
    assuming that some form of checking is performed as indicated by the
    arguments of this function.
    """
    # Implementation inspired by 2018 update of jsbueno:
    #   https://stackoverflow.com/questions/10294014
    if callback is None:
        # This trick allows to use either calling this decorator or not.
        # If you are here, your decorator has been called with kwargs.
        return functools.partial(
            cache,
            check_every=check_every,
            check_on_start=check_on_start,
        )

    def are_equal(a, b):
        if isinstance(a, dict):
            return all(are_equal(a[k], b[k]) for k in a)
        return np.allclose(a, b, equal_nan=True)

    @functools.wraps(callback)
    def wrapper(self: IState, event: 'tradingenv.IEvent'):
        self._cache_enabled = True
        verify = any([
            self._nr_callbacks > 0 and self._nr_callbacks % check_every == 0,
            check_on_start and self._nr_callbacks == 0,
        ])
        if event.time in self._cache:
            actual = self._cache[event.time]
            if verify:
                callback(self, event)
                expected = self.parse()
                if not are_equal(expected, actual):
                    raise ValueError(
                        'Chached data is different from what the state should '
                        'be. You should not use @cache if the state is '
                        'path-dependant:\n'
                        'Actual:\n'
                        '{actual}\n'
                        'Expected:\n'
                        '{expected}'
                        ''.format(actual=actual, expected=expected)
                    )
        else:
            callback(self, event)
            actual = self.parse()
        return actual
    return wrapper
