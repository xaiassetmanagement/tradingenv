import numpy as np
from tradingenv.state import IState, cache
from tradingenv.events import Observer, IEvent
from datetime import datetime
import pytest
import gym


class EventSentiment(IEvent):
    def __init__(self, time: datetime, sentiment: float):
        self.time = time
        self.sentiment = sentiment


class TestStateWithoutSpace:
    def make_state(self):
        class ConcreteState(IState):
            def __init__(self):
                self.sentiment = 0.18

            def process_EventSentiment(self, event: EventSentiment):
                self.sentiment = event.sentiment

        return ConcreteState()

    def test_observed_events(self):
        state = self.make_state()
        assert state._observed_events == {'EventSentiment': 'process_EventSentiment'}

    def test_is_observer(self):
        assert issubclass(IState, Observer)

    def test_space_is_none_by_default(self):
        state = self.make_state()
        assert state.space is None

    def test_callable(self):
        state = self.make_state()
        assert state() is state

    def test_does_not_raise_if_no_observed_events(self):
        # Does not raise. Empty state with no features.
        IState()

    def test_callback(self):
        state = self.make_state()
        state.reset()
        assert state.sentiment == 0.18
        EventSentiment(datetime(2019, 1, 1), 0.42).notify([state])
        assert state.sentiment == 0.42

    def test_not_raise_if_space_is_not_none_and_parse_is_not_implemented(self):
        class State(IState):
            def __init__(self):
                self.space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float64)
                self.sentiment = 0.18

            def process_EventSentiment(self, event: EventSentiment):
                self.sentiment = event.sentiment

            def parse(self):
                return np.array([self.sentiment])

        state = State()
        np.testing.assert_equal(
            actual=state(),
            desired=np.array([0.18])
        )

    def test_raise_if_state_does_not_belong_to_space(self):
        class State(IState):
            def __init__(self):
                self.space = gym.spaces.Box(low=-1, high=1, shape=(1,))
                self.sentiment = 2.  # greater than 1. will throw an exception

            def process_EventSentiment(self, event: EventSentiment):
                self.sentiment = event.sentiment

            def parse(self):
                return np.array([self.sentiment])

        state = State()
        with pytest.raises(ValueError):
            state()

    def test_not_raise_if_state_does_not_belong_to_space_and_False_verify(self):
        class State(IState):
            def __init__(self):
                self.space = gym.spaces.Box(low=-1, high=1, shape=(1,))
                self.sentiment = 2.  # greater than 1. will throw an exception

            def process_EventSentiment(self, event: EventSentiment):
                self.sentiment = event.sentiment

            def parse(self):
                return np.array([self.sentiment])

        state = State()
        state(verify=False)  # does not raise because verify=False


class TestStateHistory:
    def test_history_saved_by_default(self):
        class State(IState):
            def __init__(self):
                self.space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=float)
                self.sentiment = 0.

            def process_EventSentiment(self, event: EventSentiment):
                self.sentiment = event.sentiment

            def parse(self):
                return np.array([self.sentiment])

        state = State()
        assert state.history == dict()
        EventSentiment(datetime(2019, 1, 1), 0.42).notify([state])
        assert state.history == {datetime(2019, 1, 1): 0.42}

    @pytest.mark.skip(reason='Nothing bad happens if you dont implement parse')
    def test_parse_must_be_implemented_if_space_is_provided(self):
        """OpenAI-gym compatibility will fail if you don't implement parse,
        but an error will be raised probably when gym will check if state
        belongs to space. We could raise a clearer error message maybe?"""
        class State(IState):
            def __init__(self):
                self.space = gym.spaces.Box(low=-1, high=1, shape=(1,))
                self.sentiment = 0.

            def process_EventSentiment(self, event: EventSentiment):
                self.sentiment = event.sentiment

        state = State()
        with pytest.raises(NotImplementedError):
            EventSentiment(datetime(2019, 1, 1), 0.42).notify([state])

    @pytest.mark.skip(reason='Nothing bad happens if you dont implement parse')
    def test_raise_if_space_is_not_none_and_parse_is_not_implemented(self):
        """OpenAI-gym compatibility will fail if you don't implement parse,
        but an error will be raised probably when gym will check if state
        belongs to space. We could raise a clearer error message maybe?"""
        class State(IState):
            def __init__(self):
                self.space = gym.spaces.Box(low=-1, high=1, shape=(1,))
                self.sentiment = 0.18

            def process_EventSentiment(self, event: EventSentiment):
                self.sentiment = event.sentiment

        state = State()
        with pytest.raises(NotImplementedError):
            state()

    def test_no_history_is_saved_if_parse_is_not_implemented(self):
        class State(IState):
            def __init__(self):
                self.sentiment = 0.

            def process_EventSentiment(self, event: EventSentiment):
                self.sentiment = event.sentiment

        state = State()
        assert state.history == dict()
        EventSentiment(datetime(2019, 1, 1), 0.42).notify([state])
        assert state.history == dict()

    def test_save_with_space(self):
        class State(IState):
            def __init__(self):
                self.space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=float)
                self.sentiment = 0.

            def process_EventSentiment(self, event: EventSentiment):
                self.sentiment = event.sentiment

            def parse(self):
                return np.array([self.sentiment])

        state = State()
        assert state.history == dict()
        EventSentiment(datetime(2019, 1, 1), 0.42).notify([state])
        assert state.history == {
            datetime(2019, 1, 1): np.array([0.42]),
        }

    def test_save_without_space(self):
        class State(IState):
            def __init__(self):
                self.sentiment = 0.

            def process_EventSentiment(self, event: EventSentiment):
                self.sentiment = event.sentiment

            def parse(self):
                return np.array([self.sentiment])

        state = State()
        assert state.history == dict()
        EventSentiment(datetime(2019, 1, 1), 0.42).notify([state])
        assert state.history == {
            datetime(2019, 1, 1): np.array([0.42]),
        }

    def test_parse_observe_raise_if_process_has_different_than_arg_event(self):
        class NewsSentiment(IState):
            def process_EventNews(self, event: EventSentiment, excess="a"):
                return 42

        with pytest.raises(TypeError):
            # NewsSentiment.process_EventNews must accept 'event' only.
            NewsSentiment()


class TestStateReset:
    def test_initialization_without_args_or_kwargs(self):
        class State(IState):
            def __init__(self):
                self.sentiment = 0.

            def process_EventSentiment(self, event: EventSentiment):
                self.sentiment = event.sentiment

        state = State()
        state.reset()
        assert state.last_update is None
        assert state.sentiment == 0.
        EventSentiment(datetime(2019, 1, 1), 0.42).notify([state])
        assert state.last_update == datetime(2019, 1, 1)
        assert state.sentiment == 0.42
        state.reset()
        assert state.last_update is None
        assert state.sentiment == 0.

    def test_initialization_with_args_and_kwargs(self):
        class State(IState):
            def __init__(self, a, b, c='c'):
                self.a = a
                self.b = b
                self.c = c

            def process_EventSentiment(self, event: EventSentiment):
                self.a = event.sentiment
                self.b = event.sentiment
                self.c = event.sentiment

        state = State('a', b='b')
        state.reset()
        assert state.last_update is None
        assert state.a == 'a'
        assert state.b == 'b'
        assert state.c == 'c'
        EventSentiment(datetime(2019, 1, 1), 0.42).notify([state])
        assert state.last_update == datetime(2019, 1, 1)
        assert state.a == 0.42
        assert state.b == 0.42
        assert state.c == 0.42
        state.reset()
        assert state.last_update is None
        assert state.a == 'a'
        assert state.b == 'b'
        assert state.c == 'c'

    def test_initialization_with_class_instance(self):
        class Sentiment:
            def __init__(self):
                self.value = 0.

        class State(IState):
            def __init__(self):
                self.sentiment = Sentiment()

            def process_EventSentiment(self, event: EventSentiment):
                self.sentiment.value = event.sentiment

        state = State()
        state.reset()
        assert state.last_update is None
        assert state.sentiment.value == 0.
        EventSentiment(datetime(2019, 1, 1), 0.42).notify([state])
        assert state.last_update == datetime(2019, 1, 1)
        assert state.sentiment.value == 0.42
        state.reset()
        assert state.last_update is None
        assert state.sentiment.value == 0.

    def test_initialization_with_default_class_instance(self):
        class Sentiment:
            def __init__(self):
                self.value = 0.

        class State(IState):
            def __init__(self, sentiment):
                self.sentiment = sentiment

            def process_EventSentiment(self, event: EventSentiment):
                self.sentiment.value = event.sentiment

        state = State(Sentiment())
        state.reset()
        assert state.last_update is None
        assert state.sentiment.value == 0.
        EventSentiment(datetime(2019, 1, 1), 0.42).notify([state])
        assert state.last_update == datetime(2019, 1, 1)
        assert state.sentiment.value == 0.42
        state.reset()
        assert state.last_update is None

        # You are evil because you are using an object with mutable State and
        # reset will fail.
        assert state.sentiment.value != 0.

    def test_history_lifecycle_independent_from_resets(self):
        class State(IState):
            def process_EventSentiment(self, event: EventSentiment):
                self.sentiment = event.sentiment

            def parse(self):
                return self.sentiment

        for _ in range(3):
            state = State()
            state.reset()
            assert state.history == dict()
            EventSentiment(datetime(2019, 1, 2), 42).notify([state])
            assert state.history == {datetime(2019, 1, 2): 42}

    def test_dont_save_if_false(self):
        class State(IState):
            def process_EventSentiment(self, event: EventSentiment):
                self.sentiment = event.sentiment

            def parse(self):
                return self.sentiment

        for _ in range(3):
            state = State(save=False)
            state.reset()
            assert state.history == dict()
            EventSentiment(datetime(2019, 1, 2), 42).notify([state])
            assert state.history == dict()
            # assert state.history == {datetime(2019, 1, 2): 42}


class TestCache:
    def test_no_cache(self):
        class State(IState):
            def __init__(self):
                self.sentiment = 0.

            def process_EventSentiment(self, event: EventSentiment):
                self.sentiment = event.sentiment

            def parse(self):
                return self.sentiment

        # First episode. Regardless the cache, everything is calculated from
        # scratch.
        state = State()
        EventSentiment(datetime(2019, 1, 1), 0).notify([state])
        EventSentiment(datetime(2019, 1, 2), 1).notify([state])
        EventSentiment(datetime(2019, 1, 3), 2).notify([state])
        assert state.history == {
            datetime(2019, 1, 1): 0,
            datetime(2019, 1, 2): 1,
            datetime(2019, 1, 3): 2,
        }

        # Second episode. No cache, so all updates are done from scratch again.
        state.reset()
        EventSentiment(datetime(2019, 1, 1), 1).notify([state])
        EventSentiment(datetime(2019, 1, 2), 2).notify([state])
        EventSentiment(datetime(2019, 1, 3), 3).notify([state])
        assert state.history == {
            datetime(2019, 1, 1): 1,
            datetime(2019, 1, 2): 2,
            datetime(2019, 1, 3): 3,
        }

    def test_cache_default_arguments(self):
        class State(IState):
            def __init__(self):
                self.sentiment = 0.

            @cache
            def process_EventSentiment(self, event: EventSentiment):
                self.sentiment = event.sentiment

            def parse(self):
                return self.sentiment

        # First episode. Regardless the cache, everything is calculated from
        # scratch.
        state = State()
        EventSentiment(datetime(2019, 1, 1), 0).notify([state])
        EventSentiment(datetime(2019, 1, 2), 1).notify([state])
        EventSentiment(datetime(2019, 1, 3), 2).notify([state])
        assert state.history == {
            datetime(2019, 1, 1): 0,
            datetime(2019, 1, 2): 1,
            datetime(2019, 1, 3): 2,
        }

        # Second episode. Cache, so all updates are done are retrieved from
        # the first episode, despite being wrong in this case.
        # First callback is verified by default. So value != 0 would raise.
        state.reset()
        EventSentiment(datetime(2019, 1, 1), 0).notify([state])
        EventSentiment(datetime(2019, 1, 2), 2).notify([state])
        EventSentiment(datetime(2019, 1, 3), 3).notify([state])
        assert state.history == {
            datetime(2019, 1, 1): 0,  # this would be 1 without cache
            datetime(2019, 1, 2): 1,  # this would be 2 without cache
            datetime(2019, 1, 3): 2,  # this would be 3 without cache
        }

    def test_cache_arg_check_every(self):
        class State(IState):
            def __init__(self):
                self.sentiment = 0.

            @cache(check_every=2)
            def process_EventSentiment(self, event: EventSentiment):
                self.sentiment = event.sentiment

            def parse(self):
                return self.sentiment

        # First episode. Regardless the cache, everything is calculated from
        # scratch.
        state = State()
        EventSentiment(datetime(2019, 1, 1), 0).notify([state])
        EventSentiment(datetime(2019, 1, 2), 1).notify([state])
        EventSentiment(datetime(2019, 1, 3), 2).notify([state])
        assert state.history == {
            datetime(2019, 1, 1): 0,
            datetime(2019, 1, 2): 1,
            datetime(2019, 1, 3): 2,
        }

        # Second episode. Cache with check_every=2, so first and third events
        # are checked.
        state.reset()
        # First event leads to the same state of checked first event, so no
        # error is raised.
        EventSentiment(datetime(2019, 1, 1), 0).notify([state])
        # The second event is different form the cached state (state=1), but
        # check_every=2 skips the check on this event.
        EventSentiment(datetime(2019, 1, 2), 2).notify([state])
        # The third event is different form the cached state (state=2).
        # check_every=2 so now there is basis_naive_tm1 check which raises.
        with pytest.raises(ValueError):
            EventSentiment(datetime(2019, 1, 3), 3).notify([state])

    def test_cache_arg_check_on_start(self):
        class State(IState):
            def __init__(self):
                self.sentiment = 0.

            @cache(check_on_start=True)
            def process_EventSentiment(self, event: EventSentiment):
                self.sentiment = event.sentiment

            def parse(self):
                return self.sentiment

        # First episode. Regardless the cache, everything is calculated from
        # scratch.
        state = State()
        EventSentiment(datetime(2019, 1, 1), 0).notify([state])
        EventSentiment(datetime(2019, 1, 2), 1).notify([state])
        EventSentiment(datetime(2019, 1, 3), 2).notify([state])
        assert state.history == {
            datetime(2019, 1, 1): 0,
            datetime(2019, 1, 2): 1,
            datetime(2019, 1, 3): 2,
        }

        # Second episode. Cache with check_every=2, so first and third events
        # are checked.
        state.reset()
        # First event leads to different state of checked first event, so
        # error is raised because check_on_start is True.
        with pytest.raises(ValueError):
            EventSentiment(datetime(2019, 1, 1), 1).notify([state])
