import gym
import gym.utils.seeding
from tradingenv.env import TradingEnv
from tradingenv.events import IEvent, EventReset, EventNBBO, EventDone
from tradingenv.transmitter import Transmitter, AsynchronousTransmitter
from tradingenv.spaces import DiscretePortfolio, BoxPortfolio
from tradingenv.state import IState
from tradingenv.features import Feature
from tradingenv.broker.broker import Broker, EndOfEpisodeError
from tradingenv.rewards import RewardPnL, AbstractReward
from tradingenv.contracts import Cash, ETF, Rate, ES, Index
from datetime import datetime
import numpy as np
import unittest.mock
import pytest
import pandas as pd
import time
import os
import inspect


class TestTradingEnvETF:
    def get_discrete_action_space(self):
        action_space = DiscretePortfolio(
            contracts=[Cash(), ETF("Equities"), ETF("Bonds")],
            allocations=[
                [1, 0, 0],
                [0, 0, 1],
                [0, 0.25, 0.75],
                [0, 0.5, 0.5],
                [0, 0.75, 0.25],
                [0, 1, 0],
            ],
        )
        return action_space

    def get_discrete_env(self, **kwargs):
        env = TradingEnv(action_space=self.get_discrete_action_space(), **kwargs)
        return env

    def test_is_subclass_of_gym_env(self):
        assert issubclass(TradingEnv, gym.Env)

    # def test_render_modes_are_specified(self):
    #     assert len(TradingEnv.metadata["render.modes"]) > 0

    def test_initialization_sets_inputs(self):
        class MockFeature(IState):
            space = gym.spaces.Box(-np.inf, +np.inf, (1,), np.float32)

            def process_Event(self, event: IEvent):
                return 0.0

        action_space = self.get_discrete_action_space()
        feature = MockFeature()
        reward = RewardPnL()
        env = TradingEnv(
            action_space=action_space,
            state=feature,
            reward=reward,
            initial_cash=42,
        )
        assert env.action_space == action_space
        assert feature is env.state
        assert id(env._reward) == id(reward)
        assert isinstance(env._transmitter, AsynchronousTransmitter)
        assert env._initial_cash == 42
        assert env.action_space._margin == 0.0

    def test_initialization_with_prices_overrides_transmitter(self):
        prices = pd.DataFrame(
            data={Index("S&P 500"): [1, 1.1, 1.3], Index("T-Notes"): [1, 1.02, 1.3]},
            index=pd.date_range("2019-01-01", periods=3, freq="D"),
        )
        env = TradingEnv(action_space=self.get_discrete_action_space(), prices=prices)
        expected = Transmitter(prices.index)
        expected.add_prices(prices)
        for contract in env.action_space.contracts:
            expected.add_events(contract.make_events())
        assert isinstance(env._transmitter, Transmitter)
        assert env._transmitter.timesteps == expected.timesteps
        assert env._transmitter.events == expected.events

    def test_initialization_action_space_can_be_list(self):
        env = TradingEnv(action_space=[Cash(), ETF("Equities"), ETF("Bonds")])
        assert isinstance(env.action_space, BoxPortfolio)
        assert env.action_space.contracts == [Cash(), ETF("Equities"), ETF("Bonds")]

    def test_initialization_reward_can_be_a_string(self):
        env = self.get_discrete_env(reward="RewardPnL")
        assert isinstance(env._reward, RewardPnL)

    def test_initialization_sets_reset_inputs(self):
        action_space = self.get_discrete_action_space()
        env = TradingEnv(action_space)
        assert env._done is None
        assert env.exchange is None
        assert env.broker is None
        assert env._last_event is None
        assert env._observers is None

    def test_initialisation_with_list_of_features(self):
        action_space = self.get_discrete_action_space()
        f1 = Feature(name='F1')
        f2 = Feature(name='F2')
        env = TradingEnv(action_space, state=[f1, f2])
        assert env.state.features == [f1, f2]

    @unittest.mock.patch.object(AsynchronousTransmitter, "add_events")
    def test_initialization_reset_events_from_contracts(self, mock_add_events):
        action_space = DiscretePortfolio(
            contracts=[Cash(), ES(2019, 6), ES(2019, 9)],
            allocations=[
                [1, 0, 0],
                [0, 0, 1],
                [0, 0.25, 0.75],
                [0, 0.5, 0.5],
                [0, 0.75, 0.25],
                [0, 1, 0],
            ],
        )
        env = TradingEnv(action_space)
        for contract in action_space.contracts:
            events = contract.make_events()
            mock_add_events.assert_any_call(events)

    def test_add_event(self):
        env = self.get_discrete_env()
        event = IEvent()
        env.reset()
        env.notify(event=event)
        assert env._last_event == event

    def test_add_event_raises_to_main_thread(self):
        class FeatureNewsSentiment(IState):
            space = gym.spaces.Box(-np.inf, +np.inf, (1,), np.float64)

            def parse(self):
                return np.array([1.])

            def process_IEvent(self, event):
                raise ConnectionRefusedError()

        sentiment = FeatureNewsSentiment()
        env = self.get_discrete_env(state=sentiment)
        env.reset()
        with pytest.raises(ConnectionRefusedError):
            env.notify(IEvent())

    def test_add_event_updates_last_event_attribute(self):
        env = self.get_discrete_env()
        env.reset()
        event = IEvent()
        env.notify(event)
        assert env._last_event == event

    def test_add_event_notifies_observers(self):
        class EventTrendChange(IEvent):
            pass

        class State(IState):
            def __init__(self):
                self.news_sentiment = 0.
                self.trend = 0.

            def process_IEvent(self, event):
                self.news_sentiment = 42.

            def process_EventTrendChange(self, event):
                self.trend = 18.

        env = self.get_discrete_env(state=State())
        env.reset()
        assert env.state.news_sentiment == 0
        assert env.state.trend == 0
        env.notify(IEvent())
        assert env.state.news_sentiment == 42
        assert env.state.trend == 0
        env.notify(EventTrendChange())
        assert env.state.news_sentiment == 42
        assert env.state.trend == 18

    def test_reset_attribute_done(self):
        env = self.get_discrete_env()
        env.reset()
        assert env._done is False

    @unittest.mock.patch.object(IState, "reset", return_value=None)
    def test_reset_attribute_features(self, mock_reset):
        env = self.get_discrete_env()
        env.reset()
        mock_reset.assert_called_once_with(env.exchange, env.action_space, env.broker)

    @unittest.mock.patch.object(AbstractReward, "reset", return_value=None)
    def test_reset_attribute_reward(self, mock_reset):
        env = self.get_discrete_env()
        env.reset()
        mock_reset.assert_called_once_with()

    def test_reset_create_lob_for_cash(self):
        env = self.get_discrete_env()
        env.reset()
        contract_cash = env.action_space.contracts[0]
        assert env.exchange._books[contract_cash].bid_price == 1.0
        assert env.exchange._books[contract_cash].ask_price == 1.0
        assert env.exchange._books[contract_cash].bid_size == np.inf
        assert env.exchange._books[contract_cash].ask_size == np.inf

    def test_reset_attribute_exchange(self):
        env = self.get_discrete_env()
        env.reset()
        for contract, lob in env.exchange._books.items():
            if not isinstance(contract, (Cash, Rate)):  # tested separately
                assert np.isnan(env.exchange._books[contract].bid_price)
                assert np.isnan(env.exchange._books[contract].ask_price)
                assert np.isnan(env.exchange._books[contract].bid_size)
                assert np.isnan(env.exchange._books[contract].ask_size)

    def test_reset_attribute_observers(self):
        env = self.get_discrete_env()
        env.reset()
        assert len(env._observers) == 2
        assert env._observers[0] is env.state
        assert env._observers[1] == env.exchange

    def test_reset_instances_new_broker(self):
        env = self.get_discrete_env()
        env.reset()
        broker_pre = env.broker
        env.reset()
        broker_post = env.broker
        assert id(broker_pre) != id(broker_post)

    def test_reset_broker(self):
        env = self.get_discrete_env()
        env.reset()
        assert id(env.broker.exchange) == id(env.exchange)
        assert env.broker._initial_deposit == env._initial_cash

    def test_reset_sends_event_when_finished(self):
        env = self.get_discrete_env()
        env.reset()
        assert isinstance(env._last_event, EventReset)

    def test_reset_returns_current_state(self):
        env = self.get_discrete_env()
        obs = env.reset()
        assert obs is env.state

    def test_reset_events_are_added(self):
        env = self.get_discrete_env()
        env.reset()
        assert env.exchange[Cash()].mid_price == 1
        assert env.exchange[env._broker_fees.interest_rate].mid_price == 0

    def test_step_doesnt_raises_if_action_belongs_to_action_space(self):
        env = self.get_discrete_env()
        for action in range(env.action_space.n):
            env = self.get_discrete_env()
            env.reset()
            env.exchange.process_EventNBBO(EventNBBO(datetime.now(), Cash(), 1, 1))
            env.exchange.process_EventNBBO(EventNBBO(datetime.now(), ETF("Equities"), 1, 1))
            env.exchange.process_EventNBBO(EventNBBO(datetime.now(), ETF("Bonds"), 1, 1))
            env.step(action)

    def test_step_raises_if_episode_is_ended(self):
        env = self.get_discrete_env()
        env.reset()
        env._done = True
        with pytest.raises(EndOfEpisodeError):
            env.step(env.action_space.sample())

    def test_step_raises_if_step_when_account_nlv_is_negative(self):
        env = self.get_discrete_env()
        env.reset()
        _, _, done, _ = env.step(0)
        # Account has run out of capital. End of episode.
        env.broker._holdings_quantity = {env.broker.base_currency: -1}  # stub
        # Will raise EndOfEpisodeError because NLV <= 0 and done=True
        # Raises because should have returned done before stubbing. You can't
        # step in env if you are broker. You must reset first.
        with pytest.raises(EndOfEpisodeError):
            _, _, done, _ = env.step(env.action_space.sample())

    @unittest.mock.patch("tradingenv.env.Broker")
    @unittest.mock.patch.object(TradingEnv, "_process_nonlatent_events")
    def test_reset_send_process_nonlatent_events_is_called_once(
        self, mock_send_process_nonlatent_events, Broker
    ):
        env = self.get_discrete_env()
        env.reset()
        mock_send_process_nonlatent_events.assert_called_once()

    @unittest.mock.patch("tradingenv.env.Broker")
    @unittest.mock.patch.object(
        TradingEnv, "_process_nonlatent_events", return_value=False
    )
    def test_step_send_process_nonlatent_events_is_called_once(
        self, mock_send_process_nonlatent_events, Broker
    ):
        env = self.get_discrete_env()
        env.reset()
        env.exchange.process_EventNBBO(EventNBBO(datetime.now(), ETF("Equities"), 1, 2))
        env.exchange.process_EventNBBO(EventNBBO(datetime.now(), ETF("Bonds"), 1, 2))
        mock_send_process_nonlatent_events.assert_called_once()
        env.step(env.action_space.sample())
        assert mock_send_process_nonlatent_events.call_count == 2

    def test_step_raises_if_action_does_not_belong_to_action_space(self):
        env = self.get_discrete_env()
        env.reset()
        invalid_action = np.array([1, 1, 1])
        with pytest.raises(ValueError):
            env.step(invalid_action)

    def test_step_returns_state(self):
        env = self.get_discrete_env()
        env.reset()
        env.exchange.process_EventNBBO(EventNBBO(datetime.now(), ETF("Equities"), 1, 2))
        env.exchange.process_EventNBBO(EventNBBO(datetime.now(), ETF("Bonds"), 1, 2))
        state, _, _, _ = env.step(env.action_space.sample())
        assert state is env.state

    @unittest.mock.patch.object(Broker, "rebalance", return_value=None)
    def test_step_calls_broker_rebalance_just_once(self, broker_rebalance):
        class BoringReward(AbstractReward):
            def calculate(self, env):
                return 42

        env = self.get_discrete_env(reward=BoringReward())
        env.reset()
        action = env.action_space.sample()
        env.exchange.process_EventNBBO(EventNBBO(datetime.now(), ETF("Equities"), 1, 2))
        env.exchange.process_EventNBBO(EventNBBO(datetime.now(), ETF("Bonds"), 1, 2))
        env.step(action)
        broker_rebalance.assert_called_once()

    @unittest.mock.patch.object(AsynchronousTransmitter, "_now")
    @unittest.mock.patch.object(Broker, "accrued_interest")
    def test_step_calls_broker_rebalance_with_right_args(
        self, accrued_interest, transmitter_now
    ):
        class BoringReward(AbstractReward):
            def calculate(self, env):
                return 42

        accrued_interest.return_value = 0.0
        env = self.get_discrete_env(reward=BoringReward())
        env.reset()
        action = env.action_space.sample()
        env.exchange.process_EventNBBO(EventNBBO(datetime.now(), ETF("Equities"), 1, 2))
        env.exchange.process_EventNBBO(EventNBBO(datetime.now(), ETF("Bonds"), 1, 2))
        margin = 0
        env.step(action)
        assert env.broker.track_record[-1].time == transmitter_now()
        assert env.broker.track_record[-1].margin == margin
        actual = env.action_space.make_rebalancing_request(action).allocation
        desired = env.broker.track_record[-1].allocation
        assert actual == desired

    def test_step_returns_reward(self):
        env = self.get_discrete_env()
        env.reset()
        env.exchange.process_EventNBBO(EventNBBO(datetime.now(), ETF("Equities"), 1, 2))
        env.exchange.process_EventNBBO(EventNBBO(datetime.now(), ETF("Bonds"), 1, 2))
        action = env.action_space.sample()
        _, reward, _, _ = env.step(action)
        assert reward == env._reward.calculate(env)

    def test_default_state_type_if_verbosedict(self):
        class FeatureNewsSentiment(IState):
            observation_space = gym.spaces.Box(-np.inf, +np.inf, (1,), np.float32)

            def __init__(self):
                self.item = 0

            def process_EventStep(self, event):
                self.item = 42

            def parse(self):
                return np.array([self.item])

        feature = FeatureNewsSentiment()
        env = self.get_discrete_env(state=feature)
        env.reset()
        env.exchange.process_EventNBBO(EventNBBO(datetime.now(), ETF("Equities"), 1, 2))
        env.exchange.process_EventNBBO(EventNBBO(datetime.now(), ETF("Bonds"), 1, 2))
        state, _, _, _ = env.step(env.action_space.sample())
        np.testing.assert_equal(state, np.array([42.]))

    def test_step_returns_done(self):
        env = self.get_discrete_env()
        env.reset()
        env.exchange.process_EventNBBO(EventNBBO(datetime.now(), ETF("Equities"), 1, 2))
        env.exchange.process_EventNBBO(EventNBBO(datetime.now(), ETF("Bonds"), 1, 2))
        _, _, done, _ = env.step(env.action_space.sample())
        assert done is env._done

    def test_step_return_info(self):
        env = self.get_discrete_env()
        env.reset()
        env.exchange.process_EventNBBO(EventNBBO(datetime.now(), ETF("Equities"), 1, 2))
        env.exchange.process_EventNBBO(EventNBBO(datetime.now(), ETF("Bonds"), 1, 2))
        _, _, _, info = env.step(env.action_space.sample())
        assert isinstance(info, dict)

    def test_parse_reward_is_parsed_from_string(self):
        env = self.get_discrete_env(reward="RewardPnL")
        assert isinstance(env._reward, RewardPnL)

    def test_parse_reward_raises_if_unexisting_reward(self):
        with pytest.raises(AttributeError):
            # AttributeError: module 'tradingenv.rewards' has no attribute
            # 'UnexistingReward'
            self.get_discrete_env(reward="UnexistingReward")

    def test_EventDone_is_sent_when_finished(self):
        transmitter = Transmitter([datetime(2019, 1, 2), datetime(2019, 1, 3)])
        transmitter.add_events(
            [
                EventNBBO(datetime(2019, 1, 2), ETF("Equities"), 1, 2),
                EventNBBO(datetime(2019, 1, 2), ETF("Bonds"), 1, 2),
                EventNBBO(datetime(2019, 1, 3), ETF("Equities"), 1, 2),
                EventNBBO(datetime(2019, 1, 3), ETF("Bonds"), 1, 2),
            ]
        )
        env = self.get_discrete_env(transmitter=transmitter)
        env.reset()
        env.step(action=env.action_space.sample())
        assert isinstance(env._last_event, EventDone)

    def test_env_now_delegates_to_transmitter(self):
        transmitter = Transmitter([datetime(2019, 1, 2), datetime(2019, 1, 3)])
        transmitter.add_events(
            [
                EventNBBO(datetime(2019, 1, 2), ETF("Equities"), 1, 2),
                EventNBBO(datetime(2019, 1, 2), ETF("Bonds"), 1, 2),
                EventNBBO(datetime(2019, 1, 3), ETF("Equities"), 1, 2),
                EventNBBO(datetime(2019, 1, 3), ETF("Bonds"), 1, 2),
            ]
        )
        env = self.get_discrete_env(transmitter=transmitter)
        env.reset()
        assert env._now == datetime(2019, 1, 2)
        _, _, done, _ = env.step(action=env.action_space.sample())
        assert env._now == datetime(2019, 1, 3)
        assert done

    def test_error_is_raised_when_bug_in_Feature(self):
        class BuggedFeature(IState):
            space = gym.spaces.Box(-np.inf, +np.inf, (1,), np.float64)

            def __init__(self):
                self.item = 0

            def process_EventStep(self, event):
                self.item = 0 / 0

            def parse(self):
                return np.array([self.item])

        env = self.get_discrete_env(state=BuggedFeature())
        env.reset()
        env.exchange.process_EventNBBO(EventNBBO(datetime.now(), ETF("Equities"), 1, 2))
        env.exchange.process_EventNBBO(EventNBBO(datetime.now(), ETF("Bonds"), 1, 2))
        with pytest.raises(ZeroDivisionError):
            env.step(env.action_space.sample())

    def test_reset_event_time_when_backtesting(self):
        transmitter = Transmitter([datetime(2019, 1, 2), datetime(2019, 1, 3)])
        transmitter.add_events(
            [
                EventNBBO(datetime(2019, 1, 2), ETF("Equities"), 1, 2),
                EventNBBO(datetime(2019, 1, 2), ETF("Bonds"), 1, 2),
            ]
        )
        env = self.get_discrete_env(transmitter=transmitter)
        env.reset()
        assert env._last_event.time == datetime(2019, 1, 2)

    def test_env_with_continuous_portfolio(self):
        transmitter = Transmitter(
            timesteps=[datetime(2019, 1, 2), datetime(2019, 1, 3)]
        )
        transmitter.add_events(
            [
                EventNBBO(datetime(2019, 1, 2), ETF("Equities"), 1, 2),
                EventNBBO(datetime(2019, 1, 2), ETF("Bonds"), 1, 2),
                EventNBBO(datetime(2019, 1, 3), ETF("Equities"), 1, 2),
                EventNBBO(datetime(2019, 1, 3), ETF("Bonds"), 1, 2),
            ]
        )
        action_space = BoxPortfolio([Cash(), ETF("Equities"), ETF("Bonds")])
        env = TradingEnv(action_space=action_space, transmitter=transmitter)
        env.reset()
        assert env._now == datetime(2019, 1, 2)
        _, _, done, _ = env.step(action=env.action_space.sample())
        assert env._now == datetime(2019, 1, 3)
        assert done

    def test_env_reset_sets_dataset_name_in_transmitter(self):
        transmitter = Transmitter(
            timesteps=[
                datetime(2019, 1, 1),
                datetime(2019, 1, 2),
                datetime(2019, 1, 3),
                datetime(2019, 1, 4),
                datetime(2019, 1, 5),
                datetime(2019, 1, 6),
            ],
            folds={
                "training-set": (datetime(2019, 1, 1), datetime(2019, 1, 3)),
                "validation-set": (datetime(2019, 1, 4), datetime(2019, 1, 4)),
                "test-set": (datetime(2019, 1, 5), datetime(2019, 1, 6)),
            },
        )
        transmitter.add_events(
            [
                EventNBBO(datetime(2019, 1, 1), ETF("Equities"), 1, 1),
                EventNBBO(datetime(2019, 1, 2), ETF("Equities"), 3, 3),
                EventNBBO(datetime(2019, 1, 3), ETF("Equities"), 5, 5),
                EventNBBO(datetime(2019, 1, 4), ETF("Equities"), 10, 10),
                EventNBBO(datetime(2019, 1, 5), ETF("Equities"), 20, 20),
                EventNBBO(datetime(2019, 1, 6), ETF("Equities"), 22, 22),
                EventNBBO(datetime(2019, 1, 1), ETF("Bonds"), 2, 2),
                EventNBBO(datetime(2019, 1, 2), ETF("Bonds"), 4, 4),
                EventNBBO(datetime(2019, 1, 3), ETF("Bonds"), 6, 6),
                EventNBBO(datetime(2019, 1, 4), ETF("Bonds"), 11, 11),
                EventNBBO(datetime(2019, 1, 5), ETF("Bonds"), 21, 21),
                EventNBBO(datetime(2019, 1, 6), ETF("Bonds"), 23, 23),
            ]
        )
        env = TradingEnv(
            action_space=self.get_discrete_action_space(), transmitter=transmitter
        )

        # Validation set.
        env.reset("validation-set")
        assert env._now == datetime(2019, 1, 4)
        assert env.exchange[ETF("Equities")].time == datetime(2019, 1, 4)
        assert env.exchange[ETF("Equities")].mid_price == 10
        assert env.exchange[ETF("Bonds")].time == datetime(2019, 1, 4)
        assert env.exchange[ETF("Bonds")].mid_price == 11

        # Training set.
        env.reset("training-set")
        assert env._now == datetime(2019, 1, 1)
        assert env.exchange[ETF("Equities")].time == datetime(2019, 1, 1)
        assert env.exchange[ETF("Equities")].mid_price == 1
        assert env.exchange[ETF("Bonds")].time == datetime(2019, 1, 1)
        assert env.exchange[ETF("Bonds")].mid_price == 2

        # Test set.
        env.reset("test-set")
        assert env._now == datetime(2019, 1, 5)
        assert env.exchange[ETF("Equities")].time == datetime(2019, 1, 5)
        assert env.exchange[ETF("Equities")].mid_price == 20
        assert env.exchange[ETF("Bonds")].time == datetime(2019, 1, 5)
        assert env.exchange[ETF("Bonds")].mid_price == 21

        # Default set when env.reset is training set.
        env.reset()
        assert env._now == datetime(2019, 1, 1)
        assert env.exchange[ETF("Equities")].time == datetime(2019, 1, 1)
        assert env.exchange[ETF("Equities")].mid_price == 1
        assert env.exchange[ETF("Bonds")].time == datetime(2019, 1, 1)
        assert env.exchange[ETF("Bonds")].mid_price == 2

    def test_sample_episode(self):
        class EventNews(IEvent):
            def __init__(self, time: datetime, score: float, news: str):
                self.time = time
                self.score = score
                self.news = news

        class FeatureNewsSentiment(IState):
            space = gym.spaces.Box(-np.inf, +np.inf, (2,), np.float64)

            def __init__(self):
                self.score = 0.

            def process_EventNews(self, event: EventNews):
                self.score = event.score

            def parse(self):
                return np.array([self.score, 0.0])

        dates = pd.date_range("2019-01-01", periods=5)
        prices = pd.DataFrame(
            data={ETF("Equities"): [1, 2, 3, 4, 5], ETF("Bonds"): [10, 11, 12, 13, 14]},
            index=dates,
        )
        news = pd.DataFrame(
            data={
                "score": [-0.2, -0.7, +0.1, +0.6, -0.3],
                "news": ["Bad", "Nasty", "Decent", "Great", "Not great"],
            },
            index=dates,
        )
        transmitter = Transmitter(
            timesteps=prices.index,
            folds={
                "training-set": [datetime(2019, 1, 1), datetime(2019, 1, 3)],
                "test-set": [datetime(2019, 1, 4), datetime(2019, 1, 5)],
            },
        )
        transmitter.add_prices(prices)
        transmitter.add_custom_events(news, EventNews)
        env = TradingEnv(
            action_space=BoxPortfolio([Cash(), ETF("Equities"), ETF("Bonds")]),
            transmitter=transmitter,
            state=FeatureNewsSentiment(),
        )
        track_record = env.backtest()

        # Test default transmitter partition ('training-set').
        #assert track_record.interactions['time'] == [
        #    datetime(2019, 1, 1),
        #    datetime(2019, 1, 2),
        #    datetime(2019, 1, 3),
        #]
        # Note state has been dropped because it might not be pickable. Hovever,
        # the history of states is preserved.
        #np.testing.assert_equal(
        #    actual=track_record.interactions['states'],
        #    desired=[
        #        np.array([-0.2, 0.0]),
        #        np.array([-0.7, 0.0]),
        #        np.array([0.1, 0.0]),
        #    ],
        #)
        #np.testing.assert_allclose(
        #    actual=np.nansum(track_record.interactions['actions'], axis=1), desired=[1.0, 1.0, 0.0]
        #)
        #np.testing.assert_equal(
        #    actual=track_record.interactions['actions'][-1], desired=np.array([np.nan, np.nan, np.nan])
        #)
        #np.testing.assert_allclose(
        #    actual=np.isreal(track_record.interactions['rewards']), desired=[True, True, True]
        #)
        #assert track_record.interactions['rewards'][0] is np.nan
        assert track_record == env.broker.track_record

        # Test test-set.
        track_record = env.backtest("test-set")
        #assert track_record.interactions['time'] == [datetime(2019, 1, 4), datetime(2019, 1, 5)]
        # Note state has been dropped because it might not be pickable. Hovever,
        # the history of states is preserved.
        #np.testing.assert_equal(
        #    actual=track_record.interactions['states'], desired=[np.array([0.6, 0.0]), np.array([-0.3, 0.0])]
        #)
        #np.testing.assert_allclose(
        #    actual=np.sum(track_record.interactions['actions'], axis=1), desired=[1.0, np.nan]
        #)
        #np.testing.assert_equal(
        #    actual=track_record.interactions['actions'][-1], desired=np.array([np.nan, np.nan, np.nan])
        #)
        #np.testing.assert_allclose(
        #    actual=np.isreal(track_record.interactions['rewards']), desired=[True, True]
        #)
        #assert track_record.interactions['rewards'][0] is np.nan
        assert track_record == env.broker.track_record

    def test_now_with_asynchronous_transmitter_is_live(self):
        env = TradingEnv(
            action_space=BoxPortfolio([Cash(), ETF("Equities"), ETF("Bonds")])
        )
        # reset.
        env.reset()
        now0 = env.now()
        time.sleep(0.001)
        now1 = env.now()
        assert now0 != now1

    def test_now_with_synchronous_transmitter_is_not_live(self):
        transmitter = Transmitter([datetime(2019, 1, 2), datetime(2019, 1, 3)])
        transmitter.add_events(
            [
                EventNBBO(datetime(2019, 1, 2), ETF("Equities"), 1, 2),
                EventNBBO(datetime(2019, 1, 2), ETF("Bonds"), 1, 2),
                EventNBBO(datetime(2019, 1, 3), ETF("Equities"), 1, 2),
                EventNBBO(datetime(2019, 1, 3), ETF("Bonds"), 1, 2),
            ]
        )
        env = TradingEnv(
            action_space=BoxPortfolio([Cash(), ETF("Equities"), ETF("Bonds")]),
            transmitter=transmitter,
        )
        # reset.
        env.reset()
        now0 = env.now()
        time.sleep(0.001)
        now1 = env.now()
        assert now0 == now1

    def test_cash_only_strategy(self):
        idle_cash_rate = Rate("FED funds rate")
        timesteps = pd.date_range("2018-12-31", "2020-01-01")
        transmitter = Transmitter(timesteps=timesteps)
        events = [
            EventNBBO(time, idle_cash_rate, 0.01, 0.01) for time in timesteps
        ]
        transmitter.add_events(events)
        env = self.get_discrete_env(transmitter=transmitter)

        env.reset()
        done = False
        while not done:
            action = 0
            state, reward, done, info = env.step(action)

        nlv = env.broker.track_record.net_liquidation_value(before_rebalancing=False)
        assert np.isclose(float(nlv.cagr().item()), 0.01)

    def test_raises_when_latency_is_too_high(self):
        transmitter = Transmitter([datetime(2019, 1, 2), datetime(2019, 1, 3)])
        transmitter.add_events(
            [
                EventNBBO(datetime(2019, 1, 2), ETF("Equities"), 1, 2),
                EventNBBO(datetime(2019, 1, 2), ETF("Bonds"), 1, 2),
            ]
        )
        with pytest.raises(ValueError):
            self.get_discrete_env(transmitter=transmitter, latency=1e10)

    @pytest.mark.parametrize(
        argnames='latency, expected_quantity',
        argvalues=[
            (0, 100),
            (0.99, 100),
            (1, 50),
            (1.01, 50),
        ],
    )
    def test_latency_introduces_slippage(self, latency, expected_quantity):
        transmitter = Transmitter(
            timesteps=[
                datetime(2019, 1, 1, 15, 59, 0),
                datetime(2019, 1, 1, 16, 0, 0),
                datetime(2019, 1, 2, 9, 0, 0)
            ]
        )
        transmitter.add_events(
            [
                EventNBBO(datetime(2019, 1, 1, 15, 59, 0), ETF("SPY"), 1, 1),
                EventNBBO(datetime(2019, 1, 1, 15, 59, 1), ETF("SPY"), 2, 2),
            ]
        )
        env = TradingEnv(
            action_space=[Cash(), ETF('SPY')],
            transmitter=transmitter,
            latency=latency,
            initial_cash=100,
        )
        env.reset()
        env.step(action=np.array([0, 1]))
        assert env.broker.holdings_quantity[ETF('SPY')] == expected_quantity

    @pytest.mark.parametrize(
        argnames='latency, expected_quantity',
        argvalues=[
            (0, 100),
            (0.1, 100),
        ],
    )
    def test_short_latency_is_not_enough_to_trade_at_the_next_bar(self, latency, expected_quantity):
        """After 0.1 seconds after 2019-01-01 00:00 the price is still 1,
        so we still buy 100 shares."""
        transmitter = Transmitter(
            timesteps=[
                datetime(2019, 1, 1),
                datetime(2019, 1, 2),
                datetime(2019, 1, 3),
            ]
        )
        transmitter.add_events(
            [
                EventNBBO(datetime(2019, 1, 1), ETF("SPY"), 1, 1),
                EventNBBO(datetime(2019, 1, 2), ETF("SPY"), 2, 2),
                EventNBBO(datetime(2019, 1, 3), ETF("SPY"), 3, 3),
            ]
        )
        env = TradingEnv(
            action_space=[Cash(), ETF('SPY')],
            transmitter=transmitter,
            latency=latency,  # 0.1 seconds of delay are enough to trade at the next bar
            initial_cash=100,
        )
        env.reset()
        env.step(action=np.array([0, 1]))
        assert env.broker.holdings_quantity[ETF('SPY')] == expected_quantity

    @pytest.mark.parametrize(
        argnames='delay_steps, expected_quantity',
        argvalues=[
            (0, [100., 100., 100.]),
            (1, [0., 50., 50.]),
            (2, [0., 0., 25.]),
        ],
    )
    def test_delay_steps(self, delay_steps, expected_quantity):
        """After 0.1 seconds after 2019-01-01 00:00 the price is still 1,
        so we still buy 100 shares."""
        transmitter = Transmitter(
            timesteps=[
                datetime(2019, 1, 1),
                datetime(2019, 1, 2),
                datetime(2019, 1, 3),
                datetime(2019, 1, 4),
            ]
        )
        transmitter.add_events(
            [
                EventNBBO(datetime(2019, 1, 1), ETF("SPY"), 1, 1),
                EventNBBO(datetime(2019, 1, 2), ETF("SPY"), 2, 2),
                EventNBBO(datetime(2019, 1, 3), ETF("SPY"), 4, 4),
                EventNBBO(datetime(2019, 1, 4), ETF("SPY"), 8, 8),
            ]
        )
        env = TradingEnv(
            action_space=[Cash(), ETF('SPY')],
            transmitter=transmitter,
            initial_cash=100,
            steps_delay=delay_steps,
        )
        env.reset()
        for expected in expected_quantity:
            observation, reward, done, info = env.step(action=np.array([0, 1]))
            actual = env.broker.holdings_quantity.get(ETF('SPY'), 0.)
            assert actual == expected
        assert done

    def test_reset_exchange_is_injected_as_class_attribute_to_Feature(self):
        env = self.get_discrete_env()
        env.reset()
        assert id(env.exchange) == id(env.state.exchange)

    
class TestTradingEnvFutures:
    def load_futures(self):
        mapper = {
            "CME-MINI S&P 500 INDEX DEC 2018 - SETT. PRICE": ES(2018, 12),
            "CME-MINI S&P 500 INDEX MAR 2019 - SETT. PRICE": ES(2019, 3),
            "CME-MINI S&P 500 INDEX JUN 2019 - SETT. PRICE": ES(2019, 6),
        }
        filename = inspect.getframeinfo(inspect.currentframe()).filename
        path = os.path.abspath(filename)
        directory = os.path.dirname(path)
        prices = pd.read_csv(
            filepath_or_buffer=os.path.join(directory, "data", "ES.csv"),
            index_col="Date",
            parse_dates=True,
            dayfirst=True,
            usecols=["Date"] + list(mapper),
        )
        prices.dropna(how="all", inplace=True)
        prices.rename(mapper=mapper, axis="columns", inplace=True)
        return prices

    def test_env_does_not_raise_if_action_belongs_to_space(self):
        # i.e. multiplier is applied after checking if the action belongs to
        # the action space.
        action_space = BoxPortfolio([Cash(), ES(2018, 12), ES(2019, 3)])
        prices = self.load_futures()
        prices.dropna(how="any", inplace=True)
        env = TradingEnv(action_space, prices=prices)
        env.reset()
        action = np.array([0, 0.6, 0.4])
        env.step(action)  # does not raise

    def test_env_raises_if_action_does_not_belong_to_space(self):
        action_space = BoxPortfolio([Cash(), ES(2018, 12), ES(2019, 3)])
        prices = self.load_futures()
        prices.dropna(how="any", inplace=True)
        env = TradingEnv(action_space, prices=prices)
        env.reset()
        action = np.array([0, -1, 0])  # box is in [0, 1], so invalid action
        with pytest.raises(ValueError):
            env.step(action)

    def test_sample_episode_with_contracts_with_multiplier_does_not_raise(self):
        action_space = BoxPortfolio([Cash(), ES(2018, 12)])
        prices = self.load_futures()
        prices = prices[[ES(2018, 12)]]
        prices.dropna(inplace=True)
        prices = prices.loc[: ES(2018, 12).last_trading_date]
        env = TradingEnv(action_space, prices=prices)

        env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
