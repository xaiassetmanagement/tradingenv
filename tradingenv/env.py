"""The core module of trading-gym. TradingEnv is your market simulator to run
backtests or train reinforcement learning agents."""
from trading_gym.spaces import PortfolioSpace, BoxPortfolio
from trading_gym.broker.broker import Broker, EndOfEpisodeError
from trading_gym.broker.track_record import TrackRecord
from trading_gym.broker.fees import IBrokerFees, BrokerFees
from trading_gym.exchange import Exchange
from trading_gym.contracts import AbstractContract, Cash
from trading_gym.rewards import make_reward, AbstractReward, RewardSimpleReturn
from trading_gym.policy import make_policy
from trading_gym.dashboard.core import Dashboard
from trading_gym.state import IState
from trading_gym.features import Feature
from trading_gym.events import (
    Observer,
    IEvent,
    EventReset,
    EventStep,
    EventDone,
    EventNBBO,
    EventNewDate,
)
from trading_gym.transmitter import (
    AbstractTransmitter,
    AsynchronousTransmitter,
    Transmitter,
    TRAINING_SET,
)
from typing import Tuple, Sequence, Union, List, Any, Optional
from pandas.core.generic import NDFrame
from datetime import datetime
from tqdm import tqdm
from collections import deque
import pandas as pd
import numpy as np
import threading
import gym


class TradingEnv(gym.Env):
    """TradingEnv is the core class of trading-gym, representing a market
    simulator to run backtests or train reinforcement learning agents."""

    metadata = {"render.modes": ["dashboard"]}

    def __init__(
        self,
        action_space: Union[PortfolioSpace, list],
        state: Union[IState, Sequence[Feature]] = IState(),
        reward: Union[AbstractReward, str] = RewardSimpleReturn(),
        transmitter: AbstractTransmitter = AsynchronousTransmitter(),
        prices: pd.DataFrame = None,
        initial_cash: float = 100,
        broker_fees: IBrokerFees = BrokerFees(),
        latency: float = 0,
        fit_transformers: Union[bool, dict] = False,
        # episode_length: int = None,
        steps_delay: int = 0,
        sampling_span: int = None
    ):
        """
        Parameters
        ----------
        action_space
            An instance of PortfolioDiscrete or PortfolioContinuous,
            characterizing the domain of the action observation_space (target weights of
            the portfolio with eventual constraints) and contracts to be traded.
            An equivalent way to use BoxPortfolio consists in providing
            a sequence of contract _names.
        state : State
            A State instance or a list of Features.
        reward : Union[AbstractReward, str]
            The reward function, evaluated at every step. This could either
            be a custom reward (AbstractReward interface implementation) or
            the id of an existing reward (e.g. 'PnL'). For the full list
            of supported rewards see trading_env.rewards.
        transmitter : Transmitter
            Object taking care of delivering events during the interaction
            with the environment (e.g. market prices).
        prices : pd.DataFrame
            A pandas.DataFrame whose columns are asset _names, index are
            timestamps and values are prices. This optional argument allows
            you to avoid specifying a transmitter.
        broker_fees: IBrokerFees
            Instance AbstractBrokerFees with custom initialization or implement
            your own AbstractBrokerFees to set transaction costs.
        sampling_span: int
            This argument is used only if episode_length is passed in
            TrandingEnv.reset. If specified, the episode start date is sampled
            using an exponentially decaying probability. sampling_span is the
            number of recent timesteps that captures ~70% of the likelihood of
            sampling a start date from such window. Uniform distribution is
            used by default if this parameter is not provided. It's useful to
            specify a value if you wish to train RL agents to overweight
            observations in the more recent past.
        """
        if not isinstance(action_space, PortfolioSpace):
            # action observation_space is assumed to be a sequence of contracts.
            action_space = BoxPortfolio(action_space)
        if prices is not None:
            prices.validate()  # source code in metrics.py
            transmitter = Transmitter(prices.index)
            transmitter.add_prices(prices)

        # Inputs.
        self.action_space: PortfolioSpace = action_space
        self.state = IState(state) if isinstance(state, list) else state
        self.observation_space = self.state.space
        # self.episode_length = episode_length  # TODO: test
        self._verify_state = True
        self._reward = make_reward(reward)
        self._initial_cash = initial_cash
        self._broker_fees = broker_fees
        self._transmitter = transmitter
        self._latency = latency
        self._real_time = isinstance(transmitter, AsynchronousTransmitter)
        self._steps_delay =  steps_delay
        self._queue_actions: Optional[deque] = None
        self._sampling_span = sampling_span

        # Add extra events to transmitter and create partitions of events.
        for contract in self.action_space.contracts:
            self._transmitter.add_events(contract.make_events())
        self._transmitter._create_partitions(latency)

        # TODO: raise if there is a callback method registered to an event
        #  which does not exist anywhere in the transmitter.

        # Set by TradingEnv.reset.
        self._done: Union[bool, None] = None
        self.exchange: Union[Exchange, None] = None
        self.broker: Union[Broker, None] = None
        self._last_event: Union[IEvent, None] = None
        self._observers: Union[Sequence[Observer], None] = None
        self._now: Union[datetime, None] = None
        self._events_nonlatent: Union[List[IEvent], None] = None

        # Run procedure to fit transformers.
        if fit_transformers:
            # TODO: Test.
            kwargs = fit_transformers if isinstance(fit_transformers, dict) else dict()
            # needed to collect historical values of features
            # self.episode_length = None
            # self.backtest(**kwargs, episode_length=None)
            self.backtest(**kwargs)
            for feature in self.state.features:
                feature.fit_transformer()
            # self.episode_length = episode_length
            # if transformers have been fit, this verification will fail.
            #   Solution is to verify in parse by each feature and not here if
            #   transformers have been fit.
            # By default features before not transformation are verified anyway
            # even if verify state is False, so we should be still safe.
            self._verify_state = False

    def now(self) -> datetime:
        """Returns current time."""
        return self._transmitter._now() if self._real_time else self._now

    def reset(
            self,
            fold: str = TRAINING_SET,
            episode_length: int = None,
    ) -> IState:
        """Reset the environment to start a new episode. It's responsibility
        of the user to call this method before starting a new episode.

        Parameters
        ----------
        fold : str
            Only data from the specified fold will be used to simulate the
            episode. To get the full list of available folds check
            TradingEnv._transmitter.folds. By default, all data belong to a
            single fold named 'training-set'. Custom folds can be provided
            by instancing Transmitter with the optional argument 'folds'. You
            probably don't need to use this parameter unless you want to
            separate training from test data or run a walk-forward test.
        episode_length : int
            If specified, the episode will stop after this number of states.
            If not provided, the episode will last until when there are no
            more data (e.g. a complete pass trough the 'training-set' fold).
        seed : int
            This method should also reset the environment's random number
            generator(s) if `seed` is an integer or if the environment has not
            yet initialized a random number generator. If the environment already
            has a random number generator and `reset` is called with `seed=None`,
            the RNG should not be reset.
            Moreover, `reset` should (in the typical use case) be called with an
            integer seed right after initialization and then never again.
        return_info
            Not supported but here for OpenAI-gym compatibility.
        options
            Not supported but here for OpenAI-gym compatibility.

        Returns
        -------
        Initial state of the environment.
        """
        # episode_length = episode_length or self.episode_length  # TODO: test

        # Reset attributes.
        self._done = False
        self._last_event = None
        self._queue_actions = deque(
            [self.action_space.null_action() for _ in range(self._steps_delay)],
            maxlen=self._steps_delay + 1,
        )
        self._reward.reset()
        self.exchange = Exchange()
        self.broker = Broker(
            exchange=self.exchange,
            base_currency=self.action_space.base_currency,
            deposit=self._initial_cash,
            fees=self._broker_fees,
        )
        self.state.reset(self.exchange, self.action_space, self.broker)
        self._observers = (self.state, self.exchange)
        if self.state.features is not None:
            self._observers += tuple(self.state.features)
        self._now = None
        self._events_nonlatent = None

        # Process reset events.
        self.exchange.process_EventNBBO(EventNBBO(self.now(), Cash(), 1.0, 1.0))
        self.exchange.process_EventNBBO(EventNBBO(self.now(), self._broker_fees.interest_rate, 0.0, 0.0))

        # Instance generator of events to be sent at every interaction.
        self._transmitter._reset(fold, episode_length, self._sampling_span)
        self._events_latent, self._events_nonlatent = self._transmitter._next()
        self._process_latent_events()
        self._process_nonlatent_events()

        # Tear down events to notify observers.
        self.notify(EventReset(self.now(), self.broker.track_record))
        if self._done:
            self.notify(EventDone(self.now(), self.broker))
        return self.state(self._verify_state)

    def step(self, action: Union[int, np.ndarray]) -> Tuple[Any, float, bool, dict]:
        """
        Parameters
        ----------
        action : Union[int, np.ndarray]
            If the action observation_space is discrete, action is an integer representing
            the action ID. If the action observation_space is continuous, action is a
            np.array with the target weights of the portfolio to be rebalanced
            as indicated in TradingEnv.action_space. The weight will be applied
            to the notional trading value of the contract (transaction_price * multiplier).

        Returns
        -------
        A 4-tuple with: observation, reward, done, info.
        """
        if self._done:
            raise EndOfEpisodeError(
                "The current episode has ended. To start a new episode use "
                "TradingEnv.reset()."
            )
        self._queue_actions.appendleft(action)
        action = self._queue_actions.pop()
        self._process_latent_events()
        rebalancing = self.action_space.make_rebalancing_request(action, self.now(), self.broker)
        try:
            self.broker.rebalance(rebalancing)
        except EndOfEpisodeError:
            info = dict()
            self._done = True
        else:
            info = {"_rebalancing": rebalancing}
        self._process_nonlatent_events()
        reward = self._reward.calculate(self)

        # Tear down events to notify observers.
        self.notify(EventStep(self.now(), self.broker.track_record, action))
        if self._done:
            # _process again to update the state with last data points.
            # Useful in a production environment, redundant otherwise.
            self.notify(EventDone(self.now(), self.broker))
        return self.state(self._verify_state), reward, self._done, info

    def render(
            self,
            mode: str = "dashboard",
            track_records: List[Union[str, TrackRecord]]=None,
            debug: bool=False,
            port: int=8050,
    ):
        """You have two options to visualize your results: (1) Use methods of
        TradingEnv.broker.track_record or (2) use TradingEnv.render()"""
        if mode not in self.metadata["render.modes"]:
            raise ValueError(
                "Mode {} is not supported. Supported modes are {}."
                "".format(mode, self.metadata["render.modes"])
            )
        if mode == 'dashboard':
            # Note: there is no clean way for the user to stop this thread :(
            # TODO: support dynamically adding/track_records
            track_records = track_records or [self.broker.track_record]
            dashboard = Dashboard(track_records)
            thread = threading.Thread(
                daemon=True,
                target=dashboard.app.run_server,
                kwargs=dict(
                    debug=debug,
                    port=port,
                    dev_tools_silence_routes_logging=True,
                ),
            )
            thread.start()
            dashboard._thread = thread
            return dashboard

    def backtest(
        self,
        fold: str = TRAINING_SET,
        policy=None,
        episode_length: int = None,
        benchmark: NDFrame = None,
        risk_free: NDFrame = None,
        progress_bar: bool = False,
    ) -> "TrackRecord":
        """
        Parameters
        ----------
        fold : str
            Only data from the specified fold will be used to simulate the
            episode. To get the full list of available folds check
            TradingEnv._transmitter.folds. By default, all data belong to a
            single fold named ''training-set'. Custom folds can be provided
            by instancing Transmitter with the optional argument 'folds'. You
            probably don't need to use this parameter unless you want to
            separate training from test data or run a walk-forward test.
        policy : TFPolicyGraph
            A an optional policy trained using the 'ray' framework. This is
            usually retrieved with agent.get_policy(). If not provided, actions
            will be randomly sampled from the action observation_space.
        episode_length : int
            If specified, the episode will stop after this number of states.
            If not provided, the episode will last until when there are no
            more data (e.g. a complete pass trough the 'training-set' fold).
        benchmark : NDFrame
            A NDFrame representing the levels series that you want to use to
            benchmark your investment strategy. This argument is mostly used
            when rendering the environment or calculating tearsheets.
        risk_free : NDFrame
            Risk free rate of the market (e.g. T-Bills, EURIBOR) expressed as
            a level pandas Series or DataFrame.
        progress_bar
            A boolean indicating whether or not a progress bar should be
            displayed while running the backtest. False by default.

        Returns
        -------
        An Episode instance. Interactions (state, action, reward) are stored
        using Barto and Sutton's notation: (S_t, A_t, R_t) where A_t has been
        computed from S_t and R_t is associated with A_t-1.
        """
        policy = make_policy(policy, self.action_space, self.observation_space)
        state = self.reset(fold, episode_length)
        done = False
        progress_bar_context = tqdm(
            total=len(self._transmitter._steps),
            disable=not progress_bar,
        )
        with progress_bar_context as progress_bar:
            progress_bar.update(1)
            while not done:
                now = self.now()
                progress_bar.set_description(
                    "[{start} : {end}] {policy}\t{now}"
                    "".format(
                        policy=policy,
                        start=self._transmitter._steps[0].date(),
                        now=now,
                        end=self._transmitter._steps[-1].date(),
                    )
                )
                action = policy.act(state)
                state, reward, done, info = self.step(action)
                progress_bar.update(1)
            progress_bar.update(1)

        # Returns track_record with metadata.
        # TODO: refactor.
        # TODO: Config stores hyperparams of the env such as BrokerFees,
        #  risk_free, traded contracts, action_space, ...
        track_record = self.broker.track_record
        track_record.fold = fold
        track_record.risk_free = risk_free
        track_record.benchmark = benchmark
        track_record.name = repr(policy)
        track_record.state_history = self.state.history
        return track_record

    def _is_new_date(self, time: datetime):
        # TODO: test/refactor.
        return (
            # True when resetting the environment, first event.
            self._last_event is not None and
            # In case time was not set during concrete event initialization.
            self._last_event.time is not None and
            time is not None and
            self._last_event.time is not np.nan and
            time is not np.nan and
            # Is new date.
            self._last_event.time.date() != time.date()
        )

    def notify(self, event: IEvent):
        """Add event to the queue to be processed by the dedicated thread.

        Parameters
        ----------
        event : IEvent
            An event to be processed by the environment. All features which are
            registered to observe this event (i.e. Feature instance implements
            the method process_<EventName>) will update their values
            accordingly.
        """
        AbstractContract.now = event.time
        self._now = event.time
        if self._is_new_date(event.time):
            # 'event' is the first event of the day. Notify that it's a new date
            self.notify(EventNewDate(self._last_event.time, self.broker))
        event.notify(self._observers)
        self._last_event = event

    def _process_latent_events(self):
        """These events are run before after the action of the agent AND before
        executing the rebalancing request in order to simulate a latency."""
        for event in self._events_latent:
            self.notify(event)
        self._events_latent = list()

    def _process_nonlatent_events(self):
        """Send events from the transmitter (if any). The episode ends if
        Transmitter has been iterated through all its timesteps. Continuing
        indefinitely if you are using AsynchronousTransmitter."""
        for event in self._events_nonlatent:
            self.notify(event)
        try:
            self._events_latent, self._events_nonlatent = self._transmitter._next()
        except StopIteration:
            self._done = True
