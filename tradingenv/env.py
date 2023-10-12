"""The core module of trading-gym. TradingEnv is your market simulator to run
backtests or train reinforcement learning agents."""
from tradingenv.spaces import PortfolioSpace, BoxPortfolio
from tradingenv.broker.broker import Broker, EndOfEpisodeError
from tradingenv.broker.track_record import TrackRecord
from tradingenv.broker.fees import IBrokerFees, BrokerFees
from tradingenv.exchange import Exchange
from tradingenv.contracts import AbstractContract, Cash, Asset, Rate
from tradingenv.rewards import make_reward, AbstractReward, RewardSimpleReturn, LogReturn
from tradingenv.policy import make_policy
from tradingenv.state import IState, State
from tradingenv.features import Feature
from tradingenv.events import (
    Observer,
    IEvent,
    EventReset,
    EventStep,
    EventDone,
    EventNBBO,
    EventNewDate,
    EventNewObservation,
)
from tradingenv.transmitter import (
    AbstractTransmitter,
    AsynchronousTransmitter,
    Transmitter,
    TRAINING_SET,
)
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from typing import Tuple, Sequence, Union, List, Any, Optional
from pandas.core.generic import NDFrame
from datetime import datetime, timedelta
from tqdm import tqdm
from collections import deque, defaultdict
import pandas_market_calendars
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
        action_space: Union[List[AbstractContract], PortfolioSpace],
        state: Union[IState, List[Feature]] = IState(),
        reward: Union[str, AbstractReward] = RewardSimpleReturn(),
        transmitter: AbstractTransmitter = AsynchronousTransmitter(),
        prices: pd.DataFrame = None,
        initial_cash: float = 100,
        broker_fees: IBrokerFees = BrokerFees(),
        latency: float = 0,
        steps_delay: int = 0,
        fit_transformers: Union[bool, dict] = False,
        episode_length: int = None,
        sampling_span: int = None
    ):
        """
        Parameters
        ----------
        action_space: Union[List[AbstractContract], PortfolioSpace]
            This determines the investment universe of assets that can be
            traded during the simulations, as well any extra logic necessary to
            map an action to a portfolio of assets. To define a continuous
            long-only and unleveraged trading environment, it sufficient to
            pass a list of contracts to be traded. Supported contracts include:
            - tradingenv.contracts.Stock
            - tradingenv.contracts.ETF
            - tradingenv.contracts.Index
            - tradingenv.contracts.Future
            - tradingenv.contracts.FutureChain
            - tradingenv.contracts.AbstractContract for extra customisation.
            If you wish to allow short-selling or leveraged positions, you
            should pass one of the following instances:
            - tradingenv.spaces.DiscretePortfolio for discrete spaces
            - tradingenv.spaces.BoxPortfolio for continuous spaces
            - tradingenv.spaces.PortfolioSpace for extra customisation
            NOTE: there must be market data available for each contract being
            traded. Market data can be passed using the argument `prices` or
            `transmitter`.
        state: Union[IState, list[Feature]]
            This determines the observation space, which is empty by default.
            For backtesting purposes you don't need to specify this argument.
            For reinforcement learning training purposes, you can pass an
            tradingenv.state.IState object or a list of
            tradingenv.features.Feature.
        reward : Union[str, AbstractReward]
            The reward function to be used for reinforcement learning training.
            For backtesting purposes you don't need to specify this argument.
            Supported reward functions include:
            - tradingenv.rewards.RewardPnL
            - tradingenv.rewards.RewardLogReturn
            - tradingenv.rewards.RewardSimpleReturn
            - tradingenv.rewards.RewardDifferentialSharpeRatio
            - tradingenv.rewards.AbstractReward for extra customisation.
        transmitter : Transmitter
            This objects takes care of transmitting data and events at the right
            time during the simulations, as well as setting the timesteps
            at which the portfolio can be rebalanced. Use this argument to pass
            market prices, macroeconomic data or alternative data.
        prices : pd.DataFrame
            If your simulations only require market data, you can specify this
            argument instead of `transmitter`. Market prices can be passed as
            a pandas.DataFrame whose columns are contracts, index are
            timestamps and values are prices. However, this option assumes
            that bid-ask spread is zero. You are therefore recommended to use
            transmitter for more realistic simulations.
        initial_cash : float
            Each simulation starts with this amount of cash, 100 by default.
        broker_fees: IBrokerFees
            The simulation assumes absence of transaction costs by default. To
            set transaction costs including broker commissions, broker markup or
            a rate paid on idle cash you can pass an instance of
            tradingenv.broker.fees.BrokerFees.
        latency : float
            It defines the delay in seconds between the timestep at which the
            portfolio rebalancing decision was made and the time at which the
            trades to rebalance the portfolio are executed. This option can
            be useful when dealing with intraday data. For daily or even slower
            data you should use `steps_delay` instead.
        steps_delay : int
            Unlike `latency` which is expressed in seconds, this is expressed
            in amount of timesteps. For example, if a daily rebalancing
            frequency has been specified in the transmitter, then a rebalancing
            established in time t will be executed in t+1. This is set to
            zero by default to avoid surprised, but it is recommended to set
            this to 1 for more conservative and realistic simulations.
        fit_transformers : bool
            False by default. If True, observations defined by the `state`
            argument are using a class form sklearn.preprocessing.
        episode_length
            If not provided, the episode will terminate when the net liquidation
            value of the assets goes to zero or when market data stops. You
            can optionally provide a number of interaction with the environment
            after which the episode will terminate.
        sampling_span: int
            This argument is used only if episode_length is passed in
            TrandingEnv.reset. If specified, the episode start date is sampled
            using an exponentially decaying probability. sampling_span is the
            number of recent timesteps that captures ~70% of the likelihood of
            sampling a start date from such window. Uniform distribution is
            used by default if this parameter is not provided. It's useful to
            specify a value if you wish to train reinforcement learning agents
            that overweight observations from the more recent past.
        """
        if episode_length:
            # Adding 1 because there are (episode_length - 1) actions otherwise.
            episode_length += 1
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
        self._visits = defaultdict(int)
        self._episode_length = episode_length

        # Add extra events to transmitter and create partitions of events.
        for contract in self.action_space.contracts:
            self._transmitter.add_events(contract.make_events())
        self._transmitter._create_partitions(latency)

        # TODO: to improve UX, raise and error if there is a callback method
        #  registered to an event which does not exist anywhere in the
        #  transmitter.

        # Set by TradingEnv.reset.
        self._done: Union[bool, None] = None
        self.exchange: Union[Exchange, None] = None
        self.broker: Union[Broker, None] = None
        self._last_event: Union[IEvent, None] = None
        self._observers: Union[Sequence[Observer], None] = None
        self._now: Union[datetime, None] = None
        self._events_nonlatent: Union[List[IEvent], None] = None

        if fit_transformers:
            # Run procedure to fit transformers.
            # TODO: Test.
            kwargs = fit_transformers if isinstance(fit_transformers, dict) else dict()
            # needed to collect historical values of features
            self.backtest(**kwargs)
            for feature in self.state.features:
                feature.fit_transformer()
            # if transformers have been fit, this verification will fail.
            # Solution is to verify in parse by each feature and not here if
            # transformers have been fit.
            # By default, features before not transformation are verified anyway
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

        Returns
        -------
        Initial state of the environment.
        """
        episode_length = episode_length or self._episode_length

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
            If the action observation_space is discrete, action is an integer
            representing the action ID. If the action observation_space is
            continuous, action is a np.array with the target weights of the
            portfolio to be rebalanced as indicated in TradingEnv.action_space.
            The weight will be applied to the notional trading value of the
            contract (transaction_price * multiplier).

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
        self._visits[self.now()] += 1
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
            raise NotImplementedError("Legacy method that used to render "
                                      "results of the backtest or simulation.")
            # Note: there is no clean way for the user to stop this thread :(
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
        Run a backtest for a given policy. This method is useful for
        backtesting uses cases and not for reinforcement learning.

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

    def visits(self) -> pd.Series:
        """Returns series indicating how many steps have been taken each
        day in the environment. """
        return pd.Series(self._visits).sort_index()


class TradingEnvXY(TradingEnv):
    """High-level wrapper of TradingEnv that allows to pass features in tabular
    form (X) and prices for assets to be traded (Y) as pandas.DataFrame."""

    def __init__(self,
                 X: pd.DataFrame,
                 Y: pd.DataFrame,
                 start: datetime = None,
                 end: datetime = None,
                 transformer: str = 'yeo-johnson',
                 transformer_end: str = None,
                 reward: str = 'logret',
                 cash: float = 100,
                 spread: float = 0.0002,
                 margin: float = 0.02,
                 markup: float = 0.005,
                 fee: float = 0.0002,
                 fixed: float = 0.,
                 rate: pd.Series = None,
                 latency: float = 0,
                 steps_delay: int = 1,
                 window: int = 1,
                 stride: int = None,
                 clip: float = 5.,
                 reward_clipping: float = 2.,
                 risk_aversion: float = 0.,
                 max_long: float = 1.,
                 max_short: float = -1.,
                 calendar: str = 'NYSE',
                 folds: dict = None,
                 episode_length: int = None,
                 sampling_span: int = None
                 ):
        """
        Note: we are loosely borrowing the notation of supervised learning
        where X is the input and Y is the output. While it remains true for X,
        in this application Y is a time series of prices to be traded.

        Parameters
        ----------
        X
            A pandas.DataFrame where index is time and columns are names of the
            exogenous variables. Missing values are allowed. The observation
            space of the environment will have as many variables as the number
            of columns.
        Y
            A pandas.Dataframe where index is time and columns are names of
            the assets being traded. Value are prices. Missing values are
            allowed.
        start
            The backtest of episodes cannot start earlier than this date.
        end
            The backtest of episodes cannot end after this date.
        transformer
            A sklearn.preprocessing transformer used to map the input data in
            a suitable range of deviation. As a shortcut, you can pass 'z-score'
            to standardise the data or 'yeo-johnson' for a power transform that
            will reduce the magnitude of outliers. Default is 'yeo-johnson'.
        transformer_end
            If a transformer is passed and is not fit, then the transformer
            will be fit over all available features data unitl transformer_end.
            If a value is not passed, then the transformer is fitted between
            over all available data until `end`.
        reward
            The reward function. At present, the only supported reawrd function
            is 'logreturn'.
        cash : float
            Each simulation starts with this amount of cash, 100 by default.
        spread
            Bid-ask spread of the assets being traded. It's 0.0002 by default,
            that is 0.02%.
        margin
            This is 0.02 by default, that is 2%. If the notional value of a
            trade over the net liquidation value of the trading account is
            smaller than the margin, then the trade will not be implemented.
            This can be useful to reduce transaction costs by filtering out
            small trades.
        rate
            A pandas series with the risk free rate used to calculate yield on
            idle cash or cost of leverage.
        markup
            Broker markup. If a risk free rate is provided, idle cash in the
            trading account will yield risk_free - markup. Viceversa, if the
            account is leveraged by borrowing from the broker, the cost of
            leverage will by risk_free + markup. This is 0.005 (0.5%) by
            default.
        fee
            The broker will charge a fee proportional to the notional traded
            value. This is 0.0002 (0.02%) by default.
        fixed : float
            Fixed broker fees to be applied to each trade, regardless the
            notional traded value of trade.
        latency : float
            It defines the delay in seconds between the timestep at which the
            portfolio rebalancing decision was made and the time at which the
            trades to rebalance the portfolio are executed. This option can
            be useful when dealing with intraday data. For daily or even slower
            data you should use `steps_delay` instead.
        steps_delay : int
            Number of days that it takes to implement trades. This is 1 by
            default. Note: if zero, then trades are placed in the same instant
            that information act upon becomes available.
        window
            The shape of the observation space will be (window, nr_features).
            Window is 1 by default. The observation will return the last
            `window` observation for each feature. Useful when past observation
            carry useful information.
        stride
            When the window size is greater than 1, the stride parameter is
            used to reduce the dimensionality of the observation. It determines
            the spacing between consecutive rows selected from the most recent
            time.
        clip
            After having transformed the variables, values larger than this
            clip value will be truncated in a [-clip, +clip] range. The default
            clip value is 5.
        reward_clipping
            After scaling the reward to N(0, 1), rewards are clipped within
            the range [-reward_clipping, +reward_clipping]. The default value
            is 2.
        risk_aversion
            Negative rewards are multiplied by (1 + risk_aversion). Zero by
            default. Risk aversion is computed after clipping the reward.
            Empirically, values around 0.1 seem to work well but optimal
            values strongly depend on the use case.
        max_long
            1 (100%) by default. Long a
            llocation in a given asset larger than
            max_long will raise an error.
        max_short
            -1 (-100%) by default. Short allocation in a given asset smaller
            than max_short will raise an error.
        calendar
            Trading calendar code. Prevents the backtest or interaction with
            the environment to occur when markets are closed. 'NYSE' by default.
        folds
            Used to partition the data in multiple folds.
        episode_length
            If not provided, the episode will terminate when the net liquidation
            value of the assets goes to zero or when market data stops. You
            can optionally provide a number of interaction with the environment
            after which the episode will terminate.
        sampling_span: int
            This argument is used only if episode_length is passed in
            TrandingEnv.reset. If specified, the episode start date is sampled
            using an exponentially decaying probability. sampling_span is the
            number of recent timesteps that captures ~70% of the likelihood of
            sampling a start date from such window. Uniform distribution is
            used by default if this parameter is not provided. It's useful to
            specify a value if you wish to train reinforcement learning agents
            that overweight observations from the more recent past.

        Attributes
        ----------
        X : pandas.DataFrame
            Data derived from the input X after having been transformed.
            The environment will use this data to calculate serve observations.
        Y : pandas.DataFrame
            Data derived from the input Y. The environment allows trading
            these asset prices after applying the bid-ask spread.
        """
        if isinstance(start, str):
            start = pd.to_datetime(start)
        start = start or Y.first_valid_index()
        start = max(start, Y.first_valid_index())
        if isinstance(end, str):
            end = pd.to_datetime(end)
        end = end or Y.last_valid_index()
        end = min(end, Y.last_valid_index())
        transformer_end = transformer_end or end
        if rate is None:
            rate = pd.Series(name='Zero Rate', dtype=float)
        rate = rate.squeeze().loc[start:end]
        rate.name = Rate(rate.name)
        if not rate.between(-1, +1).all():
            raise ValueError(
                "Argument `rate` is expressed as a percentage and it "
                "shouldn't. For example, 1% should be expressed as 0.01."
            )

        # Transform data.
        # TODO: Unit test no look-ahead bias in transformer.
        # TODO: If folds are provided, fit only on the training fold.
        if isinstance(transformer, TransformerMixin):
            self.transformer = transformer
        elif transformer == 'z-score':
            self.transformer = StandardScaler()
        elif transformer == 'yeo-johnson':
            self.transformer = PowerTransformer(transformer)
        elif transformer == None:
            self.transformer = StandardScaler(with_mean=False, with_std=False)
        else:
            raise ValueError(f"Unsupported transformer: {transformer}")
        self.transformer.set_output(transform="pandas")
        try:
            check_is_fitted(self.transformer)
        except NotFittedError:
            # Fit transformer using data before start but not after end
            # to provide more data to transformer and reduce 0-padding
            # when forward-filling low frequency data.
            self.transformer.fit(X.loc[:transformer_end])

        if reward == 'logret':
            scale = np.log(pd.DataFrame(Y).loc[:transformer_end]).diff().std().mean().item()
            reward = LogReturn(scale=float(scale), clip=reward_clipping, risk_aversion=risk_aversion)
        else:
            raise NotImplementedError(f'Unsupported reward: {reward}')

        X = self.transformer.transform(X.loc[:end])
        X.ffill(inplace=True)
        X.fillna(0., inplace=True)
        X.clip(-clip, clip, inplace=True)

        # We drop features before the start date to speed-up warming-up
        # computations. Note if window is 1, then t0_warmup_idx is 0.
        t0 = X.loc[start:end].first_valid_index()
        t0_idx = X.index.get_loc(t0)
        t0_warmup_idx = t0_idx - window + 1
        assert t0_warmup_idx >= 0
        X = X.iloc[t0_warmup_idx:]

        # Discard data outside the data range and set contracts.
        Y = pd.DataFrame(Y).loc[start:end]
        Y.columns = [Asset(col) for col in Y.columns]
        assert X.first_valid_index() <= Y.first_valid_index()
        super().__init__(
            action_space=BoxPortfolio(Y.columns, max_short, max_long, margin=margin),
            state=State(X.columns.size, window, stride, max_=5),
            reward=reward,
            transmitter=self._make_transmitter(X, Y, calendar, spread, rate, folds, window),
            broker_fees=BrokerFees(markup, rate.name, fee, fixed),
            initial_cash=cash,
            latency=latency,
            steps_delay=steps_delay,
            episode_length=episode_length,
            sampling_span=sampling_span,
        )

        # Set attributes.
        self.X = X
        self.Y = Y
        self.start = min(self._transmitter.timesteps)
        self.end = max(self._transmitter.timesteps)

    @staticmethod
    def _make_timesteps(X: pd.DataFrame, Y: pd.DataFrame, calendar: str):
        # Drop dates where market is closed.
        calendar = pandas_market_calendars.get_calendar(calendar)
        holidays = calendar.holidays().holidays
        start = max(X.first_valid_index(), Y.first_valid_index())
        end = min(X.last_valid_index(), Y.last_valid_index())
        Y = Y.loc[start:end]
        timesteps = Y.drop([t for t in holidays if t in Y.index]).index
        return timesteps

    def _make_transmitter(self, X: pd.DataFrame, Y: pd.DataFrame, calendar: str, spread: float = 0., rate = None, folds = None, window = None):
        # TODO: test market calendar is applied.
        # TODO: test no past events are not processed - no need to spend computation in warming up if not needee.
        markov_reset = window == 1
        warmup = None if markov_reset else timedelta(days=3 + window * 2)
        timesteps = self._make_timesteps(X, Y, calendar)
        transmitter = Transmitter(timesteps, folds, markov_reset, warmup)
        for name in Y.columns:
            transmitter.add_prices(Y[[name]], spread)
        events = [EventNewObservation(t, x) for t, x in X.iterrows()]
        transmitter.add_events(events)
        transmitter.add_prices(rate.to_frame())
        return transmitter
