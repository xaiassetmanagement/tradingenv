class TestReadme:
    def test_instancing_env_detailed(self):
        from tradingenv import TradingEnv
        from tradingenv.contracts import ETF
        from tradingenv.spaces import BoxPortfolio
        from tradingenv.state import IState
        from tradingenv.rewards import RewardLogReturn
        from tradingenv.broker.fees import BrokerFees
        import yfinance

        # Load close prices from Yahoo Finance and specify contract types.
        prices = yfinance.Tickers(['SPY', 'TLT', 'TBIL']).history(period="12mo", progress=False)['Close'].tz_localize(None)
        prices.columns = [ETF('SPY'), ETF('TLT'), ETF('TBIL')]

        # Instance the trading environment.
        env = TradingEnv(
            action_space=BoxPortfolio([ETF('SPY'), ETF('TLT')], low=-1, high=+1, as_weights=True),
            state=IState(),
            reward=RewardLogReturn(),
            prices=prices,
            initial_cash=1_000_000,
            latency=0,  # seconds
            steps_delay=1,  # trades are implemented with a delay on one step
            broker_fees=BrokerFees(
                markup=0.005,  # 0.5% broker markup on deposit rate
                proportional=0.0001,  # 0.01% fee of traded notional
                fixed=1,  # $1 per trade
            ),
        )

        # OpenAI/gym protocol. Run an episode in the environment.
        # env can be passed to RL agents of ray/rllib or stable-baselines3.
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
        return env, prices

    def test_backtest_60_40_old(self):
        env, prices = self.test_instancing_env_detailed()
        # BEGIN OMIT
        from tradingenv.policy import AbstractPolicy

        class Portfolio6040(AbstractPolicy):
            """Implement logic of your investment strategy or RL agent here."""

            def act(self, state):
                """Invest 60% of the portfolio in SPY ETF and 40% in TLT ETF."""
                return [0.6, 0.4]

        # Run the backtest.
        track_record = env.backtest(
            policy=Portfolio6040(),
            risk_free=prices['TBIL'],
            benchmark=prices['SPY'],
        )

        # The track_record object stores the results of your backtest.
        track_record.tearsheet()
        track_record.fig_net_liquidation_value()
        # END OMIT

    def test_instancing_env_lazy(self):
        # BEGIN OMIT
        from tradingenv.env import TradingEnvXY
        import yfinance

        # Load data from Yahoo Finance.
        tickers = yfinance.Tickers(['SPY', 'TLT', 'TBIL', '^IRX'])
        data = tickers.history(period="12mo", progress=False)['Close'].tz_localize(None)
        Y = data[['SPY', 'TLT']]
        X = Y.rolling(12).mean() - Y.rolling(26).mean()

        # Default instance of the trading environment.
        env = TradingEnvXY(X, Y)

        # Run an episode in the environment.
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
        # END OMIT
        return env, data, X, Y

    def test_instancing_env_custom(self):
        from tradingenv.env import TradingEnvXY
        env, data, X, Y = self.test_instancing_env_lazy()

        # BEGIN OMIT
        env = TradingEnvXY(
            X=X,                      # Use moving averages crossover as features
            Y=Y,                      # to trade SPY and TLT ETFs.
            transformer='z-score',    # Features are standardised to N(0, 1).
            reward='logret',          # Reward is the log return of the portfolio at each step,
            cash=1000000,             # starting with $1M.
            spread=0.0002,            # Transaction costs include a 0.02% spread,
            markup=0.005,             # a 0.5% broker markup on deposit rate,
            fee=0.0002,               # a 0.02% dealing fee of traded notional
            fixed=1,                  # and a $1 fixed fee per trade.
            margin=0.02,              # Do not trade if trade size is smaller than 2% of the portfolio.
            rate=data['^IRX'] / 100,  # Rate used to compute the yield on idle cash and cost of leverage.
            latency=0,                # Trades are implemented with no latency
            steps_delay=1,            # but a delay of one day.
            window=1,                 # The observation is the current state of the market,
            clip=5.,                  # clipped between -5 and +5 standard deviations.
            max_long=1.5,             # The maximum long position is 150% of the portfolio,
            max_short=-1.,            # the maximum short position is 100% of the portfolio.
            calendar='NYSE',          # Use the NYSE calendar to schedule trading days.
        )
        # END OMIT
        return env, data

    def test_backtest_60_40(self):
        env, data, X, Y = self.test_instancing_env_lazy()

        # BEGIN OMIT
        from tradingenv.policy import AbstractPolicy

        class Portfolio6040(AbstractPolicy):
            """Implement logic of your investment strategy or RL agent here."""

            def act(self, state):
                """Invest 60% of the portfolio in SPY ETF and 40% in TLT ETF."""
                return [0.6, 0.4]

        # Run the backtest.
        track_record = env.backtest(
            policy=Portfolio6040(),
            risk_free=data['TBIL'],
            benchmark=data['SPY'],
        )

        # The track_record object stores the results of your backtest.
        track_record.tearsheet()
        # END OMIT