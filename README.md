![Logo](https://tradingenv.blob.core.windows.net/images/logo-background-cropped.png)

[![Documentation](https://github.com/xaiassetmanagement/tradingenv/actions/workflows/build-docs.yml/badge.svg)](https://github.com/xaiassetmanagement/tradingenv/actions/workflows/software-tests.yml)
[![Software tests](https://github.com/xaiassetmanagement/tradingenv/actions/workflows/software-tests.yml/badge.svg)](https://github.com/xaiassetmanagement/tradingenv/actions/workflows/software-tests.yml)
[![Coverage](https://raw.githubusercontent.com/xaiassetmanagement/tradingenv/coverage-badge/coverage.svg)](https://github.com/xaiassetmanagement/tradingenv/actions)

[![python](https://img.shields.io/pypi/pyversions/shap)](https://www.python.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


# Introduction

Backtest trading strategies or train reinforcement learning agents with
`tradingenv`, an event-driven market simulator that implements the
OpenAI/gym protocol.

# Installation

tradingenv supports Python 3.9 or newer versions. The following command
line will install the latest software version.

``` console
pip install tradingenv
```

Notebooks, software tests and building the documentation require extra
dependencies that can be installed with

``` console
pip install tradingenv[extra]
```

# Examples

## Reinforcement Learning

The package is built upon the industry-standard
[gym](https://github.com/openai/gym) and therefore can be used in
conjunction with popular reinforcement learning frameworks including
[rllib](https://docs.ray.io/en/latest/rllib/), 
[stable-baselines3](https://github.com/hill-a/stable-baselines) and [ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL).

``` python
from tradingenv.env import TradingEnvXY
import yfinance

# Load data from Yahoo Finance.
tickers = yfinance.Tickers(['SPY', 'TLT', 'TBIL', '^IRX'])
data = tickers.history(period="12mo", progress=False)['Close'].tz_localize(None)
Y = data[['SPY', 'TLT']]
X = Y.rolling(12).mean() - Y.rolling(26).mean()

# Instance the trading environment.
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

# OpenAI/gym protocol. Run an episode in the environment.
obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
```

## Backtesting

Thanks to the event-driven design, tradingenv is agnostic with respect
to the type and time-frequency of the events. This means that you can
run simulations either using irregularly sampled trade and quotes data,
daily closing prices, monthly economic data or alternative data.
Financial instruments supported include stocks, ETF and futures.

``` python
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
```

![](https://tradingenv.blob.core.windows.net/images/tearsheet.png)

``` python
track_record.fig_net_liquidation_value()
```

![](https://tradingenv.blob.core.windows.net/images/fig_net_liquidation_value.png)

# Relevant projects

-   [btgym](https://github.com/Kismuz/btgym): is an OpenAI
    Gym-compatible environment for
-   [backtrader](https://github.com/backtrader/backtrader)
    backtesting/trading library, designed to provide gym-integrated
    framework for running reinforcement learning experiments in \[close
    to\] real world algorithmic trading environments.
-   [gym](https://github.com/openai/gym): A toolkit for developing and
    comparing reinforcement learning algorithms.
-   [qlib](https://github.com/microsoft/qlib): Qlib provides a strong
    infrastructure to support quant research.
-   [rllib](https://docs.ray.io/en/latest/rllib/): open-source library
    for reinforcement learning.
-   [stable-baselines3](https://github.com/hill-a/stable-baselines): is
    a set of reliable implementations of reinforcement learning
    algorithms in PyTorch.

# Developers

You are welcome to contribute features, examples and documentation or
issues.

You can run the software tests typing `pytest` in the command line,
assuming that the folder `\tests` is in the current working directory.

To refresh and build the documentation:

``` 
pytest tests/notebooks
sphinx-apidoc -f -o docs/source tradingenv
cd docs
make clean
make html
```

<!---
On README.md vs README.rst.
While .rst is clearly a better format for Sphinx documentation, not all 
blocks are rendered correctly on GitHub. The rst block literalinclude in 
particular is important, as it's the only way to have code in the /tests folder 
and extract it from functions and class methods. While GitHub supports .rst,
it does not render with Sphinx and therefore .rst is overpowered for GitHub as 
of 2024-09-26 (or at least, I haven't found a solution in hours of research).
Therefore, I switch to .md as it's rendered correctly.
-->