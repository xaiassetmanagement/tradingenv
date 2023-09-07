import tradingenv
from tradingenv.env import TradingEnv
from tradingenv.spaces import DiscretePortfolio, BoxPortfolio
from tradingenv.transmitter import Transmitter
from tradingenv.state import IState
from tradingenv.events import EventReset, EventStep
from tradingenv.contracts import Cash, ETF, ES, FutureChain, Index, AbstractContract
from tradingenv.features import Feature
from tradingenv.events import EventNBBO
from pandas.api.types import is_numeric_dtype
from datetime import datetime
from time import strptime
import gym.spaces
import numpy as np
import pandas as pd
import os

here = os.path.abspath(os.path.dirname(__file__))
excel_etf = pd.ExcelFile(os.path.join(here, "data", "etf.xlsx"))
excel_futures = pd.ExcelFile(os.path.join(here, "data", "futures_prices.xlsx"))
excel_futures_solutions = pd.ExcelFile(
    os.path.join(here, "data", "futures_solutions.xlsx")
)
excel_btc_cny = pd.read_csv(
    os.path.join(here, "data", "high_freq_btc_cny.csv"),
    parse_dates=True,
    index_col='Time',
)
excel_btc_cny.rename({'Price': Index('BTC/CNY')}, axis='columns', inplace=True)


class TargetDiscretePortfolio(IState):
    """Feature whose state indicate the current target weights of the
    portfolio."""
    def __init__(self):
        self.space = gym.spaces.Box(low=0, high=1, shape=(3,), dtype=np.float64)
        self.value = [0, 0, 0]

    def process_EventReset(self, event: "EventReset"):
        self.value = [1, 0, 0]

    def process_EventStep(self, event: "EventStep"):
        rebalancing = event.track_record[-1]
        weight1 = 0
        weight2 = rebalancing.allocation.get('S&P 500', 0)
        weight3 = rebalancing.allocation.get('T-bills', 0)
        self.value = [weight1, weight2, weight3]

    def parse(self):
        return np.array(self.value)


class TestSimpleExamples:
    def test_minimal_example(self):
        # Sample prices parsed in AbstractTransmitter.
        prices = pd.DataFrame(
            data={
                ETF("S&P 500"): [100, 101, 103, 106],
                ETF("T-bills"): [1000, 1000.5, 1001, 1002],
            },
            index=[
                datetime(2019, 1, 1, 16),  # 4pm, close transaction_price
                datetime(2019, 1, 2, 16),  # 4pm, close transaction_price
                datetime(2019, 1, 3, 16),  # 4pm, close transaction_price
                datetime(2019, 1, 4, 16),  # 4pm, close transaction_price
            ],
        )
        transmitter = Transmitter(timesteps=prices.index)
        transmitter.add_prices(prices)

        # Instance the environment.
        action_space = DiscretePortfolio(
            contracts=[Cash(), ETF("S&P 500"), ETF("T-bills")],
            allocations=[[0, 1, 0], [0, 0.5, 0.5], [0, 0, 1]],
        )
        env = TradingEnv(
            action_space=action_space,
            state=TargetDiscretePortfolio(),
            reward="RewardSimpleReturn",
            transmitter=transmitter,
        )

        # Reset the environment.
        state = env.reset()
        np.testing.assert_equal(
            # Initial portfolio is 100% in cash.
            actual=state,
            desired=np.array([1.0, 0.0, 0.0]),
        )
        assert env._now == datetime(2019, 1, 1, 16)
        assert env.exchange[ETF("S&P 500")].mid_price == 100
        assert env.exchange[ETF("T-bills")].mid_price == 1000

        # Interact with the environment.
        state, reward, done, info = env.step(action=0)
        assert env._now == datetime(2019, 1, 2, 16)
        assert env.exchange[ETF("S&P 500")].mid_price == 101
        assert env.exchange[ETF("T-bills")].mid_price == 1000.5
        np.testing.assert_equal(
            actual=state,
            desired=np.array([0.0, 1.0, 0.0])
        )
        np.testing.assert_almost_equal(actual=reward, desired=101 / 100 - 1)
        assert not done

        # Interact with the environment.
        state, reward, done, info = env.step(action=1)
        assert env._now == datetime(2019, 1, 3, 16)
        assert env.exchange[ETF("S&P 500")].mid_price == 103
        assert env.exchange[ETF("T-bills")].mid_price == 1001
        np.testing.assert_equal(
            actual=state,
            desired=np.array([0.0, 0.5, 0.5])
        )
        np.testing.assert_almost_equal(
            actual=reward, desired=0.5 * (103 / 101 - 1) + 0.5 * (1001 / 1000.5 - 1)
        )
        assert not done

        # Interact with the environment.
        state, reward, done, info = env.step(action=2)
        assert env._now == datetime(2019, 1, 4, 16)
        assert env.exchange[ETF("S&P 500")].mid_price == 106
        assert env.exchange[ETF("T-bills")].mid_price == 1002
        np.testing.assert_equal(
            actual=state,
            desired=np.array([0.0, 0.0, 1])
        )
        np.testing.assert_almost_equal(actual=reward, desired=1002 / 1001 - 1)
        assert done

        # Verify track record.
        assert env.broker.track_record._time == [
            datetime(2019, 1, 1, 16),  # 4pm, close transaction_price
            datetime(2019, 1, 2, 16),  # 4pm, close transaction_price
            datetime(2019, 1, 3, 16),  # 4pm, close transaction_price
        ]
        nlvs_actual = list(env.broker.track_record.net_liquidation_value(before_rebalancing=False).squeeze())
        assert nlvs_actual == [100.0, 101.0, 102.02523738130934]


class TestETF:
    def get_env(self, cost_of_spread: float = 0, margin: float=0):
        prices = excel_etf.parse("Prices", index_col="Date", parse_dates=True)
        prices.columns = [ETF(str(symbol)) for symbol in prices.columns]
        transmitter = Transmitter(timesteps=prices.index)
        transmitter.add_prices(prices, spread=cost_of_spread)
        action_space = BoxPortfolio([Cash(), ETF("ETF:SPY"), ETF("ETF:IEF")], margin=margin)
        return TradingEnv(action_space=action_space, transmitter=transmitter)

    def assert_is_equal_nlv(self, actual, expected):
        assert isinstance(actual, pd.DataFrame)
        assert is_numeric_dtype(actual.squeeze())
        assert actual.index.equals(expected.index)
        assert np.isclose(actual.values, expected.values).all()

    def test_cash_only(self):
        env = self.get_env()

        # Interact with the environment.
        env.reset()
        done = False
        while not done:
            action = np.array([1.0, 0, 0])
            obs, reward, done, info = env.step(action)

        # Assert equivalence.
        actual = env.broker.track_record.net_liquidation_value(before_rebalancing=False)
        expected = excel_etf.parse("TestCashOnly", index_col="Date", parse_dates=True)[
            ["NLV"]
        ]
        # Drop duplicate time index.
        expected = expected.iloc[1:, :]
        self.assert_is_equal_nlv(actual, expected)

    def test_balanced_portfolio_rebalanced_daily(self):
        env = self.get_env()

        # Interact with the environment.
        env.reset()
        done = False
        while not done:
            action = np.array([0, 0.6, 0.4])
            obs, reward, done, info = env.step(action)

        # Assert equivalence.
        actual = env.broker.track_record.net_liquidation_value(before_rebalancing=False)
        expected = excel_etf.parse(
            sheet_name="TestDailyRebalancing6040", index_col="Date", parse_dates=True
        )
        # Drop duplicate time index.
        expected = expected.iloc[1:, :]
        self.assert_is_equal_nlv(actual, expected[["NLV"]])

    def test_buy_and_hold(self):
        env = self.get_env()

        # Interact with the environment.
        env.reset()
        done = False
        action = np.array([0, 0.6, 0.4])
        obs, reward, done, info = env.step(action)
        env.action_space._margin = np.inf
        while not done:
            obs, reward, done, info = env.step(action)

        # Assert equivalence.
        actual = env.broker.track_record.net_liquidation_value(before_rebalancing=False)
        expected = excel_etf.parse(
            sheet_name="TestBuyAndHold6040", index_col="Date", parse_dates=True
        )
        # Drop duplicate time index.
        expected = expected.iloc[1:, :]
        self.assert_is_equal_nlv(actual, expected[["NLV"]])

    def test_reproducible_simulation_after_reset(self):
        env = self.get_env()

        # Interact with the environment: first simulation.
        env.reset()
        done = False
        while not done:
            action = np.array([0, 0.6, 0.4])
            obs, reward, done, info = env.step(action)
        track_record_1 = env.broker.track_record

        # Interact with the environment: second simulation.
        env.reset()
        done = False
        while not done:
            action = np.array([0, 0.6, 0.4])
            obs, reward, done, info = env.step(action)
        track_record_2 = env.broker.track_record

        # Assert equivalence.
        nlv1 = track_record_1.net_liquidation_value(before_rebalancing=False)
        nlv2 = track_record_2.net_liquidation_value(before_rebalancing=False)
        assert nlv1.equals(nlv2)


class TestFutures:
    def test_example_runs_without_raising_exceptions(self):
        def mapper(name):
            year = int(name[27:31])
            month = strptime(name[23:26], "%b").tm_mon
            return ES(year, month)

        # Instance the env.
        prices = excel_futures.parse(
            sheet_name="ES", index_col="Date", parse_dates=True
        )
        prices.rename(mapper=mapper, axis="columns", inplace=True)
        transmitter = Transmitter(
            timesteps=prices.index, folds={"training-set": [datetime.min, datetime.max]}
        )
        transmitter.add_prices(prices)
        leading_future = FutureChain(ES, "1997-12", "2019-10")
        env = TradingEnv(action_space=[Cash(), leading_future], transmitter=transmitter)
        expected = 1 - ES(2019, 6).margin_requirement
        err = 1e-7

        # Interact with the env.
        env.reset()
        done = False
        while not done:
            action = np.array([0, 1])
            state, reward, done, info = env.step(action)

            # Margin requirement is 0.1 of notional ES value, so there should
            # be a constant 90% of cash which will generate a positive interest
            # if FED funds rate > 0.
            holdings_weights = env.broker.holdings_weights()
            cash_weight = holdings_weights[env.broker.base_currency]
            assert expected - err < cash_weight < expected + err

        # Asset.
        nlv_actual = env.broker.track_record.net_liquidation_value(before_rebalancing=False)

        # Reference of expected: S&P Futures Return in chart page 6.
        # https://www.cmegroup.com/education/files/deconstructing-futures-returns-the-role-of-roll-yield.pdf
        nlv_expected = excel_futures_solutions.parse(
            sheet_name="BuyAndHoldLeadingES", index_col="Date", parse_dates=True
        )
        nlv_expected.rename({'nlv': 'Net liquidation value'}, axis='columns', inplace=True)
        # Drop duplicate initial index.
        pd.testing.assert_frame_equal(nlv_actual, nlv_expected)


class TestHighFrequencyData:
    def test_example(self):
        env = TradingEnv(
            action_space=[Cash(), Index('BTC/CNY')],
            prices=excel_btc_cny,
        )

        env.reset()
        done = False
        while not done:
            action = np.array([0, 1])
            state, reward, done, info = env.step(action)

        nlv = env.broker.track_record.net_liquidation_value(before_rebalancing=False)
        # Discard last price from excel_btc_cny because we don't take action
        # at the last price, because there
        assert nlv.index.equals(excel_btc_cny.index[:-1])
        assert nlv.nr_observations() == len(excel_btc_cny) - 1
        np.testing.assert_approx_equal(
            actual=float((nlv.iloc[-1] / nlv.iloc[0]).item()),
            # isn't a next one to evaluate the action.
            desired=float((excel_btc_cny.iloc[-2] / excel_btc_cny.iloc[0]).item()),
        )
        nlv.tearsheet()  # does not raise


class SpotPrice(Feature):
    def __init__(self, fit_transformer: bool = False):
        super().__init__()
        self.data = dict()

    def process_EventNBBO(self, event: EventNBBO):
        if event.contract in [ETF("ETF:SPY"), ETF("ETF:IEF")]:
            self.data[event.contract] = event.mid_price

    def parse(self, state = None):
        return self.data


class TestStateAndFeatureHistory:
    """In this class we test IState and AbstractFeature.history when a list
    of features is passed to the state."""
    def test_features_and_state_history(self):
        # Note: this could be moved to a faster integration test, but kept here
        # to show an example of using state and feature history.
        # excel_etf = pd.ExcelFile(os.path.join("/home/federico/Desktop/repos/trading-gym/tests/regression/data", "etf.xlsx"))
        feature = SpotPrice()
        prices = excel_etf.parse("Prices", index_col="Date", parse_dates=True)
        prices.columns = [ETF(str(symbol)) for symbol in prices.columns]
        transmitter = Transmitter(timesteps=prices.index)
        transmitter.add_prices(prices)
        action_space = BoxPortfolio([Cash(), ETF("ETF:SPY"), ETF("ETF:IEF")])
        env = TradingEnv(
            action_space=action_space,
            transmitter=transmitter,
            state=IState([feature])
        )
        env.backtest()

        # Features' history.
        expected = prices.T.to_dict()
        actual = feature.history
        assert actual == expected

        # State history.
        expected = {k: {'SpotPrice': v} for k, v in prices.T.to_dict().items()}
        actual = env.state.history
        assert actual == expected

        # tear down, otherwise other tests relying on this global attr may fail
        # (yes, this logic needs to be improved but not trivial).
        AbstractContract.now = datetime.min

    def test_features_history_when_episode_length_is_provided(self):
        # Note: this could be moved to a faster integration test.
        feature = SpotPrice()
        prices = excel_etf.parse("Prices", index_col="Date", parse_dates=True)
        prices.columns = [ETF(str(symbol)) for symbol in prices.columns]
        transmitter = Transmitter(timesteps=prices.index)
        transmitter.add_prices(prices)
        action_space = BoxPortfolio([Cash(), ETF("ETF:SPY"), ETF("ETF:IEF")])
        env = TradingEnv(
            action_space=action_space,
            transmitter=transmitter,
            state=IState([feature])
        )
        env.backtest(episode_length=10)

        # Features' history.
        expected = prices.loc[:env.now()].T.to_dict()
        actual = feature.history
        assert actual == expected

        # State history.
        expected = {k: {'SpotPrice': v} for k, v in prices.loc[env._transmitter._start_date:env.now()].T.to_dict().items()}
        actual = env.state.history
        assert actual == expected

        # tear down, otherwise other tests relying on this global attr may fail
        # (yes, this logic needs to be improved but not trivial).
        AbstractContract.now = datetime.min
