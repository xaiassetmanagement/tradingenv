"""Good things to test in new features are:
- .__init__
- .parse
- .fit_transform
"""
import gym.spaces

from tradingenv.events import EventNBBO
from tradingenv.library import FeaturePortfolioWeight, FeaturePrices, FeatureSpread, FeatureIsRTH
from tradingenv.exchange import Exchange
from tradingenv.contracts import ETF
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import datetime
import pytest
import numpy as np
import pandas as pd


class TestPortfolioWeight:
    def test_init_single_contract(self):
        contract = ETF('SPY')
        feature = FeaturePortfolioWeight(contract, 0., 1.)
        assert feature.contracts == [contract]
        assert feature.space.low == 0.
        assert feature.space.high == 1.
        assert feature.space.shape == (1, 1)
        assert isinstance(feature.transformer, MinMaxScaler)
        assert feature.transformer.feature_range == (-1, 1.)

    def test_init_multiple_contracts(self):
        contracts = [ETF('SPY'), ETF('IEF')]
        feature = FeaturePortfolioWeight(contracts, 0., 1.)
        assert feature.contracts == contracts
        assert feature.space.shape == (1, 2)

    def test_init_custom(self):
        feature = FeaturePortfolioWeight(
            ETF('SPY'), 0., 1., 'NewName', MinMaxScaler((-0.5, 0.5), clip=True)
        )
        assert feature.transformer.feature_range == (-0.5, 0.5)
        assert feature.name == 'NewName'
        assert feature.transformer.clip is True

    @pytest.mark.parametrize('x, expected', [
        (0., -1.),
        (0.25, -0.5),
        (0.5, 0.),
        (0.75, 0.5),
        (1., 1.),
    ])
    def test_fit_transformer_one_contract(self, x, expected):
        """x and expected must be in the shape of feature.space.shape, so using
        nested list to respect that."""
        x = [[x]]
        expected = np.array([[expected]])
        feature = FeaturePortfolioWeight(ETF('SPY'), 0., 1., (-1, 1.))
        feature.fit_transformer()
        actual = feature.transformer.transform(x)
        np.testing.assert_equal(actual, expected)

    @pytest.mark.parametrize('x, expected', [
        [(0., 0.25), (-1, -0.5)],
        [(0.25, 0.5), (-0.5, 0.)],
        [(0.5, 0.75), (0., 0.5)],
        [(0.75, 1.), (0.5, 1.)],
    ])
    def test_fit_transformer_multiple_contracts(self, x, expected):
        """x and expected must be in the shape of feature.space.shape, so using
        nested list to respect that."""
        x = [x]
        expected = np.array([expected])
        feature = FeaturePortfolioWeight([ETF('SPY'), ETF('IEF')], 0., 1., (-1, 1.))
        feature.fit_transformer()
        actual = feature.transformer.transform(x)
        np.testing.assert_equal(actual, expected)

    def test_call_transforms_observation_if_transformer_has_been_fit(self):
        """Nice to have this test but it's redundant as __call__ is not
        implemented in FeaturePortfolioWeight and it has already been tested in
        its parent class Feature.
        """

        class MockBroker:
            def holdings_weights(self):
                return {ETF('SPY'): 0.6, ETF('IEF'): 0.4}

        broker = MockBroker()
        contract = ETF('SPY')
        feature = FeaturePortfolioWeight(contract, 0., 1., (-1, 1.))
        feature.reset(broker=broker)
        feature.fit_transformer()

        actual = feature.parse()
        expected = np.array([[0.6]])
        np.testing.assert_almost_equal(actual, expected)

        actual = feature()
        expected = np.array([[0.2]])
        np.testing.assert_almost_equal(actual, expected)


class TestFeaturePrices:
    def test_init_single_contract(self):
        contract = ETF('SPY')
        feature = FeaturePrices(contract)
        assert feature.contracts == [contract]
        assert feature.space.low == 0.
        assert feature.space.high == np.inf
        assert feature.space.shape == (1, 1)
        assert isinstance(feature.transformer, StandardScaler)

    def test_init_multiple_contracts(self):
        contracts = [ETF('SPY'), ETF('IEF')]
        feature = FeaturePrices(contracts)
        assert feature.contracts == contracts
        assert feature.space.shape == (1, 2)

    def test_parse(self):
        contract = ETF('SPY')
        feature = FeaturePrices(contract)
        exchange = Exchange()
        feature.reset(exchange=exchange)
        returns = pd.Series(
            data=np.random.normal(0.05, 0.12, 10),
            index=pd.date_range('2022-01-01', freq='B', periods=10),
        )
        spread = 0.
        prices = (1 + returns).cumprod()
        for t, p in prices.items():
            nbbo = EventNBBO(t, contract, p - spread, p + spread)
            exchange.process_EventNBBO(nbbo)
            actual = feature()
            expected = np.array([[p]])
            np.testing.assert_almost_equal(actual, expected)

    def test_transform_one_contract(self):
        contract = ETF('SPY')
        feature = FeaturePrices(contract)
        exchange = Exchange()
        feature.reset(exchange=exchange)
        returns = pd.Series(
            data=np.random.normal(0.05, 0.12, 10),
            index=pd.date_range('2022-01-01', freq='B', periods=10),
        )
        spread = 0.
        prices = (1 + returns).cumprod()

        # Full pass to create a history to be used to fit the transformer.
        for t, p in prices.items():
            nbbo = EventNBBO(t, contract, p - spread, p + spread)
            exchange.process_EventNBBO(nbbo)
            actual = feature(verify=True)

        # Fit the transformer.
        feature.fit_transformer()

        # Assessment.
        for t, p in prices.items():
            nbbo = EventNBBO(t, contract, p - spread, p + spread)
            exchange.process_EventNBBO(nbbo)
            actual = feature(verify=True)
            expected = (p - prices.mean()) / prices.std(ddof=0)
            np.testing.assert_almost_equal(actual, expected)

    def test_transform_multiple_contracts(self):
        contracts = [ETF('SPY'), ETF('IEF')]
        feature = FeaturePrices(contracts)
        exchange = Exchange()
        feature.reset(exchange=exchange)
        returns = pd.DataFrame(
            data=np.random.normal(0.05, 0.12, (10, 2)),
            index=pd.date_range('2022-01-01', freq='B', periods=10),
            columns=contracts,
        )
        spread = 0.
        prices = (1 + returns).cumprod()

        # Full pass to create a history to be used to fit the transformer.
        for t, p_series in prices.iterrows():
            for c, p in p_series.items():
                nbbo = EventNBBO(t, c, p - spread, p + spread)
                exchange.process_EventNBBO(nbbo)
            actual = feature(verify=True)

        # Fit the transformer.
        feature.fit_transformer()

        # Assessment.
        feature.reset(exchange=exchange)
        for t, p_series in prices.iterrows():
            for c, p in p_series.items():
                nbbo = EventNBBO(t, c, p - spread, p + spread)
                exchange.process_EventNBBO(nbbo)
            actual = feature(verify=True)
            expected = np.array([(p_series - prices.mean()) / prices.std(ddof=0)])
            np.testing.assert_almost_equal(actual, expected)


class TestFeatureSpread:
    def test_parse_one_contract(self):
        exchange = Exchange()
        contract = ETF('VXX')
        feature = FeatureSpread(contract)
        feature.reset(exchange=exchange)

        nbbo = EventNBBO(datetime(2022, 1, 1), contract, 15.05, 15.15)
        exchange.process_EventNBBO(nbbo)

        actual = feature.parse()
        expected = - np.array([[15.05 / 15.15 - 1]])
        np.testing.assert_almost_equal(actual, expected)

    def test_parse_multiple_contracts(self):
        exchange = Exchange()
        contracts = [ETF('SPY'), ETF('IEF')]
        feature = FeatureSpread(contracts)
        feature.reset(exchange=exchange)

        nbbo = EventNBBO(datetime(2022, 1, 1), ETF('SPY'), 439.11, 439.17)
        exchange.process_EventNBBO(nbbo)
        nbbo = EventNBBO(datetime(2022, 1, 1), ETF('IEF'), 109.13, 109.15)
        exchange.process_EventNBBO(nbbo)

        actual = feature.parse()
        expected = - np.array([[439.11 / 439.17 - 1, 109.13 / 109.15 - 1]])
        np.testing.assert_almost_equal(actual, expected)

    def test_parse_clip(self):
        exchange = Exchange()
        contract = ETF('VXX')
        feature = FeatureSpread(contract, clip=0.001)
        feature.reset(exchange=exchange)

        nbbo = EventNBBO(datetime(2022, 1, 1), contract, 15.05, 15.15)
        exchange.process_EventNBBO(nbbo)

        actual = feature.parse()
        expected = np.array([[0.001]])
        np.testing.assert_almost_equal(actual, expected)

    def test_custom_weights(self):
        class FeatureCustom(FeatureSpread):
            def _make_weights(self):
                return np.array([[0.25, 0.75]])

        exchange = Exchange()
        contracts = [ETF('SPY'), ETF('IEF')]
        feature = FeatureCustom(contracts)
        feature.reset(exchange=exchange)

        nbbo = EventNBBO(datetime(2022, 1, 1), ETF('SPY'), 439.11, 439.17)
        exchange.process_EventNBBO(nbbo)
        nbbo = EventNBBO(datetime(2022, 1, 1), ETF('IEF'), 109.13, 109.15)
        exchange.process_EventNBBO(nbbo)

        actual = feature.parse()
        expected = - np.array([[
            0.25 * (439.11 / 439.17 - 1) + 0.75 * (109.13 / 109.15 - 1)
        ]])
        np.testing.assert_almost_equal(actual, expected)

    @pytest.mark.parametrize('bid,ask,expected', [
        (15.05, 15.15, 1.),
        (15.05, 15.05, -1.),
        (15.05, 15.05 / (1 + 0.001 * 0.5), 0.),
        (15.05, 15.05 / (1 + 0.001 * 0.25), -0.5),
        (15.05, 15.05 / (1 + 0.001 * 0.75), 0.5),
    ])
    def test_fit_transform(self, bid, ask, expected):
        exchange = Exchange()
        contract = ETF('VXX')
        feature = FeatureSpread(contract, clip=0.001)
        feature.reset(exchange=exchange)
        feature.fit_transformer()

        nbbo = EventNBBO(datetime(2022, 1, 1), contract, bid, ask)
        exchange.process_EventNBBO(nbbo)
        actual = feature()
        expected = np.array([[expected]])
        np.testing.assert_almost_equal(actual, expected)


class TestFeatureIsRTH:
    def test_init(self):
        feature = FeatureIsRTH()
        assert isinstance(feature.space, gym.spaces.MultiBinary)

    @pytest.mark.parametrize('t,expected', [
        (datetime(2022, 1, 1, 13), 0),  # Saturday
        (datetime(2022, 1, 2, 13), 0),  # Sunday
        (datetime(2022, 1, 5, 8, 29), 0),
        (datetime(2022, 1, 5, 8, 31), 1),
        (datetime(2022, 1, 5, 9), 1),
        (datetime(2022, 1, 5, 14, 29), 1),
        (datetime(2022, 1, 5, 15, 16), 0),
    ])
    def test_parse(self, t, expected):
        expected = np.array([expected])
        exchange = Exchange()
        feature = FeatureIsRTH()
        feature.reset(exchange=exchange)
        nbbo = EventNBBO(pd.Timestamp(t), ETF('SPY'), 439.11, 439.17)
        exchange.process_EventNBBO(nbbo)
        actual = feature.parse()
        np.testing.assert_almost_equal(actual, expected)

    def test_fit_transform(self):
        exchange = Exchange()
        feature = FeatureIsRTH()
        feature.reset(exchange=exchange)
        feature.fit_transformer()
        t = datetime(2022, 1, 5, 9)
        nbbo = EventNBBO(pd.Timestamp(t), ETF('SPY'), 439.11, 439.17)
        exchange.process_EventNBBO(nbbo)

    @pytest.mark.parametrize('t,expected', [
        (datetime(2022, 1, 1, 13), 0.),  # Saturday
        (datetime(2022, 1, 2, 13), 0.),  # Sunday
        (datetime(2022, 1, 5, 8, 29), 0.),
        (datetime(2022, 1, 5, 8, 31), 0.00246914),
        (datetime(2022, 1, 5, 9), 0.0740741),
        (datetime(2022, 1, 5, 14, 29), 0.88641975),
        (datetime(2022, 1, 5, 15, 14), 0.9975309),
    ])
    def test_parse_rth(self, t, expected):
        expected = np.array([[expected]])
        exchange = Exchange()
        feature = FeatureIsRTH(kind='rth')
        feature.reset(exchange=exchange)
        nbbo = EventNBBO(pd.Timestamp(t), ETF('SPY'), 439.11, 439.17)
        exchange.process_EventNBBO(nbbo)
        actual = feature.parse()
        np.testing.assert_almost_equal(actual, expected)

    @pytest.mark.parametrize('t,expected', [
        (datetime(2022, 1, 1, 13), 0.3333333),  # Saturday
        (datetime(2022, 1, 2, 13), 0.70114943),  # Sunday
        (datetime(2022, 1, 3, 8, 29), 0.9997446),  # Sunday
        (datetime(2022, 1, 5, 8, 29), 0.99903382),
        (datetime(2022, 1, 5, 8, 31), 0.),
        (datetime(2022, 1, 5, 9), 0.),
        (datetime(2022, 1, 5, 14, 29), 0.),
        (datetime(2022, 1, 5, 15, 14), 0.),
    ])
    def test_parse_eth(self, t, expected):
        expected = np.array([[expected]])
        exchange = Exchange()
        feature = FeatureIsRTH(kind='eth')
        feature.reset(exchange=exchange)
        nbbo = EventNBBO(pd.Timestamp(t), ETF('SPY'), 439.11, 439.17)
        exchange.process_EventNBBO(nbbo)
        actual = feature.parse()
        np.testing.assert_almost_equal(actual, expected)

