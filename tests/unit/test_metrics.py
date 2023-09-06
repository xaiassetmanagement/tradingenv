from tradingenv.metrics import to_pandas, PandasMetrics
from datetime import datetime
import pandas as pd
import numpy as np
import pytest
import unittest.mock
from pandas.core.generic import NDFrame


def test_to_pandas_default_args():
    df = pd.DataFrame()
    series = pd.Series(dtype=float)
    with pytest.raises(AttributeError):
        df.get_answer()
    with pytest.raises(AttributeError):
        series.get_answer()

    @to_pandas
    def get_answer(pandas_obj):
        return 42

    assert df.get_answer() == 42
    assert series.get_answer() == 42


def test_to_pandas_custom_target_class():
    df = pd.DataFrame()
    series = pd.Series(dtype=float)
    with pytest.raises(AttributeError):
        df.get_magic_number()
    with pytest.raises(AttributeError):
        series.get_magic_number()

    @to_pandas(target=pd.Series)
    def get_magic_number(pandas_obj):
        return 42

    with pytest.raises(AttributeError):
        df.get_magic_number()
    assert series.get_magic_number() == 42


class TestMetrics:
    def test_nr_days_with_non_business(self):
        index = pd.date_range("2017-12-31", "2018-12-31", freq="D")
        series = pd.Series(range(1, 1 + len(index)), index)
        assert series.nr_calendar_days() == 365
        assert series.nr_observations() == 366

    def test_nr_days_with_business(self):
        index = pd.date_range("2017-12-31", "2018-12-31", freq="B")
        series = pd.Series(range(1, 1 + len(index)), index)
        assert series.nr_calendar_days() == 364
        assert series.nr_observations() == 261

    def test_nr_days_with_unspecified_freq(self):
        index = pd.date_range("2017-12-31", "2018-12-31")
        series = pd.Series(range(1, 1 + len(index)), index)
        assert series.nr_calendar_days() == 365
        assert series.nr_observations() == 366

    def test_nr_years_with_non_business(self):
        index = pd.date_range("2017-12-31", "2018-12-31", freq="D")
        series = pd.Series(range(1, 1 + len(index)), index)
        assert series.nr_years() == 1

    def test_nr_years_with_business(self):
        index = pd.date_range("2017-12-31", "2018-12-31", freq="B")
        series = pd.Series(range(1, 1 + len(index)), index)
        assert series.nr_years() == 364 / 365

    def test_nr_years_with_unspecified_freq(self):
        index = pd.date_range("2017-12-31", "2018-12-31")
        series = pd.Series(range(1, 1 + len(index)), index)
        assert series.nr_years() == 1

    def test_level_returns_self(self):
        index = pd.date_range("2017-12-31", "2018-12-31")
        expected = pd.Series(range(1, 1 + len(index)), index)
        actual = expected.level()
        pd.testing.assert_series_equal(actual, expected)

    @unittest.mock.patch.object(pd.Series, "validate")
    def test_level_validates_self(self, mock_validate):
        index = pd.date_range("2017-12-31", "2018-12-31")
        series = pd.Series(range(1, 1 + len(index)), index)
        series.level()
        mock_validate.assert_called_once()

    def test_level_only_returns_last_observation_for_each_day(self):
        series = pd.Series(
            data={
                datetime(2019, 1, 1, 2): 1,
                datetime(2019, 1, 1, 20): 2,
                datetime(2019, 1, 3, 5, 15, 2): 3,
                datetime(2019, 1, 3, 5, 16, 1): 4,
                datetime(2019, 1, 3, 5, 16, 2): 5,
            }
        )
        actual = series.level()
        expected = pd.Series(data={datetime(2019, 1, 1): 2, datetime(2019, 1, 3): 5})
        actual.index = pd.to_datetime(actual.index)
        expected.index = pd.to_datetime(expected.index)
        pd.testing.assert_series_equal(actual, expected, check_index_type=False)

    def test_cagr(self):
        series = pd.Series([1, 2], [datetime(2017, 12, 31), datetime(2018, 12, 31)])
        assert series.cagr() == 1

    def test_log_returns(self):
        series = pd.Series([1, 2, 4], pd.date_range("2019-01-01", periods=3))
        expected = pd.Series(
            [np.log(2 / 1), np.log(4 / 2)], pd.date_range("2019-01-02", periods=2)
        )
        pd.testing.assert_series_equal(series.log_returns(), expected)

    def test_simple_returns(self):
        series = pd.Series([1, 2, 4], pd.date_range("2019-01-01", periods=3))
        expected = pd.Series(
            [2 / 1 - 1, 4 / 2 - 1], pd.date_range("2019-01-02", periods=2)
        )
        pd.testing.assert_series_equal(series.simple_returns(), expected)

    def test_simple_returns_with_nans(self):
        series = pd.Series([1, np.nan, 4], pd.date_range("2019-01-01", periods=3))
        with pytest.raises(ValueError):
            series.simple_returns()

    def test_volatility(self):
        series = pd.Series([1, 2, 4, 3, 6, 4], pd.date_range("2019-01-01", periods=6))
        expected = series.pct_change().iloc[1:].std() * np.sqrt(252)
        assert series.volatility() == expected

    def test_drawdown(self):
        series = pd.Series(
            [1, 2, 4, 3, 6, 4], pd.date_range("2019-01-01", periods=6), name="S&P 500"
        )
        expected = series / series.cummax() - 1
        pd.testing.assert_series_equal(series.drawdown(), expected)

    def test_max_drawdown(self):
        series = pd.Series(
            [1, 2, 4, 3, 6, 4], pd.date_range("2019-01-01", periods=6), name="S&P 500"
        )
        expected = 4 / 6 - 1
        assert series.max_drawdown() == expected

    def test_value_at_risk(self):
        # 102 prices -> 101 returns.
        series = pd.Series(range(1, 103), pd.date_range("2019-01-01", periods=102))
        assert series.value_at_risk(0) == 102 / 101 - 1
        assert series.value_at_risk(0.01) == 101 / 100 - 1
        assert series.value_at_risk(0.5) == (52 / 51) - 1
        assert series.value_at_risk(0.99) == 3 / 2 - 1
        assert series.value_at_risk(1) == 2 / 1 - 1

    def test_expected_shortfall(self):
        # 102 prices -> 101 returns.
        series = pd.Series(range(1, 103), pd.date_range("2019-01-01", periods=102))
        returns = series.pct_change().iloc[1:]
        assert series.expected_shortfall(0) == returns.iloc[100]
        assert series.expected_shortfall(0.01) == returns.iloc[99:].mean()
        assert series.expected_shortfall(0.5) == returns.iloc[50:].mean()
        assert series.expected_shortfall(0.99) == returns.iloc[1:].mean()
        assert series.expected_shortfall(1) == returns.mean()

    def test_downside_volatility(self):
        series = pd.Series(
            [1, 2, 1.8, 1.5, 1.2, 1.1],
            pd.date_range("2019-01-01", periods=6),
            name="S&P 500",
        )
        returns = series.pct_change().iloc[1:]
        expected = np.sqrt(252) * returns[returns < 0].std()
        assert series.downside_volatility() == expected

    def test_upside_volatility(self):
        series = pd.Series(
            [1, 2, 4, 3, 5, 7], pd.date_range("2019-01-01", periods=6), name="S&P 500"
        )
        returns = series.pct_change().iloc[1:]
        expected = np.sqrt(252) * returns[returns > 0].std()
        assert series.upside_volatility() == expected

    def test_cumulative_returns(self):
        series = pd.Series(
            [1, 1.2, 1.4, 1.3, 1.5, 1.7], pd.date_range("2019-01-01", periods=6)
        )
        actual = series.cumulative_return()
        expected = series / series.iloc[0] - 1
        assert actual.equals(expected)

    def test_cagr_equals_excess_return_over_zero(self):
        series = pd.Series(
            [1, 1.2, 1.4, 1.3, 1.5, 1.7], pd.date_range("2019-01-01", periods=6)
        )
        assert series.cagr() == series.excess_cagr(over=0.0)

    def test_validate_raises_if_missing_values(self):
        series = pd.Series([1, np.nan, 3], pd.date_range("2019-01-01", periods=3))
        with pytest.raises(ValueError):
            series.validate()

    def test_validate_raises_if_non_posive_values(self):
        series = pd.Series([1, -1, 3], pd.date_range("2019-01-01", periods=3))
        with pytest.raises(ValueError):
            series.validate()

    def test_validate_raises_if_duplicate_indices(self):
        series = pd.Series(
            [1, 2, 3],
            [datetime(2019, 1, 1), datetime(2019, 1, 2), datetime(2019, 1, 2)],
        )
        with pytest.raises(ValueError):
            series.validate()

    def test_validate_raises_if_not_DatetimeIndex(self):
        series = pd.Series([1, 2, 3])
        with pytest.raises(ValueError):
            series.validate()

    def test_validate_raises_if_index_has_nans(self):
        series = pd.Series(
            [1, 2, 3], [datetime(2019, 1, 1), np.nan, datetime(2019, 1, 2)]
        )
        with pytest.raises(ValueError):
            series.validate()

    def test_validate_raises_if_index_is_not_sorted(self):
        series = pd.Series(
            [1, 2, 3],
            [datetime(2019, 1, 2), datetime(2019, 1, 3), datetime(2019, 1, 1)],
        )
        with pytest.raises(ValueError):
            series.validate()


class TestTearsheet:
    def test_tearsheet(self):
        pass
