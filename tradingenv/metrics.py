"""It is harder to convert returns to levels than viceversa because the former
requires to know what was the base date."""
from typing import Callable, Type, Union
from pandas.core.generic import NDFrame
from functools import wraps
from collections import OrderedDict
from inspect import isclass
import numpy as np
import pandas as pd
from statsmodels.api import OLS
from statsmodels.regression.linear_model import RegressionResultsWrapper
from numbers import Number
import scipy.cluster.hierarchy as cluster
import calendar

BDAYS = 252


def to_pandas(method: Callable = None, target: Type = NDFrame):
    """This decorator simply adds method to the signature of cls, which is
    NDFrame by default.

    Parameters
    ----------
    method : Callable
        A function definition whose first argument is an instance of cls.
    target : Type
        A class type. Input 'method' will be set as attribute of this 'target'
        class.


    Notes
    -----
    When optional arguments are passed to properties, then the first argument
    ('method' in this case) is None. This requires special handling, managed
    by the big if-else below. The implementation has been inspired by:
    https://stackoverflow.com/a/24617244/7677636

    Examples
    --------
    >>> df = pd.DataFrame()
    >>> series = pd.Series(dtype=float)
    >>> # df.method()  # raises
    >>> # series.method()  # raises
    ...
    >>> @to_pandas(target=pd.DataFrame)
    ... def method(frame: NDFrame):
    ...     return 'Hello!'
    >>> df.method()
    'Hello!'
    >>> # series.method()  # AttributeError: 'Series' object has no attribute 'method'
    ...
    >>> @to_pandas(target=pd.Series)
    ... def method(frame: NDFrame):
    ...     return 'Hello!'
    >>> df.method()
    'Hello!'
    >>> series.method()
    'Hello!'
    """
    if isclass(method):
        raise ValueError("to_pandas only supports keyword arguments.")
    optional_kwargs = method is None
    if optional_kwargs:

        @wraps(method)
        def decorated(method):
            setattr(target, method.__name__, method)
            return method

        return decorated
    else:
        setattr(target, method.__name__, method)
        return method


class PandasMetrics(NDFrame):
    """Define here methods to be added to pandas.Series AND pandas.DataFrame
    using the decorator to_pandas. All these methods assume that data are
    daily levels."""

    @to_pandas
    def validate(self):
        """Check that the data is a proper level dataset."""
        if np.any(np.any(np.isnan(self.values))):
            raise ValueError("Missing values are not allowed.")
        if np.any(np.any(self.values <= 0)):
            raise ValueError(
                "This method can only work with level data, so "
                "all values must be positive."
            )
        if self.index.has_duplicates:
            raise ValueError("Duplicate indices have been found")
        if not isinstance(self.index, pd.DatetimeIndex):
            raise ValueError("Input index must be a DatetimeIndex")
        if self.index.hasnans:
            raise ValueError("Input must not missing values in the index.")
        if not self.index.is_monotonic_increasing:
            raise ValueError("Input index must be monotonic increasing.")

    @to_pandas
    def level(self):
        """Returns the object itself, which is assumed to be a level. In case
        of duplicate dates, only the last timestamp for each day will be kept.
        """
        self.validate()
        if len(np.unique(self.index.date)) != len(self.index.date):
            self = self.groupby(by=self.index.date).last()
        return self

    @to_pandas
    def simple_returns(self):
        """
        Examples
        --------
        >>> prices = pd.Series([1, 2, 3], pd.date_range('2019-01-01', periods=3))
        >>> prices.simple_returns()
        2019-01-02    1.0
        2019-01-03    0.5
        Freq: D, dtype: float64
        """
        # https://github.com/pandas-dev/pandas/issues/6389
        level = self.level()
        # FutureWarning: The 'fill_method' and 'limit' keywords in
        # Series.pct_change are deprecated and will be removed in a future
        # version. Call ffill before calling pct_change instead.
        simple_returns = level.pct_change()  # fill_method=None
        return simple_returns.iloc[1:]

    @to_pandas
    def log_returns(self):
        """
        Examples
        --------
        >>> prices = pd.Series([1, 2, 3], pd.date_range('2019-01-01', periods=3))
        >>> prices.log_returns()
        2019-01-02    0.693147
        2019-01-03    0.405465
        Freq: D, dtype: float64
        """
        level = self.level()
        log_returns = np.log(level).diff()
        return log_returns.iloc[1:]

    @to_pandas
    def nr_calendar_days(self) -> int:
        """
        Examples
        --------
        >>> prices = pd.DataFrame(
        ...     data=[1., 1.01, 1.03, 0.99, 1.02],
        ...     index=pd.date_range(start='2019-01-01', periods=5, freq='B'),
        ... )
        >>> prices.nr_calendar_days()
        6
        """
        return (self.last_valid_index() - self.first_valid_index()).days

    @to_pandas
    def nr_observations(self) -> int:
        """
        Examples
        --------
        >>> prices = pd.DataFrame(
        ...     data=[1., 1.01, 1.03, 0.99, 1.02],
        ...     index=pd.date_range(start='2019-01-01', periods=5, freq='B'),
        ... )
        >>> prices.nr_observations()
        5
        """
        valid_obs = self.loc[self.first_valid_index() : self.last_valid_index()]
        return len(valid_obs)

    @to_pandas
    def nr_years(self) -> float:
        """
        Examples
        --------
        >>> prices = pd.DataFrame(
        ...     data=np.linspace(1., 2., 366),
        ...     index=pd.date_range(start='2018-12-31', periods=366, freq='D'),
        ... )
        >>> prices.nr_years()
        1.0
        >>> prices.squeeze().nr_years()
        1.0
        """
        return self.nr_calendar_days() / 365

    @to_pandas
    def cagr(self):
        """
        Examples
        --------
        >>> prices = pd.DataFrame(
        ...     data=np.linspace(1., 1.5, 366),
        ...     index=pd.date_range(start='2018-12-31', periods=366, freq='D'),
        ...     columns=['S&P 500'],
        ... )
        >>> prices.cagr()
        S&P 500    0.5
        dtype: float64
        >>> prices.squeeze().cagr()
        np.float64(0.5)
        """
        level = self.level().bfill()
        years = self.nr_years()
        cagr = (level.iloc[-1] / level.iloc[0]) ** (1 / years) - 1
        return cagr

    @to_pandas
    def cumulative_return(self):
        """
        Examples
        --------
        >>> prices = pd.DataFrame(
        ...     data=np.linspace(1., 1.5, 10),
        ...     index=pd.date_range(start='2018-12-31', periods=10, freq='B'),
        ...     columns=['S&P 500'],
        ... )
        >>> prices.cumulative_return()
                     S&P 500
        2018-12-31  0.000000
        2019-01-01  0.055556
        2019-01-02  0.111111
        2019-01-03  0.166667
        2019-01-04  0.222222
        2019-01-07  0.277778
        2019-01-08  0.333333
        2019-01-09  0.388889
        2019-01-10  0.444444
        2019-01-11  0.500000
        """
        level = self.level()
        level_start = level.loc[level.first_valid_index()]
        return level / level_start - 1

    @to_pandas
    def overall_return(self):
        return self.cumulative_return().iloc[-1]

    @to_pandas
    def volatility(self):
        """
        Examples
        --------
        >>> prices = pd.DataFrame(
        ...     data=np.linspace(1., 1.5, 10),
        ...     index=pd.date_range(start='2018-12-31', periods=10, freq='B'),
        ...     columns=['S&P 500'],
        ... )
        >>> prices.volatility()
        S&P 500    0.092578
        dtype: float64
        >>> prices.squeeze().volatility()
        np.float64(0.092578...)
        """
        return np.sqrt(BDAYS) * self.simple_returns().std()

    @to_pandas
    def drawdown(self):
        """
        Examples
        --------
        >>> prices = pd.DataFrame(
        ...     data=[1., 1.01, 1.03, 0.99, 1.02, 1.03],
        ...     index=pd.date_range(start='2019-01-01', periods=6, freq='B'),
        ...     columns=['S&P 500'],
        ... )
        >>> prices.drawdown()
                     S&P 500
        2019-01-01  0.000000
        2019-01-02  0.000000
        2019-01-03  0.000000
        2019-01-04 -0.038835
        2019-01-07 -0.009709
        2019-01-08  0.000000
        """
        level = self.level()
        return level / level.cummax() - 1

    @to_pandas
    def max_drawdown(self):
        """
        Examples
        --------
        >>> prices = pd.DataFrame(
        ...     data=[1., 1.01, 1.03, 0.99, 1.02, 1.03],
        ...     index=pd.date_range(start='2019-01-01', periods=6, freq='B'),
        ...     columns=['S&P 500'],
        ... )
        >>> prices.max_drawdown()
        S&P 500   -0.038835
        dtype: float64
        """
        return self.drawdown().min()

    @to_pandas
    def value_at_risk(self, quantile: float = 0.025):
        """
        Examples
        --------
        >>> prices = pd.DataFrame(
        ...     data=[1., 1.01, 1.03, 0.99, 1.02, 1.03, 1., 0.98],
        ...     index=pd.date_range(start='2019-01-01', periods=8, freq='B'),
        ...     columns=['S&P 500'],
        ... )
        >>> prices.value_at_risk(quantile=0.025)
        S&P 500   -0.037379
        Name: 0.025, dtype: float64
        """
        simple_returns = self.simple_returns()
        return simple_returns.quantile(quantile)

    @to_pandas
    def expected_shortfall(self, quantile: float = 0.025):
        """
        Examples
        --------
        >>> prices = pd.DataFrame(
        ...     data=[1., 1.01, 1.03, 0.99, 1.02, 1.03, 1., 0.98],
        ...     index=pd.date_range(start='2019-01-01', periods=8, freq='B'),
        ...     columns=['S&P 500'],
        ... )
        >>> prices.expected_shortfall(quantile=0.025)
        S&P 500   -0.038835
        dtype: float64
        """
        simple_returns = self.simple_returns()
        value_at_risk = simple_returns.quantile(quantile)
        tail = simple_returns[simple_returns <= value_at_risk]
        return tail.mean()

    @to_pandas
    def downside_volatility(self):
        """
        Examples
        --------
        >>> prices = pd.DataFrame(
        ...     data=[1., 1.01, 1.03, 0.99, 1.02, 1.03, 1., 0.98],
        ...     index=pd.date_range(start='2019-01-01', periods=8, freq='B'),
        ...     columns=['S&P 500'],
        ... )
        >>> prices.downside_volatility()
        S&P 500    0.149522
        dtype: float64
        """
        simple_returns = self.simple_returns()
        neg_returns = simple_returns[simple_returns < 0]
        return np.sqrt(BDAYS) * neg_returns.std()

    @to_pandas
    def upside_volatility(self):
        """
        Examples
        --------
        >>> prices = pd.DataFrame(
        ...     data=[1., 1.01, 1.03, 0.99, 1.02, 1.03, 1., 0.98],
        ...     index=pd.date_range(start='2019-01-01', periods=8, freq='B'),
        ...     columns=['S&P 500'],
        ... )
        >>> prices.upside_volatility()
        S&P 500    0.154643
        dtype: float64
        """
        simple_returns = self.simple_returns()
        pos_returns = simple_returns[simple_returns > 0]
        return np.sqrt(BDAYS) * pos_returns.std()

    @to_pandas
    def martin_risk(self):
        """
        Examples
        --------
        >>> prices = pd.DataFrame(
        ...     data=[1., 1.01, 1.03, 0.99, 1.02, 1.03, 1., 0.98],
        ...     index=pd.date_range(start='2019-01-01', periods=8, freq='B'),
        ...     columns=['S&P 500'],
        ... )
        >>> prices.martin_risk()
        S&P 500    0.024513
        dtype: float64
        """
        # http://www.tangotools.com/ui/ui.htm
        # https://en.wikipedia.org/wiki/Ulcer_index
        return np.sqrt(self.drawdown().pow(2).mean())

    @to_pandas
    def tracking_error(self, other: NDFrame):
        """
        Examples
        --------
        >>> prices = pd.DataFrame(
        ...     data={
        ...         'S&P 500': [1., 1.01, 1.03, 0.99, 1.02, 1.03, 1., 0.98],
        ...         'SPY': [1., 1.011, 1.0298, 0.991, 1.0208, 1.031, 1., 0.979],
        ...     },
        ...     index=pd.date_range(start='2019-01-01', periods=8, freq='B'),
        ... )
        >>> prices['S&P 500'].tracking_error(prices['SPY'])
        np.float64(0.01536269...)
        """
        excess_returns = self.excess_returns(other)
        return np.sqrt(BDAYS) * excess_returns.std()

    @to_pandas
    def _parse_rate(self, rate: Union[NDFrame, float]) -> float:
        if isinstance(rate, NDFrame):
            rate = rate.squeeze().cagr()
        return rate

    @to_pandas
    def excess_returns(self, other: NDFrame):
        """
        Examples
        --------
        >>> prices = pd.DataFrame(
        ...     data={
        ...         'S&P 500': (1, 1.15, 1.18, 1.12),
        ...         'RiskFreeRate': (1, 1.02, 1.03, 1.04),
        ...     },
        ...     index=pd.date_range(start='2019-01-01', periods=4, freq='D'),
        ...     columns=['S&P 500', 'RiskFreeRate'],
        ... )
        >>> prices['S&P 500'].excess_returns(prices['RiskFreeRate'])
        2019-01-02    0.130000
        2019-01-03    0.016283
        2019-01-04   -0.060556
        Freq: D, dtype: float64
        """
        self_returns = self.simple_returns()
        other_returns = other.simple_returns().squeeze().reindex(self_returns.index)
        excess_returns = self_returns.subtract(other_returns, axis=0)
        return excess_returns

    @to_pandas
    def excess_cagr(self, over: Union[NDFrame, float] = 0.0):
        """
        Examples
        --------
        >>> prices = pd.DataFrame(
        ...     data={
        ...         'S&P 500': np.linspace(1, 1.15, 366),
        ...         'RiskFreeRate': np.linspace(1, 1.02, 366),
        ...     },
        ...     index=pd.date_range(start='2019-01-01', periods=366, freq='D'),
        ...     columns=['S&P 500', 'RiskFreeRate'],
        ... )
        >>> prices.excess_cagr(over=0.) == prices.cagr()
        S&P 500         True
        RiskFreeRate    True
        dtype: bool
        >>> prices.excess_cagr(over=0.02).round(6)
        S&P 500         0.13
        RiskFreeRate    0.00
        dtype: float64
        """
        return self.cagr() - self._parse_rate(over)

    @to_pandas
    def sharpe_ratio(self, risk_free: Union[NDFrame, float] = 0.0) -> float:
        """
        Examples
        --------
        >>> prices = pd.DataFrame(
        ...     data={
        ...         'S&P 500': [1., 1.01, 1.03, 1.03, 1., 1.01],
        ...         'RiskFreeRate': [1., 1.001, 1005, 1.006, 1.009, 1.01],
        ...     },
        ...     index=pd.date_range(start='2019-01-01', periods=6, freq='B'),
        ...     columns=['S&P 500', 'RiskFreeRate'],
        ... )
        >>> prices.sharpe_ratio(risk_free=prices['RiskFreeRate'])
        S&P 500         0.0
        RiskFreeRate    0.0
        dtype: float64
        >>> prices.sharpe_ratio(risk_free=prices['RiskFreeRate'].cagr())
        S&P 500         0.0
        RiskFreeRate    0.0
        dtype: float64
        >>> prices = pd.DataFrame(
        ...     data={
        ...         'S&P 500': [1., 1.01, 1.03, 1.03, 1., 1.03],
        ...         'RiskFreeRate': [1., 1.001, 1005, 1.007, 1.009, 1.01],
        ...     },
        ...     index=pd.date_range(start='2019-01-01', periods=6, freq='B'),
        ...     columns=['S&P 500', 'RiskFreeRate'],
        ... )
        >>> prices['S&P 500'].sharpe_ratio(risk_free=prices['RiskFreeRate'])
        np.float64(8.31680812940463)
        """
        return self.excess_cagr(risk_free) / self.volatility()

    @to_pandas
    def sortino_ratio(self, risk_free: Union[NDFrame, float] = 0.0) -> float:
        """
        Examples
        --------
        >>> prices = pd.DataFrame(
        ...     data={
        ...         'S&P 500': [1., 1.01, 1.02, 1.05, 1.03, 1.02],
        ...         'RiskFreeRate': [1., 1.001, 1005, 1.007, 1.009, 1.01],
        ...     },
        ...     index=pd.date_range(start='2019-01-01', periods=6, freq='B'),
        ...     columns=['S&P 500', 'RiskFreeRate'],
        ... )
        >>> prices['S&P 500'].sortino_ratio(risk_free=prices['RiskFreeRate'])
        np.float64(10.7621879372323)
        >>> risk_free_cagr = prices['RiskFreeRate'].cagr()
        >>> risk_free_cagr
        np.float64(0.6800754114925203)
        >>> prices['S&P 500'].sortino_ratio(risk_free_cagr)
        np.float64(10.7621879372323)
        """
        return self.excess_cagr(risk_free) / self.downside_volatility()

    @to_pandas
    def information_ratio(self, benchmark: NDFrame) -> float:
        return self.excess_cagr(benchmark) / self.tracking_error(benchmark)

    @to_pandas
    def calmar_ratio(self, risk_free: Union[NDFrame, float] = 0.0) -> float:
        return self.excess_cagr(risk_free) / -self.max_drawdown()

    @to_pandas
    def martin_ratio(self, risk_free: Union[NDFrame, float] = 0.0) -> float:
        # http://www.tangotools.com/ui/ui.htm
        # https://en.wikipedia.org/wiki/Ulcer_index
        # https://www.keyquant.com/Download/GetFile?Filename=%5CPublications%5CKeyQuant_WhitePaper_APT_Part1.pdf
        return self.excess_cagr(risk_free) / self.martin_risk()

    @to_pandas
    def omega_ratio(self, risk_free: Union[NDFrame, float] = 0.0, threshold: float = None):
        # Code from aric.
        # I'm not convinced with data quality and wouldn't trust the output.
        #if isinstance(risk_free, Number):
        #    risk_free = self.make_series_from_cagr(risk_free, "RiskFree")
        risk_free = self._parse_rate(risk_free)
        if threshold is None:
            threshold = (1 + risk_free) ** (1 / BDAYS) - 1

        def lpm(returns, threshold, order):
            # This method returns a lower partial moment of the returns
            # Create an array he same length as returns containing the minimum return threshold
            #threshold_array = np.empty(len(returns))
            #threshold_array.fill(threshold)
            # Calculate the difference between the threshold and the returns
            diff = threshold - returns
            # Set the minimum of each to 0
            diff = diff.clip(lower=0)
            # Return the sum of the different to the power of order
            return np.sum(diff ** order, axis=0) / len(returns) * np.sqrt(BDAYS)

        excess_cagr = self.excess_cagr(risk_free)
        return excess_cagr / lpm(self.simple_returns(), threshold, 1)

    @to_pandas
    def capm(
        self, benchmark: NDFrame, risk_free: Union[NDFrame, float] = 0.0
    ) -> RegressionResultsWrapper:
        """Run a regression for each asset (column) of self."""
        if isinstance(risk_free, Number):
            risk_free = self.make_series_from_cagr(risk_free, "RiskFree")
        benchmark = benchmark.squeeze()

        x = benchmark.excess_returns(risk_free)
        x = x.to_frame('beta')
        x["alpha"] = 1.0
        y = self.squeeze().excess_returns(risk_free)
        common_idx = x.index.intersection(y.index)
        linear_regression = OLS(
            endog=y.loc[common_idx],
            exog=x.loc[common_idx],
            hasconst=True,
        )
        return linear_regression.fit()

    @to_pandas
    def make_series_from_cagr(self, cagr: float, name: str = None) -> pd.Series:
        """
        Parameters
        ----------
        cagr : float
            Annualized compounded growth rate of the series that you want
            to get.
        name : str
            Name of the returned pd.Series.

        Returns
        -------
        Returns a series with the same index and values corresponding to
        a level series growing at the specified rate.

        Examples
        --------
        >>> series = pd.Series(index=pd.date_range('2019-01-01', periods=BDAYS+1), dtype=float)
        >>> prices = series.make_series_from_cagr(0.01)
        >>> prices.head()
        2019-01-01    1.000000
        2019-01-02    1.000039
        2019-01-03    1.000079
        2019-01-04    1.000118
        2019-01-05    1.000158
        Freq: D, dtype: float64
        >>> prices.tail()
        2019-09-06    1.00984
        2019-09-07    1.00988
        2019-09-08    1.00992
        2019-09-09    1.00996
        2019-09-10    1.01000
        Freq: D, dtype: float64
        """
        rate_daily = (1 + cagr) ** (1 / BDAYS)
        series = pd.Series(data=rate_daily, index=self.index, name=name)
        series.iloc[0] = 1.0
        return series.cumprod()

    @to_pandas
    def alpha(self, benchmark: NDFrame, risk_free: Union[NDFrame, float] = 0.0):
        """Returns alpha coefficient of CAPM regression."""
        capm = self.capm(benchmark, risk_free)
        return (1 + capm.params["alpha"]) ** BDAYS - 1

    @to_pandas
    def beta(self, benchmark: NDFrame, risk_free: Union[NDFrame, float] = 0.0):
        """Returns beta coefficient of CAPM regression."""
        capm = self.capm(benchmark, risk_free)
        return capm.params['beta']

    @to_pandas
    def _intersect_index(self, other: NDFrame):
        idx = self.index.intersection(other.index)
        return other.loc[idx]

    @to_pandas
    def returns_by_month(self) -> pd.DataFrame:
        """Returns a DataFrame with monthly returns by year and month.

        Examples
        -------
        >>> import tradingenv
        >>> import pandas as pd
        >>> series = pd.Series(index=pd.date_range('2019-01-01', periods=252*3), dtype=float)
        >>> prices = series.make_series_from_cagr(0.01).to_frame()
        >>> returns_by_month = prices.returns_by_month()
        """
        level = self.level()
        df = level.squeeze().resample('ME').last().pct_change()
        df.index = pd.MultiIndex.from_tuples(
            zip(df.index.strftime('%Y'), df.index.strftime('%b')),
            names=['Year', 'Month']
        )
        df = df.unstack()
        df.dropna(how='all', axis=0, inplace=True)
        df.dropna(how='all', axis=1, inplace=True)
        months = [m for m in calendar.month_abbr if m in df.columns]
        df = df[months]
        df['Year'] = (1 + df.fillna(0)).cumprod(1).iloc[:, -1] - 1
        return df

    @to_pandas
    def tearsheet(
        self,
        benchmark: NDFrame = None,
        risk_free: Union[NDFrame, float] = 0.0,
        weights: NDFrame = None,
        prices: NDFrame = None,
    ):
        """
        Calculate a tearsheet using daily finacial time series.

        Parameters
        ----------
        benchmark : NDFrame
            A NDFrame representing the levels series that you want to use to
            benchmark your investment strategy. This argument is mostly used
            when rendering the environment or calculating tearsheets.
        risk_free : Union[NDFrame, float]
            Risk free rate of the market (e.g. T-Bills, EURIBOR) expressed as
            a level pandas Series or DataFrame.
        weights : NDFrame
            Historical weights weights of the strategy. These can be found in
            TrackRecord.target_weights.
        prices : pd.DataFrame
            A pandas.DataFrame whose columns are asset _names, index are
            timestamps and values are prices.


        Returns
        -------
        A pd.DataFrame with several return, risk and risk-adjusted return
        metrics.

        Examples
        -------
        >>> returns = pd.DataFrame(
        ...     data=np.random.normal(0, 0.01, (1000, 3)),
        ...     index=pd.date_range('2010', freq='D', periods=1000),
        ...     columns=['Strategy', 'S&P 500', 'T-Notes']
        ... )
        >>> level = (1 + returns).cumprod()
        >>> tearsheet = level['Strategy'].tearsheet(
        ...     benchmark=level['S&P 500'],
        ...     risk_free=level['T-Notes'],
        ... )
        """
        if isinstance(risk_free, Number):
            risk_free = self.make_series_from_cagr(risk_free, "RiskFree")

        # TODO: test tearsheet
        # Ensure that we analyse data over the same time span (daily data)
        index = self.index
        for data in [benchmark, risk_free, weights, prices]:
            if data is not None:
                index = index.intersection(data.index)
        self = self.loc[index]
        risk_free = risk_free.loc[index]

        # Construct the tearsheet.
        tearsheet = OrderedDict()
        tearsheet[("Context", "From")] = self.first_valid_index().date()
        tearsheet[("Context", "To")] = self.last_valid_index().date()
        tearsheet[("Context", "Years")] = self.nr_years()
        tearsheet[("Context", "Observations")] = self.nr_observations()
        tearsheet[("Context", "Risk-free asset")] = str(risk_free.squeeze().name)
        tearsheet[("Context", "Risk-free CAGR")] = self._parse_rate(risk_free)
        tearsheet[("Return", "CAGR")] = self.cagr()
        tearsheet[("Return", "CAGR over cash")] = self.excess_cagr(risk_free)
        tearsheet[("Return", "Overall return")] = self.overall_return()
        tearsheet[("Risk", "Volatility")] = self.volatility()
        tearsheet[("Risk", "Downside volatility")] = self.downside_volatility()
        tearsheet[("Risk", "Upside volatility")] = self.upside_volatility()
        tearsheet[("Risk", "Max drawdown")] = self.max_drawdown()
        tearsheet[("Risk", "Martin risk")] = self.martin_risk()
        tearsheet[("Risk", "VaR 5%")] = self.value_at_risk(0.05)
        tearsheet[("Risk", "VaR 2%")] = self.value_at_risk(0.02)
        tearsheet[("Risk", "Expected shortfall 5%")] = self.expected_shortfall(0.05)
        tearsheet[("Risk", "Expected shortfall 2%")] = self.expected_shortfall(0.02)
        tearsheet[("Risk-adjusted return", "Sharpe ratio")] = self.sharpe_ratio(
            risk_free
        )
        tearsheet[("Risk-adjusted return", "Sortino ratio")] = self.sortino_ratio(
            risk_free
        )
        tearsheet[("Risk-adjusted return", "Calmar ratio")] = self.calmar_ratio(
            risk_free
        )
        tearsheet[("Risk-adjusted return", "Martin ratio")] = self.martin_ratio(
            risk_free
        )
        tearsheet[("Risk-adjusted return", "Omega ratio")] = self.omega_ratio(
            risk_free
        )
        if benchmark is not None:
            benchmark = benchmark.loc[index]
            tearsheet[("Outperformance", "Benchmark id")] = str(benchmark.squeeze().name)
            tearsheet[("Outperformance", "CAGR over benchmark")] = self.excess_cagr(
                benchmark
            )
            tearsheet[("Outperformance", "Information ratio")] = self.information_ratio(
                benchmark
            )
            tearsheet[("Outperformance", "CAPM Alpha")] = self.alpha(
                benchmark, risk_free
            )
            tearsheet[("Outperformance", "CAPM Beta")] = self.beta(benchmark, risk_free)
            tearsheet[("Outperformance", "Correlation")] = self.simple_returns().squeeze().corr(benchmark.simple_returns().squeeze())
        if weights is not None:
            weights = weights.loc[index]
            leverage = weights.sum(1)
            turnover = weights.diff().abs().sum(1)
            tearsheet[("Weights", "Leverage mean")] = leverage.mean()
            tearsheet[("Weights", "Turnover daily")] = turnover.mean()
            tearsheet[("Weights", "Turnover annual")] = BDAYS * turnover.mean()
            for asset_name, weight_avg in weights.mean().items():
                tearsheet[("Weights", str(asset_name))] = weight_avg
        if prices is not None:
            prices = prices.loc[index]
            for name, asset in prices.items():
                if asset.std() > 0:  # avoid stuff like cash
                    tearsheet[("Markets", "CAGR " + str(name))] = asset.cagr()
        return pd.DataFrame(self.__class__(tearsheet).T)


class SeriesMetrics(NDFrame):
    """Define here methods to be added to pandas.Series. All these methods
    assume that data are daily levels."""

    @to_pandas(target=pd.Series)
    def drawdown_periods(self):
        """
        Examples
        --------
        >>> prices = pd.Series(
        ...     data=[1., 1.01, 1.03, 0.99, 1.02, 1.03, 1., 0.98],
        ...     index=pd.date_range(start='2019-01-01', periods=8, freq='B'),
        ...     name='S&P 500',
        ... )
        >>> prices.drawdown_periods()
           Max drawdown       From     Trough         To  Nr days
        1     -0.048544 2019-01-09 2019-01-10 2019-01-10        2
        0     -0.038835 2019-01-04 2019-01-04 2019-01-07        2
        """
        drawdown = self.drawdown()
        group_ids = (drawdown == 0).cumsum()
        group_ids = group_ids[drawdown != 0]
        groups = drawdown.groupby(group_ids)
        drawdown_periods = pd.DataFrame(
            {
                "Max drawdown": groups.min(),
                "From": groups.aggregate(lambda series: series.index.min()),
                "Trough": groups.idxmin(),
                "To": groups.aggregate(lambda series: series.index.max()),
                "Nr days": groups.count(),
            }
        )
        drawdown_periods.reset_index(inplace=True, drop=True)
        return drawdown_periods.sort_values(by="Max drawdown")

    @to_pandas(target=pd.Series)
    def linear_regression(self, reset_index: bool=False):
        """
        Examples
        -------
        >>> prices = pd.Series(
        ...     data=[1., 1.01, 1.03, 0.99, 1.02, 1.03, 1., 0.98],
        ...     index=pd.date_range(start='2019-01-01', periods=8, freq='B'),
        ...     name='S&P 500',
        ... )
        >>> trendline = prices.linear_regression(reset_index=True)
        >>> trendline['trendline']
        2019-01-01    1.014167
        2019-01-02    1.012262
        2019-01-03    1.010357
        2019-01-04    1.008452
        2019-01-07    1.006548
        2019-01-08    1.004643
        2019-01-09    1.002738
        2019-01-10    1.000833
        Freq: B, dtype: float64
        >>> trendline['residuals']
        2019-01-01   -0.014167
        2019-01-02   -0.002262
        2019-01-03    0.019643
        2019-01-04   -0.018452
        2019-01-07    0.013452
        2019-01-08    0.025357
        2019-01-09   -0.002738
        2019-01-10   -0.020833
        Freq: B, dtype: float64
        >>> trendline['std']  # doctest: +ELLIPSIS
        0.01771850...
        """
        y = self
        if reset_index:
            x = pd.DataFrame(np.arange(len(self)), index=self.index)
        else:
            x = self.index.to_frame()
        x['constant'] = 1.
        ols = OLS(endog=y, exog=x, hasconst=True)
        ols_results = ols.fit()
        return {
            "model": ols_results,
            "level": self,
            "trendline": ols_results.predict(x),
            "residuals": ols_results.resid,
            "std": float(ols_results.resid.std()),
        }


class DataFrameMetrics(NDFrame):
    """Define here methods to be added to pandas.Series. All these methods
    assume that data are daily levels."""

    @to_pandas(target=pd.DataFrame)
    def covariance_matrix(self):
        """Returns covariance matrix of simple returns."""
        returns = self.simple_returns()
        return returns.cov()

    @to_pandas(target=pd.DataFrame)
    def correlation_matrix(self, clustering: bool = False):
        """Returns correlation matrix. If clustering=True, the correlation
        matrix will be sorted accoring to the clusters and clusters will be
        returned."""
        returns = self.simple_returns()
        corr_mat = returns.corr()
        if clustering:
            distances = cluster.distance.pdist(corr_mat.values)
            linkages = cluster.linkage(distances, method="complete")
            ind = cluster.fcluster(linkages, 0.5 * distances.max(), "distance")
            columns = [corr_mat.columns[i] for i in np.argsort(ind)]
            corr_mat = corr_mat.reindex(columns, axis=0).reindex(columns, axis=1)

            # Formatting of clusters.
            clusters = dict()
            for i in set(ind):
                mask = ind == i
                clusters[i] = corr_mat.columns[mask].to_list()
            return corr_mat, clusters
        else:
            return corr_mat
