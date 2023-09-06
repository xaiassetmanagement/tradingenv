"""TrackRecord stores historical _rebalancing execution performed by Broker."""
from tradingenv.broker.rebalancing import Rebalancing
from typing import Dict, Union, Sequence
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objs as go
import plotly.figure_factory
from pathlib import Path
import os


class TrackRecord:
    """Stores historical _rebalancing requests and _rebalancing."""

    def __init__(self):
        self._time = list()
        self._rebalancing: Dict[datetime, Rebalancing] = dict()
        self._nr_steps_to_burn = 0
        self._trading_started = False

    def __getitem__(self, item: Union[int, datetime]) -> Rebalancing:
        """If item is an int, the item-th RebalancingResponse will be returned.
        If item is datetime, the report associated with the specified time
        will be returned."""
        if isinstance(item, datetime):
            return self._rebalancing[item]
        else:
            return self._rebalancing[self._time[item]]

    def __len__(self):
        """Returns the count of historical checkpoints."""
        return len(self._rebalancing)

    def __repr__(self):
        return "{cls}({start}:{end}; items={nr_items})" "".format(
            cls=self.__class__.__name__,
            start=self._time[0],
            end=self._time[-1],
            nr_items=len(self._time),
        )

    def _checkpoint(self, rebalancing: Rebalancing):
        """Store an instance of RebalancingResponse. This class will take
        care of parsing the content using other methods such as
        TrackRecord.to_frame."""
        time = rebalancing.time
        if isinstance(time, pd.Timestamp):
            time = time.to_pydatetime()
        if time in self._time:
            raise ValueError(
                "All RebalancingResponse must have different timestamps. "
                "Duplicated timestamp found: {}".format(time)
            )
        self._time.append(time)
        self._rebalancing[time] = rebalancing
        self._trading_started = self._trading_started or len(rebalancing.trades) != 0
        if not self._trading_started:
            self._nr_steps_to_burn += 1

    def save(self, directory: str, name: str=None):
        # Use cloudpickle instead?
        os.makedirs(directory, exist_ok=True)
        file_name = name or self.name + '.pickle'
        path = Path(directory).joinpath(file_name)
        with open(str(path), "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> 'TrackRecord':
        # Use cloudpickle instead?
        with open(path, "rb") as f:
            track_record = pickle.load(f)
        return track_record

    def net_liquidation_value(
            self,
            before_rebalancing=True,
            burn: bool=False,
            name: str='Net liquidation value',
            index_name: str='Date',
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        before_rebalancing
        burn : bool
            Default is False (least surprise principle). If True, initial
            checkpoints with no trades (i.e. when the portfolio is cash only)
            will be discarded when calling TrackRecord.to_frame, unless
            a burn argument is explicitly passed to TrackRecord.to_frame.
        name
        index_name

        Returns
        -------

        """
        data = dict()
        for time, rebalancing in self._rebalancing.items():
            if before_rebalancing:
                context = rebalancing.context_pre
            else:
                context = rebalancing.context_post
            data[time] = context.nlv
        series = pd.Series(data=data, name=name, dtype=np.float64)
        series.index.name = index_name
        if burn:
            series = series.iloc[self._nr_steps_to_burn:]
        return series.to_frame()

    def weights_target(
            self,
            burn: bool=False,
            name: str='Portfolio weights target',
            index_name: str='Date',
            aggregate_future_chain: bool=False,
    ) -> pd.DataFrame:
        data = dict()
        for time, rebalancing in self._rebalancing.items():
            data[time] = rebalancing.allocation
        df = pd.DataFrame(data=data, dtype=np.float32).T
        df.index.name = index_name
        df.columns.name = name
        if burn:
            df = df.iloc[self._nr_steps_to_burn:]
        if aggregate_future_chain:
            groups = [c.symbol_short for c in df.columns]
            if len(groups) != len(df.columns):
                df = df.groupby(groups, axis='columns').sum()
        return df

    def weights_actual(
            self,
            before_rebalancing=True,
            burn: bool=False,
            name: str='Portfolio weights actual',
            index_name: str='Date',
            aggregate_future_chain: bool = False,
    ) -> pd.DataFrame:
        data = dict()
        for time, rebalancing in self._rebalancing.items():
            if before_rebalancing:
                context = rebalancing.context_pre
            else:
                context = rebalancing.context_post
            data[time] = context.weights
        df = pd.DataFrame(data=data, dtype=np.float32).T
        df.index.name = index_name
        df.columns.name = name
        if burn:
            df = df.iloc[self._nr_steps_to_burn:]
        if aggregate_future_chain:
            groups = [c.symbol_short for c in df.columns]
            if len(groups) != len(df.columns):
                df = df.groupby(groups, axis='columns').sum()
        return df

    def transaction_costs(
            self,
            burn: bool=False,
            index_name: str='Date',
            cumulative: bool=True,
    ) -> pd.DataFrame:
        profit_on_idle_cash = dict()
        cost_of_spread = dict()
        cost_of_commissions = dict()
        for time, rebalancing in self._rebalancing.items():
            profit_on_idle_cash[time] = rebalancing.profit_on_idle_cash
            cost_of_spread[time] = sum(trade.cost_of_spread for trade in rebalancing.trades)
            cost_of_commissions[time] = sum(trade.cost_of_commissions for trade in rebalancing.trades)
        df = pd.concat(
            objs=[
                pd.Series(profit_on_idle_cash).to_frame('Profit on idle Cash'),
                pd.Series(cost_of_spread).to_frame('Spread'),
                pd.Series(cost_of_commissions).to_frame('Broker fees'),
            ],
            axis='columns',
        )
        df.index.name = index_name
        if burn:
            df = df.iloc[self._nr_steps_to_burn:]
        if cumulative:
            df = df.cumsum().join(self.net_liquidation_value())
        return df

    def to_excel(self, path):
        """Export track record to excel."""
        nlv = self.net_liquidation_value()
        nlv.index = nlv.index.date
        nlv.index.name = 'Date'
        weights = self.weights_target(aggregate_future_chain=True)
        weights.index = weights.index.date
        weights.index.name = 'Date'
        costs = self.transaction_costs()
        costs.index = costs.index.date
        costs.index.name = 'Date'
        with pd.ExcelWriter(path) as excel:
            nlv.to_excel(excel, 'Performance')
            weights.to_excel(excel, 'Weights')
            costs.to_excel(excel, 'Costs')

    def cost_of_spread(self):
        raise NotImplementedError()

    def cost_of_commissions(self):
        raise NotImplementedError()
    
    def _get_df_performance(self):
        nlv = self.net_liquidation_value()
        risk_free = self.risk_free.loc[nlv.index]
        benchmark = self.benchmark.loc[nlv.index]
        relative = (nlv.squeeze() / benchmark.squeeze()).to_frame(
            'Outperformance')
        df = pd.concat([nlv, risk_free, benchmark, relative], axis='columns')
        base_date = df.first_valid_index()
        df /= df.loc[base_date]
        df *= 100
        return df

    def fig_net_liquidation_value(self) -> go.Figure:
        df = self._get_df_performance()
        layout = go.Layout(
            title=go.layout.Title(
                text='Performance of {}'.format(self.name),
            ),
            yaxis=go.layout.YAxis(
                type='log',
                title=go.layout.yaxis.Title(text="Level")
            ),
        )
        data = list()
        for col in df.columns:
            scatter = go.Scatter(
                x=df.index,
                y=df[col],
                mode='lines',
                name=str(col),
                line=go.scatter.Line(width=2),
            )
            data.append(scatter)
        return go.Figure(data=data, layout=layout)

    def fig_drawdown(self) -> go.Figure:
        df = self._get_df_performance()
        df = df.drawdown()
        df *= 100

        # Layout.
        layout = go.Layout(
            title=go.layout.Title(
                text='Drawdown',
            ),
            yaxis=go.layout.YAxis(
                title=go.layout.yaxis.Title(text="%")
            ),
        )

        # Data.
        data = list()
        for col in df.columns:
            scatter = go.Scatter(
                x=df.index,
                y=df[col],
                mode='lines',
                name=str(col),
                line=go.scatter.Line(width=2),
            )
            data.append(scatter)
        return go.Figure(data=data, layout=layout)

    def fig_annual_returns(self) -> go.Figure:
        df = self._get_df_performance()

        annual_rets = df.resample("Y").last() / df.resample("Y").first() - 1
        annual_rets.index = annual_rets.index.year
        annual_rets *= 100

        # Layout.
        layout = go.Layout(
            title=go.layout.Title(
                text='Annual returns of {}'.format(self.name),
            ),
            yaxis=go.layout.YAxis(
                title=go.layout.yaxis.Title(text="%")
            ),
        )

        # Data.
        data = list()
        for col in annual_rets.columns:
            scatter = go.Bar(
                x=annual_rets.index,
                y=annual_rets[col],
                name=str(col),
            )
            data.append(scatter)
        return go.Figure(data=data, layout=layout)

    def fig_transaction_costs(self) -> go.Figure:
        df = self.transaction_costs(cumulative=True)
        layout = go.Layout(
            title=go.layout.Title(
                text='Transaction costs',
            ),
            yaxis=go.layout.YAxis(
                title=go.layout.yaxis.Title(text="$")
            ),
        )
        data = list()
        for col in df.columns:
            scatter = go.Scatter(
                x=df.index,
                y=df[col],
                mode='lines',
                name=str(col),
                line=go.scatter.Line(width=2),
            )
            data.append(scatter)
        return go.Figure(data=data, layout=layout)

    def fig_returns_distribution(self):
        nlv = pd.concat(
            objs=[
                self.net_liquidation_value(),
                self.benchmark,
            ],
            axis='columns',
            sort=False,
        )
        nlv.columns = [str(col) for col in nlv.columns]
        nlv.dropna(inplace=True)
        ret = nlv.log_returns()
        ret *= 100
        layout = go.Layout(
            title=go.layout.Title(text="Kernel Density Estimation"),
            yaxis=go.layout.YAxis(
                title=go.layout.yaxis.Title(text="Density")
            ),
            xaxis=go.layout.XAxis(
                title=go.layout.xaxis.Title(text="Return (%)")
            ),
        )
        data = plotly.figure_factory.create_distplot(
            hist_data=[ret[col] for col in ret.columns],
            group_labels=ret.columns,
            show_hist=False,
            show_curve=True,
            show_rug=False,
            curve_type="kde",
            histnorm="probability density",
        ).data
        return go.Figure(data=data, layout=layout)

    def fig_capital_asset_pricing_model(self) -> go.Figure:
        nlv = self.net_liquidation_value()
        capm = nlv.capm(
            benchmark=self.benchmark,
            risk_free=self.risk_free,
        )
        alpha = capm.params['alpha']
        beta = capm.params['beta']
        x = capm.model.data.exog[:, 0] * 100
        y = capm.model.data.endog * 100
        trendline = capm.fittedvalues * 100
        layout = go.Layout(
            title=go.layout.Title(
                text="Capital Asset Pricing Model<br>α={alpha:.2%} (annualized), β={beta:.2f}".format(
                    alpha=alpha, beta=beta),
            ),
            yaxis=go.layout.YAxis(
                title=go.layout.yaxis.Title(
                    text="Return (%) of the strategy over cash")
            ),
            xaxis=go.layout.XAxis(
                title=go.layout.xaxis.Title(
                    text="Return (%) of {} over cash".format(
                        self.benchmark.squeeze().name))
            ),
        )
        data = [
            go.Scatter(
                x=x,
                y=y,
                text=[t.date() for t in nlv.index],
                mode="markers",
                name="Returns",
                marker=go.scatter.Marker(size=5),
            ),
            go.Scatter(
                x=x,
                y=trendline,
                mode="lines",
                name="Trendline",
                showlegend=True,
            ),
        ]
        return go.Figure(data=data, layout=layout)

    def _historical_portfolio_weights(
            self, df: pd.DataFrame, title: str
    ) -> go.Figure:
        df *= 100
        layout = go.Layout(
            title=go.layout.Title(
                text=title,
            ),
            yaxis=go.layout.YAxis(
                title=go.layout.yaxis.Title(text="%")
            ),
        )
        data = list()
        for col, series in df.items():
            series = series.replace(0, np.nan).dropna()
            scatter = go.Scatter(
                x=series.index,
                y=series,
                mode='lines',
                name=str(col),
                line=go.scatter.Line(width=2),
            )
            data.append(scatter)
        return go.Figure(data=data, layout=layout)

    def fig_historical_portfolio_weights_actual(
            self, group_futures: bool=True, post_rebalancing: bool=True
    ) -> go.Figure:
        df = self.weights_actual(
            aggregate_future_chain=bool(group_futures),
            before_rebalancing=not bool(post_rebalancing),
        )
        title = 'Historical actual portfolio weights'
        return self._historical_portfolio_weights(df, title)

    def fig_historical_portfolio_weights_target(self, group_futures: bool=True):
        df = self.weights_target(
            aggregate_future_chain=bool(group_futures),
        )
        title='Historical target portfolio weights'
        return self._historical_portfolio_weights(df, title)

    def fig_historical_portfolio_weights_diff(
            self, group_futures: bool=True, post_rebalancing: bool=True
    ) -> go.Figure:
        df1 = self.weights_target(
            aggregate_future_chain=bool(group_futures),
        )
        df2 = self.weights_actual(
            aggregate_future_chain=bool(group_futures),
            before_rebalancing=not bool(post_rebalancing),
        )
        df = df1 - df2
        title = 'Historical target minus actual portfolio weights'
        return self._historical_portfolio_weights(df, title)


class TrackRecordComparison:
    def __init__(self, track_records: Sequence[TrackRecord]):
        self.track_records = self._parse_track_records(track_records)
        self._names = [tr.name for tr in self.track_records]
        if len(self._names) != len(set(self._names)):
            raise ValueError(
                'Duplicate track record names are now allowed:\n{}'
                ''.format(self._names)
            )

    def __getitem__(self, idx: Union[int, str]) -> TrackRecord:
        if isinstance(idx, str):
            idx = self._names.index(idx)
        return self.track_records[idx]

    def __repr__(self) -> str:
        return repr(self.track_records)

    @staticmethod
    def _parse_track_records(data) -> Sequence[TrackRecord]:
        track_records = list()
        if isinstance(data, TrackRecord):
            track_records.append(data)
        else:
            for item in data:
                if isinstance(item, str):
                    # Item is assumed to be a path to TrackRecord.
                    track_records.append(TrackRecord.load(item))
                elif isinstance(item, TrackRecord):
                    track_records.append(item)
                else:
                    raise ValueError("Unexpected 'data'.")
        return track_records

    def fig_net_liquidation_value(self, logy: bool=False) -> go.Figure:
        layout = go.Layout(
            title=go.layout.Title(
                text='Net liquidation value',
            ),
            yaxis=go.layout.YAxis(
                type='log' if logy else None,
                title=go.layout.yaxis.Title(text="$"),
            ),
        )
        data = list()
        for track_record in self.track_records:
            series = track_record.net_liquidation_value()
            scatter = go.Scatter(
                x=series.index,
                y=series.squeeze(),
                mode='lines',
                name=track_record.name,
                line=go.scatter.Line(width=2),
            )
            data.append(scatter)
        return go.Figure(data=data, layout=layout)
