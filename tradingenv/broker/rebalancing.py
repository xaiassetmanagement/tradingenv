from tradingenv.broker.allocation import Weights, NrContracts
import tradingenv
from tradingenv.contracts import AbstractContract
from tradingenv.broker.trade import Trade
from typing import Sequence, List
from datetime import datetime


class Rebalancing:
    """Private class returned by PortfolioSpace._make_rebalancing_request and
    passed to RebalancingResponse during its initialization."""

    def __init__(
        self,
        contracts: Sequence[AbstractContract] = None,
        allocation: Sequence[float] = None,
        measure: str = 'weight',
        absolute: bool = True,
        fractional: bool = True,
        margin: float = 0.0,
        time: datetime = None,
    ):
        """
        Parameters
        ----------
        contracts
            A sequence of contracts. The i-th contract is associated with the
            i-th element in 'allocation'.
        allocation
            A sequence of target portfolio allocations of the _rebalancing.
            The i-th allocation is associated with the i-th element
            in 'contracts'. The representation of the values in the sequence
            depends on the parameter 'kind'.
        measure
            The unit of measurement of the parameter 'allocation'. If:
            - 'weight': a 0.2 would mean 20% of the net liquidation value of
            the current portfolio.
            - 'nr-contracts': a 21 would mean 21 contracts.
        absolute
            If True (default), 'allocation' is assumed to represents the
            desired target allocation. For example, if
            allocation={ETF(SPY): 0.03} and measure='weight', then
            the desired _rebalancing will result in an allocation in ETF(SPY)
            corresponding to 3% of the net liquidation value and the remaining
            97% in cash.
            If False, 'allocation' is assumed to represent the desired change
            from the current allocation. or example, if
            allocation={ETF(SPY): 0.03} and measure='weight', then
            the desired _rebalancing will result in the previously held
            portfolio with 3% less of cash and 3% more of ETF(SPY).
        fractional
            True by default. If false, decimals will be ignored from every
            element of 'allocation'.
        margin
            A non-negative float. Rebalancing of positions which are not at
            least different by '_margin' in absolute value, will not be
            executed. Setting a small positive value (e.g. 0.02==2%) might be
            a good practice to reduce transaction costs.
            Skip trade if the absolute weight imbalance is below the margin
            AND the target weight is different from zero. The latter condition
            allows to liquidate a position in the situation where we hold in
            the portfolio a contract with a tiny weight below the margin.
        time
            Time of the _rebalancing request. Current time by default.

        Notes
        -----
        Target weights are applied to the net liquidation value of the account
        before paying transaction costs to make the problem more trackable).
        Therefore, in presence transaction costs, the amount of cash in the
        broker account will be -x where x are the broker commissions paid.
        """
        if measure == 'weight':
            self.allocation = Weights(keys=contracts, values=allocation)
        elif measure == 'nr-contracts':
            self.allocation = NrContracts(keys=contracts, values=allocation)
        else:
            raise ValueError("Unsupported argument for 'measure'.")
        self.absolute = absolute
        self.fractional = fractional
        self.margin = margin
        self.time = time or datetime.now()

        # Attributes filled when running Broker.rebalance.
        self.profit_on_idle_cash: float = ...
        self.context_pre: "tradingenv.broker.broker.Context" = ...
        self.trades: List[Trade] = ...
        self.context_post: "tradingenv.broker.broker.Context" = ...

    def make_trades(self, broker: "tradingenv.broker.Broker") -> List[Trade]:
        """
        Parameters
        ----------
        broker
            A broker instance that will be used e.g. to calculate the offset
            between target _rebalancing and current holdings.

        Returns
        -------
        A list of trade objects, recommended to perform the portfolio
        _rebalancing.
        """
        imbalance = self.allocation._to_nr_contracts(broker)
        if self.absolute:
            imbalance -= NrContracts(broker.holdings_quantity)
        weights = imbalance._to_weights(broker)
        trades = list()
        for contract, quantity in imbalance.items():
            if not self.fractional:
                # Fractional shares are not supported. Round to smallest digit.
                quantity = int(quantity)
            if abs(weights[contract]) < self.margin and contract in self.allocation:
                # Imbalance weight is smaller than margin. Skip to save costs.
                continue
            trade = Trade(
                time=self.time,
                contract=contract,
                quantity=quantity,
                bid_price=broker.exchange[contract].bid_price,
                ask_price=broker.exchange[contract].ask_price,
                broker_fees=broker.fees,
            )
            trades.append(trade)
        return trades

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.time)
