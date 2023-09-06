"""This module provides the core logic to perform portfolio _rebalancing.
Holdings (cash and contracts) are held in a Brokerage account and historical
_rebalancing are keep in TrackRecord."""
from tradingenv.broker.trade import Trade
from tradingenv.broker.rebalancing import Rebalancing
from tradingenv.broker.fees import IBrokerFees, BrokerFees
from tradingenv.broker.track_record import TrackRecord
from tradingenv.contracts import AbstractContract, Cash
from tradingenv.exchange import Exchange
from typing import Dict, Union, Sequence
from datetime import datetime
from collections import defaultdict
import numpy as np
SECONDS_IN_YEAR = 365 * 24 * 60 * 60


class EndOfEpisodeError(Exception):
    """Raised when the net liquidation state of the account goes below zero.
    This exception will trigger an end of episode."""


class Context:
    def __init__(self, nlv, weights, values, nr_contracts,
                 margins):
        self.nlv = nlv
        self.weights = weights
        self.values = values
        self.nr_contracts = nr_contracts
        self.margins = margins

    def __eq__(self, other: 'Context') -> bool:
        return all(
            self.nlv == other.nlv and
            self.weights == other.weights and
            self.values == other.values and
            self.nr_contracts and other.nr_contracts and
            self.margins and other.margins
        )


class Broker:
    """This class allows to simulate a sequence of _rebalancing requests.

    Attributes
    ----------
    _holdings_margins : Dict[AbstractContract, float]
        A dictionary mapping contracts to margins deposited at the clearing
        house, e.g. when buying derivatives products such as futures.
    _holdings_quantity : Dict[AbstractContract, float]
        A dictionary mapping contracts to number of contracts held now.
    track_record : TrackRecord
        Keeps track of historical _rebalancing request.
    """

    def __init__(
        self,
        exchange: Exchange,
        base_currency: Cash = Cash(),
        deposit: float = 100.0,
        fees: IBrokerFees = BrokerFees(),
        epsilon: float = 1e-7
    ):
        """
        Parameters
        ----------
        exchange : Exchange
            An exchanged used to retrieve market prices and conditions to
            execute _rebalancing requests and trade accordingly.
        base_currency: Cash
            Base currency.
        deposit : float
            Initial deposit, 100 units of base currency by default.
        """
        # Inputs.
        self.exchange = exchange
        self.base_currency = base_currency
        self.fees = fees
        self._epsilon = epsilon

        # More attributes.
        self._holdings_margins: Dict[AbstractContract, float] = defaultdict(float)
        self._holdings_quantity: Dict[AbstractContract, float] = defaultdict(float)
        self._holdings_quantity[base_currency] = deposit
        self._initial_deposit = deposit
        self._last_accrual: Union[datetime, None] = None
        self._last_marking_to_market_price = dict()
        self.track_record = TrackRecord()
        
    @property
    def holdings_quantity(self) -> Dict[AbstractContract, float]:
        return dict(self._holdings_quantity)

    @property
    def holdings_margins(self) -> Dict[AbstractContract, float]:
        return dict(self._holdings_margins)

    def rebalance(self, rebalancing: Rebalancing):
        """Given a _rebalancing object, execute all trades needed to perform the
        _rebalancing.

        Developer notes
        ---------------
        The _rebalancing logic is slit between Rebalancing and Broker.
        Rebalancing is responsible for calculating (but not executing) the
        trading instructions (i.e. a list of trades), wheres Broker is
        responsible for anything else (i.e. pre-processing, execution and
        post-processing).
        This design allows to optionally compute the trades for a variety of
        _rebalancing objects and see what the trades would be, without
        performing any _rebalancing. Also, the separation of concerns between
        calculating the trades and execution the trades makes it easier to
        test the code.
        """
        rebalancing.profit_on_idle_cash = self.accrued_interest(rebalancing.time, True)
        rebalancing.context_pre = self.context()
        rebalancing.trades = rebalancing.make_trades(self)
        for trade in rebalancing.trades:
            self.transact(trade)
        rebalancing.context_post = self.context()
        self.track_record._checkpoint(rebalancing)

    def transact(self, trade: Trade):
        """
        Parameters
        ----------
        trade : Trade
            A trade instance to be executed.

        Notes
        -----
        Transacting in long lasting positions might lead to numerical errors in
        the order of ~1e-15. It's a design choice not to deal with such errors
        because for prices --> +inf, _holdings_quantity --> 0 < 1e-15. In other
        words, we don't deal with such numerical errors to avoid unexpected
        side effects when asset prices are relatively too high.
        https://floating-point-gui.de/basic/
        """
        # Marking to market here before transacting to avoid to update the
        # last marking to market price used to price the position before
        # overriding with the acquisition price at the end of this code.
        self.marking_to_market(trade.contract)

        # Calculate target _margin requirements. Note that trade will decrease
        # margins to be held whenever the exposure of the position is reduced.
        target_quantity = abs(self._holdings_quantity[trade.contract] + trade.quantity)
        margin_expected = (
            trade.acq_price
            * target_quantity
            * trade.contract.multiplier
            * trade.contract.margin_requirement
        )
        margin_actual = self._holdings_margins[trade.contract]
        margin_diff = margin_expected - margin_actual

        # Pay transaction costs.
        self._holdings_quantity[self.base_currency] -= trade.cost_of_commissions

        # Acquisition.
        self._holdings_quantity[self.base_currency] -= trade.cost_of_cash
        self._holdings_quantity[self.base_currency] -= margin_diff
        self._holdings_margins[trade.contract] += margin_diff
        self._holdings_quantity[trade.contract] += trade.quantity

        # https://stackoverflow.com/questions/588004/is-floating-point-math-broken
        # Small quantities different from zero will make nlv to retrieve
        # prices from the exchange of expired contracts, resulting in errors as:
        # ValueError: Missing liquidation transaction_price for VX(VXN04).
        # This will make also _holdings_quantity more clean, ie no numerical
        # errors. The drawback are bugs when the actual target quantity is
        # 1e-15, occuring when nlv is close to zero and huge asset price.
        if abs(self._holdings_quantity[trade.contract]) < self._epsilon:
            self._holdings_quantity[trade.contract] = 0.

        # Update _margin requirements. Bid-ask spread is implicitly paid
        # here and now.
        self._last_marking_to_market_price[trade.contract] = trade.acq_price
        self.marking_to_market(trade.contract)

    def marking_to_market(
        self, contract: Union[AbstractContract, Sequence[AbstractContract]] = None
    ):
        """
        Parameters
        ----------
        contract : Union[AbstractContract, Sequence[AbstractContract]]
            An optional contract or sequence of contracts to for which
            marking-to-market has to be performed. If blank, all contracts will
            be assumed to be updated.

        References
        ----------
        https://www.cmegroup.com/education/courses/introduction-to-futures/mark-to-market.html
        https://www.cmegroup.com/education/courses/introduction-to-futures/margin-know-what-is-needed.html
        """
        # NOTE: this might be a performance bottleneck.
        if contract is None:
            contracts = list(self._holdings_margins)
        else:
            contracts = [contract]
        for contract in contracts:
            # Vary the _margin to reflect gains or losses.
            if contract.margin_requirement == 0:
                continue
            quantity = self._holdings_quantity[contract]
            liq_price = self.exchange[contract].liq_price(quantity)
            if np.isnan(liq_price):
                continue
            try:
                last_price = self._last_marking_to_market_price[contract]
            except KeyError:
                continue
            price_change = liq_price - last_price
            profit = quantity * contract.multiplier * price_change
            self._holdings_margins[contract] += profit
            self._last_marking_to_market_price[contract] = liq_price

            # Withdraw excess _margin or deposit if _margin call.
            current_margin = self._holdings_margins[contract]
            target_margin = (
                liq_price * abs(quantity) * contract.multiplier * contract.margin_requirement
            )
            excess_margin = current_margin - target_margin
            self._holdings_margins[contract] -= excess_margin
            self._holdings_quantity[self.base_currency] += excess_margin
            if self._holdings_margins[contract] < 0:
                raise ValueError(
                    "Unexpected situation during sanity check. "
                    "Margin for {} is negative: {}"
                    "".format(contract, self._holdings_margins[contract])
                )

    def holdings_values(self, kind: str = "notional") -> Dict[AbstractContract, float]:
        """Returns np.array with state of all positions within the portfolio
        (cash included) in base currency. The array has as many entries as
        traded contracts.
        Notional values, so transaction_price * multiplier."""
        # Note: when quantity is zero, value is trivially zero. Skip instead?
        holdings_values = defaultdict(float)
        for contract, quantity in self._holdings_quantity.items():
            if quantity == 0:
                value = 0.0
            else:
                order_book = self.exchange[contract]
                liq_price = (
                    order_book.bid_price if quantity >= 0 else order_book.ask_price
                )
                if np.isnan(liq_price):
                    raise ValueError(
                        "Missing liquidation transaction_price for {}.".format(contract)
                    )
                if kind == "notional":
                    value = quantity * liq_price * contract.multiplier
                elif kind == "liquidation":
                    value = contract.cash_requirement * quantity * liq_price
                    value += self._holdings_margins[contract]
                else:
                    raise ValueError("Unsupported 'kind'.")
            holdings_values[contract] = value
        return holdings_values

    def holdings_weights(self) -> Dict[AbstractContract, float]:
        """Relative weights of the notional traded value over NLV.
        Returns current weights of the portfolio as positions values
        (including market spread) divided by the net liquidation state of the
        account (including market spread)."""
        nlv = self.net_liquidation_value()
        holdings_notional_values = self.holdings_values()
        return {
            contract: value / nlv
            for contract, value in holdings_notional_values.items()
        }

    def net_liquidation_value(self, raise_if_broke: bool = True) -> float:
        """
        Parameters
        ----------
        raise_if_broke : bool
            An EndOfEpisodeError will be raised if the notional liquidation
            value of the account is non-positive.

        Returns
        -------
        The net liquidation value of the account pricing long (short) position
        using bid (ask) prices.
        """
        self.marking_to_market()
        holdings_values = self.holdings_values(kind="liquidation")
        nlv = sum(holdings_values.values())
        if raise_if_broke and nlv <= 0:
            raise EndOfEpisodeError(
                "Net liquidation state of the account is ${}".format(nlv)
            )
        return nlv

    def accrued_interest(self, now: datetime, accrue: bool = False) -> float:
        """
        Parameters
        ----------
        now : datetime
            Current time, used to calculate how long it has passed since the
            last time that interest rates were deposited or charged.
        accrue : bool
            If True, the interest rate will be accrued, i.e. charged if
            negative or deposited if positive. The interest rate applied is
            determined by Broker.fees.interest_rate.

        Notes
        -----
        No interest rate paid on margins. This is a realistic conservative
        assumption (e.g. Interactive Brokers does not pay interest rates on
        margins and financial instrument cannot be deposited as _margin).

        Returns
        -------
        Amount of interest rate owed (either positive or negative) since
        when if was last deposited/charged from the account's holdings.
        """
        if self._last_accrual is None:
            self._last_accrual = now
        if now < self._last_accrual:
            raise ValueError("now={} < last_update={}".format(now, self._last_accrual))
        years = (now - self._last_accrual).total_seconds() / SECONDS_IN_YEAR
        order_book = self.exchange[self.fees.interest_rate]
        amount = self._holdings_quantity[self.base_currency]

        # Compounded annual growth rate earned (paid) on idle (borrowed) cash.
        cagr = order_book.mid_price - self.fees.markup * np.sign(amount)
        rate_period = (1 + cagr) ** years - 1
        accrued_interest = amount * rate_period
        if amount > 0. and accrued_interest < 0.:
            accrued_interest = 0.
        if accrue:
            self._holdings_quantity[self.base_currency] += accrued_interest
            self._last_accrual = now
        return accrued_interest

    def context(self) -> Context:
        return Context(
            nlv=self.net_liquidation_value(),
            weights=self.holdings_weights(),
            values=self.holdings_values(),
            nr_contracts=self.holdings_quantity,
            margins=self.holdings_margins,
        )
