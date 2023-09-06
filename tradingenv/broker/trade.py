"""Execution details of a single trade."""
from datetime import datetime
from tradingenv.contracts import AbstractContract, Cash
from tradingenv.broker.fees import IBrokerFees, BrokerFees
import numpy as np


class Trade:
    """Execution details of a single trade. This class is instanced by
    RebalancingResponse.

    Attributes
    ----------
    acq_price : float
        Average acquisition or liquidation price across all lots to execute
        the trade. By default, there is no market impact nor slippage.
        Therefore, avg_price corresponds to the bid price (ask price) in case
        of buy (sell).
    cost_of_cash : float
        Cash to be paid (earned) upfront to buy (sell) the contract.
    cost_of_commissions : float
        Broker commissions paid when transacting.
    """

    __slots__ = (
        "time",
        "contract",
        "quantity",
        "bid_price",
        "ask_price",
        "notional",
        "acq_price",
        "cost_of_cash",
        "cost_of_commissions",
        "cost_of_spread",
    )

    def __init__(
        self,
        time: datetime,
        contract: AbstractContract,
        quantity: float,
        bid_price: float,
        ask_price: float,
        broker_fees: "IBrokerFees" = BrokerFees(),
    ):
        """
        Parameters
        ----------
        time : datetime
            Execution time of the trade.
        contract : AbstractContract
            Traded contract.
        quantity : float
            Traded quantity. A negative number denotes a sell, liquidation or
            short-selling.
        bid_price : float
            Bid price in the limit order book of the contract. Bid price must
            be non-greater then the ask price.
        ask_price : float
            Ask price in the limit order book of the contract. Ask price must
            be non.smaller than the bid price.
        broker_fees : IBrokerFees
            An concrete implementation of AbstractBrokerFees,
            responsible to calculate the total broker fees to be paid
            for the trade.
        """
        if np.isnan(bid_price):
            raise ValueError("Missing bid price for contract {}.".format(contract))
        if np.isnan(ask_price):
            raise ValueError("Missing ask price for contract {}.".format(contract))
        if np.isnan(quantity):
            raise ValueError("Missing quantity for contract {}.".format(contract))
        if quantity == 0:
            raise ValueError("Quantity for contract {} is zero.".format(contract))
        if isinstance(contract, Cash):
            raise ValueError("Contract Cash cannot be traded.")
        self.time = time
        self.contract = contract
        self.quantity = quantity
        self.bid_price = bid_price
        self.ask_price = ask_price
        self.acq_price = ask_price if quantity > 0 else bid_price
        self.notional = self.acq_price * quantity * contract.multiplier
        self.cost_of_cash = self.notional * contract.cash_requirement
        self.cost_of_commissions = broker_fees.commissions(self)
        self.cost_of_spread = (
            abs(quantity) * contract.multiplier * (ask_price - bid_price)
        )

    def __repr__(self) -> str:
        """e.g. Trade(1991-11-05 00:00:00; Buy 340 ETF(SPY) at 282.16)"""
        signal = "Buy" if self.quantity > 0 else "Sell"
        return "{}({}; {} {} {} at {})" "".format(
            self.__class__.__name__,
            self.time,
            signal,
            abs(self.quantity),
            self.contract,
            self.acq_price,
        )

    def __eq__(self, other: 'Trade'):
        return all([
            self.time == other.time,
            self.contract == other.contract,
            np.isclose(self.quantity, other.quantity),
            self.acq_price == other.acq_price
        ])
