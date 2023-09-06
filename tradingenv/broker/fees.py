import tradingenv
from tradingenv.contracts import Rate
from abc import ABC, abstractmethod


class IBrokerFees(ABC):
    def __init__(
        self,
        markup: float = 0.0,
        interest_rate: Rate = Rate("FED funds rate"),
        proportional: float = 0.0,
        fixed: float = 0.0,
    ):
        """
        Parameters
        ----------
        markup : float
            A non-negative CAGR, zero by default. This mark-up is the profit
            of the broker when applying the interest rate to idle cash and
            loans. For example, if interest_rate is 0.03 (3%) and fees=0.01
            (1%), CAGR on idle cash will be 2% and cost of loans will be 4%.
        interest_rate : Rate
            This interest rate is used to: (1) generate a positive stream of
            cash flows on idle cash (e.g. when the portfolio is not fully
            invested) and (2) generate a negative stream of cash flows on
            loans (e.g. borrowed cash to pay in a leveraged portfolio of ETFs).
            By default, this is Rate('FED funds rate') and it's expressed in
            terms of compounded annual growth rate (CAGR).
        proportional : float
            Broker fees to be applied to the notional traded value of the
            security. E.g. if 0.01 (1%) mean that 1 USD is paid for every
            100 USD traded.
        fixed : float
            Fixed broker fees to be applied to each trade, regardless the
            notional traded value of trade.
        """
        self.markup = markup
        self.interest_rate = interest_rate
        self.proportional = proportional
        self.fixed = fixed

    @abstractmethod
    def commissions(self, trade: "tradingenv.broker.trade.Trade") -> float:
        """Returns total broker fees."""


class BrokerFees(IBrokerFees):
    """No broker fees of any sort will be applied when using this class."""

    def commissions(self, trade: "tradingenv.broker.trade.Trade") -> float:
        """Zero fees."""
        return self.fixed + abs(trade.notional) * self.proportional


class InteractiveBrokersFees(IBrokerFees):
    """Replicate Interactive Broker commisions.

    References
    ----------
    https://www.interactivebrokers.com/en/index.php?f=1590
    """

    def commissions(self, trade: "tradingenv.broker.trade.Trade") -> float:
        raise NotImplementedError()
