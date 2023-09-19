"""The reward is a parameter of TradingEnv. Valid rewards must implement the
abstract class AbstractReward."""
import tradingenv
from abc import ABC, abstractmethod
import numpy as np
from typing import Union


def make_reward(reward: Union["AbstractReward", str]):
    """Valid reward are implementation of the interface AbstractReward. However,
    this method allows the user to specify the reward as a string and the
    corresponding id will be retrieved from tradingenv.rewards."""
    if isinstance(reward, str):
        reward = getattr(tradingenv.rewards, reward)()
    if not isinstance(reward, AbstractReward):
        raise ValueError(
            "{} is an invalid reward. Valid rewards must be objects "
            "implementing tradingenv.rewards.AbstractReward or strings "
            "indicating referring to class _names defined in "
            "tradingenv.rewards."
        )
    return reward


class AbstractReward(ABC):
    """All custom rewards must implement this interface."""

    @abstractmethod
    def calculate(self, env: "tradingenv.env.TradingEnv") -> float:
        """Return float associated with the last reward of the agent,
        generally manipulating stuff from env.broker.track_record. See
        implementations for examples."""

    def reset(self) -> None:
        """Reset values of this class, if any."""


class RewardPnL(AbstractReward):
    """Profit and Loss reward."""

    def calculate(self, env: "tradingenv.env.TradingEnv") -> float:
        nlv_last_rebalancing = env.broker.track_record[-1].context_pre.nlv
        nlv_now = env.broker.net_liquidation_value()
        return float(nlv_now - nlv_last_rebalancing)


class RewardLogReturn(AbstractReward):
    """Log change of the net liquidation value of the account at each step."""

    def calculate(self, env: "tradingenv.env.TradingEnv") -> float:
        nlv_last_rebalancing = env.broker.track_record[-1].context_pre.nlv
        nlv_now = env.broker.net_liquidation_value()
        return float(np.log(nlv_now / nlv_last_rebalancing))


class LogReturn(AbstractReward):
    def __init__(self, scale: float = 1., clip: float = 2.):
        """
        Parameters
        ----------
        scale
            Reward is divided by this number before being returned. This is a
            helper to rescale the reward closer to a [-1, +1] range.
        clip
            Rewards larger than this clip value are truncated.

        Notes
        -----
        For a good reference on why reward shaping is important see [1].

        References
        ----------
        [1] van Hasselt, Hado P., et al. "Learning values across many orders of
        magnitude." Advances in neural information processing systems 29 (2016).
        """
        self.scale = scale
        self.clip = clip

    def calculate(self, env: "tradingenv.env.TradingEnv") -> float:
        nlv_last_rebalancing = env.broker.track_record[-1].context_pre.nlv
        nlv_now = env.broker.net_liquidation_value()
        ret = np.log(nlv_now / nlv_last_rebalancing)
        ret /= self.scale
        ret = np.clip(ret, -self.clip, +self.clip)
        return float(ret)


class RewardSimpleReturn(AbstractReward):
    """Simple change of the net liquidation value of the account at each
    step."""

    def calculate(self, env: "tradingenv.env.TradingEnv") -> float:
        nlv_last_rebalancing = env.broker.track_record[-1].context_pre.nlv
        nlv_now = env.broker.net_liquidation_value()
        return float(nlv_now / nlv_last_rebalancing) - 1


class RewardDifferentialSharpeRatio:
    """An elegant online Sharpe ratio which uses the second order Taylor
    expansion. I still wonder how problematic this approximation might be?

    References
    ----------
    http://papers.nips.cc/paper/1551-reinforcement-learning-for-trading.pdf
    https://quant.stackexchange.com/questions/37969/what-s-the-derivative-of-the-sharpe-ratio-for-one-asset-trying-to-optimize-on-i
    https://www.reddit.com/r/algotrading/comments/9xkvby/how_to_calculate_differential_sharpe_ratio/
    """

    def calculate(self, env: "tradingenv.env.TradingEnv") -> float:
        raise NotImplementedError()
