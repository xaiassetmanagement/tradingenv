# https://github.com/robertmartin8/PyPortfolioOpt
# https://quant.stackexchange.com/questions/tagged/portfolio-optimization?sort=votes&pageSize=15
# Fractional differentiation.
# Kelly (optimal f)
# RiskParity with custom definition of risk, eg Martin risk.
# Use OAS for robust covariance matrix estimation:
#    https://scikit-learn.org/stable/modules/generated/sklearn.covariance.OAS.html
"""Logic to wrap policy passed in TradingEnv.sample_episode(policy)."""
from tradingenv.spaces import PortfolioSpace
from abc import ABC, abstractmethod
import gymnasium.spaces


class AbstractPolicy(ABC):
    # Class attributes are filled by make_policy()
    action_space: PortfolioSpace = None
    observation_space: gymnasium.Space = None

    @abstractmethod
    def act(self, state):
        """Returns an action which belongs to the action observation_space, so
        action_space.contains(action) must return True."""

    def __repr__(self):
        return self.__class__.__name__


class RandomPolicy(AbstractPolicy):
    def act(self, state=None):
        return self.action_space.sample()


def make_policy(policy, action_space, observation_space) -> AbstractPolicy:
    if policy is None:
        policy = RandomPolicy()
    elif isinstance(policy, AbstractPolicy):
        pass
    else:
        try:
            from ray.rllib.policy.policy import Policy
            from ray.rllib.agents.trainer import Trainer
        except ModuleNotFoundError:
            pass
        else:
            from tradingenv.ray.walkforward.policy import RayAgent, RayPolicy
            if isinstance(policy, Trainer):
                policy = RayAgent(policy)
            elif isinstance(policy, Policy):
                policy = RayPolicy(policy)
    # Inject action and observation space.
    policy.action_space = action_space
    policy.observation_space = observation_space
    return policy
