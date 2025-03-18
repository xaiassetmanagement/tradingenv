import gymnasium
import gymnasium.utils.seeding
from tradingenv.env import TradingEnv
from tradingenv.spaces import DiscretePortfolio
from tradingenv.contracts import Cash, ETF
from tradingenv.transmitter import AsynchronousTransmitter
import unittest.mock


class TestTradingEnv:
    def get_discrete_env(self, **kwargs):
        env = TradingEnv(
            action_space=DiscretePortfolio(
                allocations=[
                    [0, 0, 1],
                    [0, 0.25, 0.75],
                    [0, 0.5, 0.5],
                    [0, 0.75, 0.25],
                    [0, 1, 0],
                ],
                contracts=[Cash(), ETF("Equities"), ETF("Bonds")],
            ),
            **kwargs,
        )
        return env

    def test_is_subclass_of_gym_env(self):
        assert issubclass(TradingEnv, gymnasium.Env)

    # def test_render_modes_are_specified(self):
    #     assert len(TradingEnv.metadata["render.modes"]) > 0

    @unittest.mock.patch.object(AsynchronousTransmitter, "_create_partitions")
    def test_transmitter_create_partitions_is_called(self, create_partitions):
        self.get_discrete_env()
        create_partitions.assert_called_once()
