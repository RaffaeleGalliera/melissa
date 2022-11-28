"""
This wrapper is needed to convert MultiDiscrete to Discrete actions due 
to compatibility issues with Tianshou
"""

from gymnasium.spaces import Discrete, MultiDiscrete
import gym
from .action_wrapper import ActionWrapper
import numpy as np


class MultiDiscreteToDiscreteWrapper(ActionWrapper):
    """Asserts if the action given to step is outside of the action space. Applied in PettingZoo environments with discrete action spaces."""

    def __init__(self, env):
        super().__init__(env)
        assert all(
            isinstance(self.action_space(agent), MultiDiscrete)
            for agent in getattr(self, "possible_agents", [])
        ), "should only use MultiDiscreteToDiscreteWrapper for MultiDiscrete spaces"
        for agent in self.possible_agents:
            nvec = self.action_space(agent).nvec
            assert nvec.ndim == 1
            self.bases = np.ones_like(nvec)
            for i in range(1, len(self.bases)):
                self.bases[i] = self.bases[i - 1] * nvec[-i]
            self.env.action_spaces[agent] = Discrete(np.prod(nvec))

    def step(self, action):
        super().step(action)

    def action(self, act: np.ndarray) -> np.ndarray:
        converted_act = []
        for b in np.flip(self.bases):
            converted_act.append(act // b)
            act = act % b

        return np.array(converted_act).transpose()
