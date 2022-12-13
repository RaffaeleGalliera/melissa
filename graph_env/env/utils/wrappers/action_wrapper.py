"""
Wrapper to modify actions before step is taken
"""
from pettingzoo.utils.wrappers.base import BaseWrapper


class ActionWrapper(BaseWrapper):
    def step(self, action):
        return super().step(self.action(action))

    def action(self, action):
        """Returns a modified action before `env.step` is called."""
        raise NotImplementedError

    def reverse_action(self, action):
        """Returns a reversed ``action``."""
        raise NotImplementedError
