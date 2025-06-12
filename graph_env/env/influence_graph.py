import gymnasium
import numpy as np

from .utils.constants import NUMBER_OF_FEATURES
from .utils.core import InfluenceWorld
from .graph import GraphEnv
from .utils.selector import CustomSelector
from pettingzoo.utils import wrappers


class InfluenceGraph(GraphEnv):
    """
    A GraphEnv variant that uses InfluenceWorld under the hood.
    Minimal changes: swaps World → InfluenceWorld and exposes `model`.
    """

    def __init__(
            self,
            graph=None,
            render_mode=None,
            number_of_agents=10,
            radius=10,
            max_cycles=100,
            device='cuda',
            local_ratio=None,
            scripted_agents_ratio=0.0,
            heuristic=None,
            heuristic_params=None,
            is_testing=False,
            random_graph=False,
            dynamic_graph=False,
            all_agents_source=False,
            num_test_episodes=None,
            model="LT"
    ):
        self.seed()
        self.device = device
        self.number_of_agents = number_of_agents
        self.render_mode = render_mode
        self.renderOn = False
        self.local_ratio = local_ratio
        self.radius = radius
        self.is_new_round = None
        self.is_testing = is_testing

        self.world = InfluenceWorld(
            graph=graph,
            number_of_agents=self.number_of_agents,
            radius=radius,
            np_random=self.np_random,
            heuristic=heuristic,
            heuristic_params=heuristic_params,
            is_testing=is_testing,
            random_graph=random_graph,
            dynamic_graph=False, # TODO: dynamic_graph is not used in InfluenceWorld at the moment
            all_agents_source=all_agents_source,
            num_test_episodes=num_test_episodes,
            scripted_agents_ratio=scripted_agents_ratio,
            model=model
            # fixed_interest_density=0.5
        )

        self.agents = [agent.name for agent in self.world.agents]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(
            zip(self.agents, list(range(len(self.possible_agents))))
        )
        self._agent_selector = CustomSelector(self.agents)

        # We'll store a shared observation matrix updated each step
        # shape (number_of_agents, 2 + NUMBER_OF_FEATURES)
        self.obs_matrix = np.zeros(
            (self.number_of_agents, 2 + NUMBER_OF_FEATURES + 1), dtype=np.float32
        )

        # Flattened dimension = N*(2+NUMBER_OF_FEATURES) + 1 controlling-agent-index
        obs_dim = self.number_of_agents * (2 + NUMBER_OF_FEATURES + 1) + 1

        self.action_spaces = {}
        self.observation_spaces = {}
        for agent in self.world.agents:
            self.observation_spaces[agent.name] = gymnasium.spaces.Dict({
                'observation': gymnasium.spaces.Box(
                    low=-1e6,
                    high=1e6,
                    shape=(obs_dim,),
                    dtype=np.float32,
                ),
                'action_mask': gymnasium.spaces.Box(
                    low=0,
                    high=1,
                    shape=(2,),
                    dtype=np.int8,
                ),
            })
            self.action_spaces[agent.name] = gymnasium.spaces.Discrete(2)

        self.state_space = gymnasium.spaces.Box(
            low=-1e6,
            high=1e6,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.max_cycles = max_cycles
        self.num_moves = 0
        self.current_actions = [None] * self.number_of_agents
        self.episode_rewards_sum = 0.0

        self.reset()

def make_env(raw_env):
    def env(**kwargs):
        env_ = raw_env(**kwargs)
        env_ = wrappers.AssertOutOfBoundsWrapper(env_)
        env_ = wrappers.OrderEnforcingWrapper(env_)
        return env_
    return env

env = make_env(GraphEnv)
