import gymnasium as gym
import numpy as np
from pettingzoo.utils import wrappers

from .graph import GraphEnv
from .utils.constants import NUMBER_OF_FEATURES
from .utils.core import InfluenceWorld
from .utils.selector import CustomSelector


class InfluenceGraph(GraphEnv):
    """GraphEnv variant that swaps *World* → *InfluenceWorld* **and**
    exposes coarse per‑edge influence classes (-1/0/1/2) as one‑hot edge
    attributes.  Each observation now contains

    * flattened node‑feature matrix  (N × (2+NUMBER_OF_FEATURES+1))
    * flattened integer edge‑class   (N × N)
    * controlling‑agent index (1)
    """

    def __init__(
        self,
        graph=None,
        render_mode=None,
        number_of_agents: int = 10,
        radius: int = 10,
        max_cycles: int = 100,
        device: str = "cuda",
        local_ratio=None,
        scripted_agents_ratio: float = 0.0,
        heuristic=None,
        heuristic_params=None,
        is_testing: bool = False,
        random_graph: bool = False,
        dynamic_graph: bool = False,
        all_agents_source: bool = False,
        num_test_episodes=None,
        model: str = "LT",
    ):
        # rng + misc
        self.seed()
        self.device = device
        self.number_of_agents = number_of_agents
        self.render_mode = render_mode
        self.renderOn = False
        self.radius = radius
        self.local_ratio = local_ratio
        self.is_testing = is_testing
        self.is_new_round = None

        self.world = InfluenceWorld(
            graph=graph,
            number_of_agents=number_of_agents,
            radius=radius,
            np_random=self.np_random,
            heuristic=heuristic,
            heuristic_params=heuristic_params,
            is_testing=is_testing,
            random_graph=random_graph,
            dynamic_graph=dynamic_graph,
            all_agents_source=all_agents_source,
            num_test_episodes=num_test_episodes,
            scripted_agents_ratio=scripted_agents_ratio,
            model=model,
            fixed_interest_density=1.0
        )

        self.agents = [ag.name for ag in self.world.agents]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = {name: i for i, name in enumerate(self.agents)}
        self._agent_selector = CustomSelector(self.agents)

        self.obs_matrix = np.zeros(
            (number_of_agents, 2 + NUMBER_OF_FEATURES + 1), dtype=np.float32
        )
        self.edge_matrix = np.full(
            (number_of_agents, number_of_agents), fill_value=-1, dtype=np.int8
        )

        # flattened observation length
        self._nodes_flat_len = number_of_agents * (2 + NUMBER_OF_FEATURES + 1)
        self._edges_flat_len = number_of_agents * number_of_agents
        obs_dim = self._nodes_flat_len + self._edges_flat_len + 1  # +ctrl idx

        self.action_spaces = {}
        self.observation_spaces = {}
        for ag in self.world.agents:
            self.observation_spaces[ag.name] = gym.spaces.Dict({
                "observation": gym.spaces.Box(
                    low=-1e6, high=1e6, shape=(obs_dim,), dtype=np.float32
                ),
                "action_mask": gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.int8),
            })
            self.action_spaces[ag.name] = gym.spaces.Discrete(2)

        self.state_space = gym.spaces.Box(
            low=-1e6, high=1e6, shape=(obs_dim,), dtype=np.float32
        )

        self.max_cycles = max_cycles
        self.num_moves = 0
        self.current_actions = [None] * number_of_agents
        self.episode_rewards_sum = 0.0

        self.reset()

    def _update_edge_matrix(self):
        """Copy the pre‑computed infl_class from InfluenceWorld."""
        self.edge_matrix[:, :] = self.world.infl_class.astype(np.int8)

    def _init_obs_matrix(self):
        super()._init_obs_matrix()
        self._update_edge_matrix()

    # ------------------------------------------------------------------
    def observe(self, agent: str):
        """Override parent observe: adds flattened edge-matrix while
        retaining every bookkeeping field that the parent fills.
        """
        ctrl_idx = self.agent_name_mapping[agent]

        nodes_flat = self.obs_matrix.reshape(-1)
        edges_flat = self.edge_matrix.reshape(-1).astype(np.float32)
        obs_final = np.concatenate([nodes_flat, edges_flat, [ctrl_idx]]).astype(np.float32)

        action_mask = np.array([1, 1], dtype=np.int8)
        if self.terminations[agent] or self.truncations[agent]:
            action_mask[:] = 0

        self.infos.setdefault(agent, {})
        self.infos[agent]['env_step'] = self.num_moves
        self.infos[agent]['environment_step'] = False
        self.infos[agent]['explicit_reset'] = False

        one_hop_idx = self.world.agents[self.agent_name_mapping[agent]].one_hop_neighbours_ids.copy()
        for idx, _ in enumerate(one_hop_idx):
            if self.world.agents[idx].truncated and (self.world.agents[idx].name not in self.agents):
                one_hop_idx[idx] = 0
        self.infos[agent]['active_one_hop_neighbors'] = one_hop_idx.astype(np.bool_)

        if all(self.terminations.get(a, False) for a in self.agents) and len(self.agents) == 1:
            self.is_new_round = False
            self.infos[agent]['explicit_reset'] = True
        if self.is_new_round:
            self.infos[agent]['environment_step'] = True
            self.is_new_round = False

        return {"observation": obs_final, "action_mask": action_mask}


def make_env(raw_env):
    def _env(**kwargs):
        e = raw_env(**kwargs)
        e = wrappers.AssertOutOfBoundsWrapper(e)
        e = wrappers.OrderEnforcingWrapper(e)
        return e

    return _env


env = make_env(InfluenceGraph)
