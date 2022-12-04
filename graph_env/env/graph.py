import os

import gymnasium
import networkx
import torch
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers, agent_selector
import numpy as np
from torch_geometric.utils import from_networkx

NUMBER_OF_AGENTS = 10

from tianshou.data.batch import Batch

class MprAgentState():
    def __init__(self):
        self.received_from = None
        self.transmitted_to = None
        self.relays_for = None
        self.relayed_by = None
        self.message_origin = 0


class MprAgent():
    def __init__(self, id, local_view, state=None):
        # state
        self.id = id
        self.state = MprAgentState() if state is None else state
        self.local_view = local_view
        self.geometric_data = from_networkx(local_view)
        self.one_hop_neighbours_ids = None
        self.one_hop_neighbours_neighbours_ids = None
        self.allowed_actions = None

    def reset(self, local_view):
        self.__init__(id=self.id, local_view=local_view, state=self.state)


class GraphEnv(AECEnv):
    metadata = {
        'render_modes': ["human"],
        'name': "graph_environment",
        'is_parallelizable': True
    }

    def __init__(
            self,
            number_of_agents=NUMBER_OF_AGENTS,
            radius=0.40,
            max_cycles=50,
            render_mode=None,
    ):
        super().__init__()
        self.radius = radius
        self.graph = networkx.random_geometric_graph(n=number_of_agents, radius=radius)
        self.world_agents = [MprAgent(i, networkx.ego_graph(self.graph, i, undirected=True)) for i in range(number_of_agents)]

        # Needs to be a string for assertions check in tianshou
        self.agents = [str(agent.id) for agent in self.world_agents]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self._agent_selector = agent_selector(self.agents)

        self.max_cycles = max_cycles
        self.steps = 0
        self.render_mode = render_mode
        self.current_actions = [None] * self.num_agents

        self.reset()

        self.action_spaces = {agent: gymnasium.spaces.Discrete(1024) for agent in self.agents}

        self.observation_spaces = {
            agent: {
                'observation': gymnasium.spaces.Box(low=0, high=1, shape=((len(self.observe(agent)['observation'])), ))
            }
            for agent in self.agents
        }

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render without specifying any render mode."
            )
            return

        pass

    def close(self):
        pass

    def observation_space(self, agent) -> gymnasium.spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent) -> gymnasium.spaces.Space:
        return self.action_spaces[agent]

    def seed(self, seed=None):
        pass

    def observe(self, agent: str):
        return self.observation(
            self.world_agents[self.agent_name_mapping[agent]]
        )

    def state(self):
        states = tuple(
            self.observation(
                self.world_agents[self.agent_name_mapping[agent]]
            )
            for agent in self.possible_agents
        )

        return np.concatenate(states, axis=None)

    def observation(self, agent):
        agent_observation = agent.geometric_data

        data = Batch(observation=np.zeros([4], object))
        data[0] = Batch(observation=agent_observation.edge_index)
        data[1] = Batch(observation=np.asarray(agent_observation.label))
        data[2] = Batch(observation=agent_observation.features.to(dtype=torch.float32))
        data[3] = Batch(observation=np.asarray(agent.id))

        return data

    def reset(self, seed=None, return_info=False, options=None):
        if seed is not None:
            pass

        self.agents = self.possible_agents[:]
        self.rewards = {name: 0.0 for name in self.agents}
        self._cumulative_rewards = {name: 0.0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        self.agent_selection = self._agent_selector.reset()
        self.steps = 0

        self.current_actions = [None] * self.num_agents

        self.graph = networkx.random_geometric_graph(n=self.num_agents, radius=self.radius)

        for agent in self.world_agents:
            agent.state.received_from = np.zeros(self.num_agents)
            agent.state.transmitted_to = np.zeros(self.num_agents)
            agent.state.relays_for = np.zeros(self.num_agents)
            agent.state.relayed_by = np.zeros(self.num_agents)
            agent.state.message_origin = 0

            one_hop_neighbours_ids = np.zeros(self.num_agents)
            for agent_index in self.graph.neighbors(agent.id):
                one_hop_neighbours_ids[agent_index] = 1
            self.graph.nodes[agent.id]['features'] = one_hop_neighbours_ids
            self.graph.nodes[agent.id]['label'] = agent.id

        for agent in self.world_agents:
            agent.reset(local_view=networkx.ego_graph(self.graph, agent.id, undirected=True))

        self.action_spaces = {agent: gymnasium.spaces.Discrete(2 ** self.num_agents) for agent in self.agents}

        self.observation_spaces = {
            agent: {
                'observation': gymnasium.spaces.Box(low=0, high=1, shape=((len(self.observe(agent)['observation'])), ))
            }
            for agent in self.agents
        }

    def step(self, action):
        if(
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        current_agent = self.agent_selection
        current_idx = self.agent_name_mapping[self.agent_selection]
        next_idx = (current_idx + 1) % self.num_agents
        self.agent_selection = self._agent_selector.next()

        self.current_actions[current_idx] = action

        if next_idx == 0:
            # self._execute_world_step()
            self.steps += 1
            if self.steps >= self.max_cycles:
                for a in self.agents:
                    self.truncations[a] = True
        else:
            self._clear_rewards()

        self._cumulative_rewards[current_agent] = 0
        self._accumulate_rewards()

        if self.render_mode == 'human':
            self.render()

        # n_received = sum([1 for agent in self.world.agents if
        #                   sum(agent.state.received_from) or agent.state.message_origin])
        # if n_received == NUMBER_OF_AGENTS:
        #     logging.debug(
        #         f"Every agent has received the message, terminating in {self.steps}, messages transmitted: {self.world.messages_transmitted}")
        #     for agent in self.agents:
        #         self.terminations[agent] = True


def make_env(raw_env):
    def env(**kwargs):
        env = raw_env(**kwargs)
        # env = MultiDiscreteToDiscreteWrapper(env)
        env = wrappers.AssertOutOfBoundsWrapper(env)
        env = wrappers.OrderEnforcingWrapper(env)
        return env

    return env


env = make_env(GraphEnv)
