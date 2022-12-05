import itertools
import logging
import os

import gymnasium
import networkx
import torch
from gymnasium.utils import seeding
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers, agent_selector
import numpy as np
from torch_geometric.utils import from_networkx

from graph_env.env.utils.wrappers.multi_discrete_to_discrete import \
    MultiDiscreteToDiscreteWrapper

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
        self.name = str(id)
        self.state = MprAgentState() if state is None else state
        self.local_view = local_view
        self.geometric_data = from_networkx(local_view)
        self.one_hop_neighbours_ids = None
        self.one_hop_neighbours_neighbours_ids = None
        self.allowed_actions = None
        self.action = None

    def reset(self, local_view):
        self.__init__(id=self.id, local_view=local_view, state=self.state)

    def has_received_from_relayed_node(self):
        return sum([received and relay for received, relay in
                    zip(self.state.received_from, self.state.relays_for)])


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
            local_ratio = None
    ):
        super().__init__()
        self.np_random = None
        self.seed(9)

        self.local_ratio = local_ratio
        self.messages_transmitted = None
        self.radius = radius
        self.graph = networkx.random_geometric_graph(n=number_of_agents, radius=radius)
        self.world_agents = [MprAgent(i, networkx.ego_graph(self.graph, i, undirected=True)) for i in range(number_of_agents)]

        # Needs to be a string for assertions check in tianshou
        self.agents = [agent.name for agent in self.world_agents]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self._agent_selector = agent_selector(self.agents)

        self.max_cycles = max_cycles
        self.steps = 0
        self.render_mode = render_mode
        self.current_actions = [None] * self.num_agents

        self.reset(seed=9)

        actions_dim = np.zeros(NUMBER_OF_AGENTS)
        actions_dim.fill(2)

        state_dim = 0
        for agent in self.world_agents:
            obs_dim = len(self.observe(agent.name)['observation'])

            self.action_spaces[agent.name] = gymnasium.spaces.MultiDiscrete(actions_dim)
            self.observation_spaces[agent.name] = {
                    'observation': gymnasium.spaces.Box(low=0, high=self.num_agents, shape=(obs_dim, )),
                    'action_mask': gymnasium.spaces.Box(low=0, high=1,
                                              shape=(NUMBER_OF_AGENTS,),
                                              dtype=np.int8)
            }

        self.state_space = gymnasium.spaces.Box(
            low=0,
            high=self.num_agents,
            shape=(state_dim, ),
            dtype=np.int8
        )

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
        self.np_random, seed = seeding.np_random(seed)

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
        data[0] = Batch(observation=np.asarray(agent_observation.edge_index))
        data[1] = Batch(observation=np.asarray(agent_observation.label))
        data[2] = Batch(observation=np.asarray(agent_observation.features.to(dtype=torch.float32)))
        data[3] = Batch(observation=np.asarray(agent.id))

        agent_observation_mask = {
            "observation": data,
            "action_mask": agent.allowed_actions
        }

        return agent_observation_mask

    def reset(self, seed=9, return_info=False, options=None):
        if seed is not None:
            self.seed(seed=seed)

        self.agents = self.possible_agents[:]
        self.rewards = {name: 0.0 for name in self.agents}
        self._cumulative_rewards = {name: 0.0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}
        self.messages_transmitted = 0
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
            agent.one_hop_neighbours_ids = self.graph.nodes[agent.id]['features']

            # Calc every combination of the agent's neighbours to get allowed actions
            neighbours_combinations = list(itertools.product([0, 1], repeat=int(
                sum(agent.one_hop_neighbours_ids))))
            indices = [i for i, x in enumerate(agent.one_hop_neighbours_ids) if x == 1]

            actions_dim = np.zeros(NUMBER_OF_AGENTS)
            actions_dim.fill(2)
            allowed_actions_mask = [False] * int(np.prod(actions_dim))
            for element in neighbours_combinations:
                allowed_action_binary = [0] * NUMBER_OF_AGENTS
                for i in range(len(element)):
                    allowed_action_binary[indices[i]] = element[i]

                res = int("".join(str(x) for x in allowed_action_binary), 2)
                allowed_actions_mask[res] = True

            agent.allowed_actions = allowed_actions_mask

        self.action_spaces = {agent: gymnasium.spaces.Discrete(2 ** self.num_agents) for agent in self.agents}

        self.observation_spaces = {
            agent: {
                'observation': gymnasium.spaces.Box(low=0, high=1, shape=((len(self.observe(agent)['observation'])), )),
                'action_mask': gymnasium.spaces.Box(low=0, high=1,
                                                    shape=(NUMBER_OF_AGENTS,),
                                                    dtype=np.int8)
            }
            for agent in self.agents
        }

        random_agent = self.np_random.choice(self.world_agents)
        random_agent.state.message_origin = 1

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
            self._execute_world_step()
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

        n_received = sum([1 for agent in self.world_agents if
                          sum(agent.state.received_from) or agent.state.message_origin])
        if n_received == NUMBER_OF_AGENTS:
            logging.debug(
                f"Every agent has received the message, terminating in {self.steps}, messages transmitted: {self.messages_transmitted}")
            for agent in self.agents:
                self.terminations[agent] = True

    def _execute_world_step(self):
        for i, agent in enumerate(self.world_agents):
            action = self.current_actions[i]
            scenario_action = []
            scenario_action.append(action)
            self._set_action(scenario_action, agent, self.action_spaces[agent.name])
        # Step the world
        # Set MPRs
        for agent in self.world_agents:
            self.set_relays(agent)

        # Send message
        for agent in self.world_agents:
            logging.debug(f"Agent {agent.name} Action: {agent.action} with Neigh: {agent.one_hop_neighbours_ids}")
            self.update_agent_state(agent)

        global_reward = 0.0
        # TODO: exp with local ratio
        if self.local_ratio is not None:
            pass
            # global_reward = float(self.global_reward())

        for agent in self.world_agents:
            agent_reward = float(self.reward(agent))
            if self.local_ratio is not None:
                reward = (
                    global_reward * (1 - self.local_ratio)
                    + agent_reward * self.local_ratio
                )
            else:
                reward = agent_reward

            self.rewards[agent.name] = reward

    def _set_action(self, action, agent, param):
        agent.action = np.zeros((self.num_agents,))
        agent.action = action[0]
        action = action[1:]
        assert len(action) == 0

    def set_relays(self, agent):
        if agent.action is not None:
            agent.state.relayed_by = agent.action
            neighbour_indices = [i for i, x in
                                 enumerate(agent.one_hop_neighbours_ids) if
                                 x == 1]
            relay_indices = [i for i, x in enumerate(agent.state.relayed_by)
                             if x == 1]
            # Assert relays are subset of one hope neighbours of the agent
            assert (set(relay_indices) <= set(neighbour_indices))
            for index, value in enumerate(agent.state.relayed_by):
                self.world_agents[index].state.relays_for[agent.id] = value

    def update_agent_state(self, agent):
        # if it has received from a relay node or is message origin and has not already transmitted the message
        if (agent.has_received_from_relayed_node() or agent.state.message_origin) and not sum(agent.state.transmitted_to):
            logging.debug(f"Agent {agent.name} sending to Neighs: {agent.one_hop_neighbours_ids}")

            agent.state.transmitted_to = agent.one_hop_neighbours_ids
            self.messages_transmitted += 1
            neighbour_indices = [i for i, x in enumerate(agent.one_hop_neighbours_ids) if x == 1]
            for index in neighbour_indices:
                self.world_agents[index].state.received_from[agent.id] = 1
        else:
            logging.debug(f"Agent {agent.name} could not send")

    def reward(self, agent):
        accumulated = 0
        for other_agent in self.world_agents:
            if other_agent is agent:
                continue
            if sum(other_agent.state.received_from):
                accumulated += 1
        completion = accumulated / len(self.world_agents) if accumulated > 0 else 0
        logging.debug(f"Agent {agent.name} received : {- 1 + completion}")
        return (- 1 + completion) * self.messages_transmitted


def make_env(raw_env):
    def env(**kwargs):
        env = raw_env(**kwargs)
        env = MultiDiscreteToDiscreteWrapper(env)
        env = wrappers.AssertOutOfBoundsWrapper(env)
        env = wrappers.OrderEnforcingWrapper(env)
        return env

    return env


env = make_env(GraphEnv)
