import functools
import logging

import gymnasium
import networkx as nx
from gymnasium.utils import seeding
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
import numpy as np
from tianshou.data.batch import Batch
from .utils.constants import NUMBER_OF_AGENTS, NUMBER_OF_FEATURES, RENDER_PAUSE
from .utils.core import World
from .utils.selector import CustomSelector
from torch_geometric.utils import from_networkx

import matplotlib.pyplot as plt
import math


class GraphEnv(AECEnv):
    metadata = {
        'render_modes': ["human"],
        'name': "graph_environment",
        'is_parallelizable': False
    }

    def __init__(
            self,
            graph=None,
            render_mode=None,
            number_of_agents=10,
            radius=10,
            max_cycles=100,
            device='cuda',
            local_ratio=None,
            is_scripted=False,
            is_testing=False,
            random_graph=False,
            dynamic_graph=False
    ):
        super().__init__()
        self.seed()
        self.device = device

        self.render_mode = render_mode
        self.renderOn = False
        self.local_ratio = local_ratio
        self.radius = radius
        self.is_new_round = None

        self.world = World(graph=graph,
                           number_of_agents=number_of_agents,
                           radius=radius,
                           np_random=self.np_random,
                           is_scripted=False,
                           is_testing=is_testing,
                           random_graph=random_graph,
                           dynamic_graph=dynamic_graph)

        # Needs to be a string for assertions check in tianshou
        self.agents = [agent.name for agent in self.world.agents]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(
            zip(self.agents, list(range(len(self.possible_agents))))
        )
        self._agent_selector = CustomSelector(self.agents)

        # set spaces
        self.action_spaces = dict()
        self.observation_spaces = dict()
        state_dim = 0
        obs_dim = NUMBER_OF_FEATURES
        for agent in self.world.agents:
            self.observation_spaces[agent.name] = gymnasium.spaces.Dict({
                'observation': gymnasium.spaces.Box(
                    low=0,
                    high=100,
                    shape=(obs_dim,),
                    dtype=np.float32,
                ),
                'action_mask': gymnasium.spaces.Box(low=0,
                                                    high=1,
                                                    shape=(2,),
                                                    dtype=np.int8),
            })
            state_dim += obs_dim
            self.action_spaces[agent.name] = gymnasium.spaces.Discrete(2)

        self.state_space = gymnasium.spaces.Box(
            low=0,
            high=10,
            shape=(state_dim,),
            dtype=np.float32,
        )

        self.max_cycles = max_cycles
        self.num_moves = 0
        self.current_actions = [None] * NUMBER_OF_AGENTS

        self.reset()

    def enable_render(self, mode="human"):
        if not self.renderOn and mode == "human":
            self.renderOn = True

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if self.render_mode == "human" and self.world.agents:
            if self.world.dynamic_graph:
                draw_graph(self.world.pre_move_graph, self.world.pre_move_agents)
            draw_graph(self.world.graph, self.world.agents)

        return

    def close(self):
        if self.renderOn:
            self.renderOn = False

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent) -> gymnasium.spaces.Space:
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent) -> gymnasium.spaces.Space:
        return self.action_spaces[agent]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def observe(self, agent: str):
        if all(value for key, value in self.terminations.items() if
               key in self.agents) and len(self.agents) == 1:
            self.infos[self.agents[0]] = {'reset_all': True,
                                          'messages_transmitted': self.world.messages_transmitted,
                                          'coverage': sum([1 for agent in self.world.agents if
                                                           sum(agent.state.received_from) or agent.state.message_origin]) / self.world.num_agents,
                                          'environment_step': True # Unload the collector sub-buffer
                                          }
            self.is_new_round = False

        # Todo: Need to fix resetting of the environment doesn't issue environment_step
        if self.is_new_round:
            self.infos[self.agent_selection]['environment_step'] = True
            self.is_new_round = False

        return self.observation(
            self.world.agents[self.agent_name_mapping[agent]]
        )

    def state(self):
        states = tuple(
            self.observation(
                self.world.agents[self.agent_name_mapping[agent]]
            )
            for agent in self.possible_agents
        )

        return np.concatenate(states, axis=None)

    def observation(self, agent):
        agent_observation = agent.geometric_data
        one_hop_neighbor_indices = np.where(agent.one_hop_neighbours_ids)[0]
        active_one_hop = [neighbor for neighbor in one_hop_neighbor_indices if not self.world.agents[neighbor].truncated and str(neighbor) in self.agents]

        # Every entry needs to be wrapped in a Batch object, otherwise
        # we will have shape errors in the data replay buffer
        edge_index = np.asarray(agent_observation.edge_index, dtype=np.int32)
        features = np.asarray(agent_observation.features_actor, dtype=np.float32)

        # Store Network for Total/Neighbourhood-wise VDN
        # network = networkx.ego_graph(self.world.graph, agent.id, undirected=True, radius=2)
        network = from_networkx(self.world.graph)
        network_edge_index = np.asarray(network.edge_index, np.int32)
        network_features = np.asarray(network.features_actor, dtype=np.float32)

        labels = np.asarray(network.label, dtype=object)
        data = Batch.stack([Batch(observation=edge_index),
                            Batch(observation=labels),
                            Batch(observation=features),
                            Batch(observation=np.where(labels == agent.id)),
                            Batch(observation=network_edge_index),
                            Batch(observation=network_features),
                            Batch(observation=active_one_hop)])

        agent.allowed_actions[1] = True if (sum(agent.state.received_from) or agent.state.message_origin) and not sum(
            agent.state.transmitted_to) else False
        # Message origin is handled before the first step, hence
        # there is no need to prohibit non transmission
        agent.allowed_actions[0] = False if sum([1 for index in one_hop_neighbor_indices if sum(
            self.world.agents[index].one_hop_neighbours_ids) == 1]) and not sum(agent.state.transmitted_to) else True

        return data

    def reset(self, seed=None, return_info=False, options=None):
        # TODO check that workers seed is set from train_nvdn.py
        if seed is not None:
            self.seed(seed=seed)
        self.world.np_random = self.np_random

        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.rewards = {name: 0.0 for name in self.agents}
        self._cumulative_rewards = {name: 0.0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}
        self.num_moves = 0

        self.world.reset()
        self.agents = [agent.name for agent in self.world.agents
                       if (sum(agent.state.received_from) and
                           not agent.state.message_origin)]
        self._agent_selector.enable(self.agents)

        self.agent_selection = self._agent_selector.next()
        self.current_actions = [None] * NUMBER_OF_AGENTS

    # Tianshou PettingZoo Wrapper returns the reward of every agent in a single
    # time not using CumulativeReward
    def step(self, action):
        if (
                self.terminations[self.agent_selection]
                or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(None)
            self._agent_selector.disable(self.agent_selection)
            return
        current_agent = self.agent_selection

        # current_idx is the agent's index
        current_idx = self.agent_name_mapping[self.agent_selection]
        self.current_actions[current_idx] = action
        agent_tmp = self.world.agents[int(current_agent)]
        agent_tmp.steps_taken += 1

        # the agent which stepped last had its _cumulative_rewards accounted
        # for (because it was returned by last()), so the _cumulative_rewards
        # for this agent should start again at 0
        self._cumulative_rewards[current_agent] = 0

        self.agent_selection = self._agent_selector.next()
        if not self.agent_selection:
            self.infos[current_agent] = {}
            self._accumulate_rewards()
            self._clear_rewards()
            self._execute_world_step()
            self.num_moves += 1

            for agent in self.agents:
                agent_obj = self.world.agents[int(agent)]
                if agent_obj.steps_taken >= 4 and not agent_obj.truncated:
                    agent_obj.truncated = True
                    # terminations must be shifted somehow due to how
                    # the collector gathers the next obs
                    self.terminations[agent_obj.name] = True

            self.agents = [agent.name for agent in self.world.agents
                           if (sum(agent.state.received_from) and
                               not agent.state.message_origin and
                               agent.name in self.terminations)]
            self._agent_selector.enable(self.agents)
            self._agent_selector.start_new_round()
            self.is_new_round = True
            self.agent_selection = self._agent_selector.next()

            # previous_agent = self.agent_selection
            self.current_actions = [None] * NUMBER_OF_AGENTS

            n_received = sum(
                [1 for agent in self.world.agents if
                 sum(agent.state.received_from) or agent.state.message_origin]
            )

            if n_received == NUMBER_OF_AGENTS and self.render_mode == 'human':
                cds = [agent.id for agent in self.world.agents if agent.messages_transmitted > 0]
                print(
                    f"Every agent has received the message, terminating in {self.num_moves}, "
                    f"messages transmitted: {self.world.messages_transmitted}")
                print(cds)
            if self.render_mode == 'human':
                self.render()

        self._deads_step_first()

    def _execute_world_step(self):
        for i, agent in enumerate(self.world.agents):
            action = self.current_actions[i]
            scenario_action = [action]
            self._set_action(scenario_action,
                             agent,
                             self.action_spaces[agent.name])

        self.world.step()

        global_reward = 0.0

        if self.local_ratio is not None:
            global_reward = float(self.global_reward())

        for agent in [a for a in self.world.agents if a.name in self.agents]:
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
        agent.action = action[0]
        action = action[1:]
        assert len(action) == 0

    def global_reward(self):
        # Negative reward
        accumulated = 0
        alpha = 0.001
        for agent in self.world.agents:
            if sum(agent.state.received_from) or agent.state.message_origin:
                accumulated += 1
        completion = accumulated / len(self.world.agents)
        logging.debug(f"Agent {agent.name} received : {- 1 + completion}")
        reward = 1 if completion == 1 else math.log(completion) - math.log(
            self.world.messages_transmitted)
        # reward = - ((alpha * self.world.messages_transmitted) / accumulated)
        return reward

    def reward(self, agent):
        one_hop_neighbor_indices = np.where(agent.one_hop_neighbours_ids)[0]
        two_hop_neighbor_indices = np.where(agent.two_hop_neighbours_ids)[0]
        assert (set(one_hop_neighbor_indices) <= set(two_hop_neighbor_indices))

        reward = agent.two_hop_cover / len(two_hop_neighbor_indices)
        if sum(agent.state.transmitted_to):
            penalty_1 = sum([self.world.agents[index].messages_transmitted for index in one_hop_neighbor_indices]) / len(one_hop_neighbor_indices)
            reward = reward - penalty_1
        if not sum(agent.state.transmitted_to):
            uncovered_n_lens = [len(np.where(self.world.agents[index].one_hop_neighbours_ids)[0]) for index in one_hop_neighbor_indices if
                                sum(self.world.agents[index].state.received_from) == 0
                                and self.world.agents[index].state.message_origin == 0]
            penalty_2 = max(uncovered_n_lens) if len(uncovered_n_lens) else 0
            penalty_2 = penalty_2
            reward = reward - penalty_2

        return reward


def draw_graph(graph, agent_list):
    plt.clf()
    pos = nx.get_node_attributes(graph, "pos")
    color_map = []
    for node in graph:
        if sum(agent_list[node].state.received_from) and not sum(agent_list[node].state.transmitted_to):
            color_map.append('green')
        elif agent_list[node].state.message_origin:
            color_map.append('blue')
        elif agent_list[node].messages_transmitted > 1:
            color_map.append('purple')
        elif sum(agent_list[node].state.transmitted_to):
            color_map.append('red')
        else:
            color_map.append("yellow")

    nx.draw(graph, node_color=color_map, pos=pos, with_labels=True)
    plt.pause(RENDER_PAUSE)


def make_env(raw_env):
    def env(**kwargs):
        env = raw_env(**kwargs)
        env = wrappers.AssertOutOfBoundsWrapper(env)
        env = wrappers.OrderEnforcingWrapper(env)
        # env = frame_stack_v2(env)
        return env

    return env


env = make_env(GraphEnv)
