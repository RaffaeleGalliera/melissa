import itertools
import logging
import math
import os
import pygame

import gymnasium
import networkx as nx
import torch
from gymnasium.utils import seeding
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers, agent_selector
import numpy as np
from tianshou.data.batch import Batch

from .utils.constants import RADIUS_OF_INFLUENCE, NUMBER_OF_AGENTS, NUMBER_OF_FEATURES
from .utils.core import Agent, World

import matplotlib.pyplot as plt
import math


class GraphEnv(AECEnv):
    metadata = {
        'render_modes': ["human"],
        'name': "graph_environment",
        'is_parallelizable': True
    }

    def __init__(
            self,
            graph=None,
            render_mode=None,
            number_of_agents=10,
            radius=10,
            max_cycles=15,
            device='cuda',
            local_ratio=None,
            seed=9,
    ):
        super().__init__()
        self.seed(seed)
        self.device = device

        self.render_mode = render_mode
        self.renderOn = False
        self.local_ratio = local_ratio
        self.radius = radius

        self.world = World(graph=graph,
                           number_of_agents=number_of_agents,
                           radius=radius,
                           np_random=self.np_random,
                           seed=seed,
                           is_scripted=False)

        # Needs to be a string for assertions check in tianshou
        self.agents = [agent.name for agent in self.world.agents]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(
            zip(self.agents, list(range(len(self.possible_agents))))
        )
        self._agent_selector = agent_selector(self.agents)

        # set spaces
        self.action_spaces = dict()
        self.observation_spaces = dict()
        state_dim = 0
        for agent in self.world.agents:
            obs_dim = NUMBER_OF_FEATURES
            self.observation_spaces[agent.name] = gymnasium.spaces.Dict({
                'observation': gymnasium.spaces.Box(
                    low=0,
                    high=100,
                    shape=(obs_dim,),
                    dtype=np.float32,
                ),
                'action_mask': gymnasium.spaces.Box(low=0, high=1, shape=(2,), dtype=np.int8),
            })
            state_dim += obs_dim
            self.action_spaces[agent.name] = gymnasium.spaces.Discrete(2)

        self.state_space = gymnasium.spaces.Box(
            low=0,
            high=100,
            shape=(state_dim, ),
            dtype=np.float32,
        )

        self.max_cycles = max_cycles
        self.steps = 0
        self.current_actions = [None] * self.num_agents

        self.reset()

        self.np_random = None

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
            self.draw_graph()

        return

    def draw_graph(self):
        plt.clf()
        pos = nx.get_node_attributes(self.world.graph, "pos")
        color_map = []
        for node in self.world.graph:
            if sum(self.world.agents[node].state.received_from) and not sum(self.world.agents[node].state.transmitted_to):
                color_map.append('green')
            elif self.world.agents[node].state.message_origin:
                color_map.append('blue')
            elif sum(self.world.agents[node].state.transmitted_to):
                color_map.append('red')
            else:
                color_map.append("yellow")

        nx.draw(self.world.graph, node_color=color_map, pos=pos, with_labels=True)
        plt.pause(.001)

    def close(self):
        if self.renderOn:
            self.renderOn = False

    def observation_space(self, agent) -> gymnasium.spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent) -> gymnasium.spaces.Space:
        return self.action_spaces[agent]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        np.random.seed(seed)

    def observe(self, agent: str):
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

        # Every entry needs to be wrapped in a Batch object, otherwise
        # we will have shape errors in the data replay buffer
        edge_index = np.asarray(agent_observation.edge_index, dtype=np.int32)
        features = np.asarray(agent_observation.features, dtype=np.float32)

        labels = np.asarray(agent_observation.label, dtype=object)
        data = Batch.stack([Batch(observation=edge_index),
                            Batch(observation=labels),
                            Batch(observation=features),
                            Batch(observation=np.where(labels == agent.id))])

        agent.allowed_actions[1] = True if (sum(agent.state.received_from) or agent.state.message_origin) and not sum(agent.state.transmitted_to) else False
        agent.allowed_actions[0] = False if (agent.state.message_origin and not sum(agent.state.transmitted_to)) else True

        agent_observation_with_mask = {
            "observation": data,
            "action_mask": agent.allowed_actions
        }

        return agent_observation_with_mask

    def reset(self, seed=9, return_info=False, options=None):
        if seed is not None:
            self.seed(seed=seed)

        self.agents = self.possible_agents[:]
        self.rewards = {name: 0.0 for name in self.agents}
        self._cumulative_rewards = {name: 0.0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}
        self.agent_selection = self._agent_selector.reset()
        self.steps = 0

        self.world.reset()
        self.current_actions = [None] * self.num_agents

    def step(self, action):
        if(
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        current_agent = self.agent_selection
        # current_idx is the agent's index
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

            n_received = sum([1 for agent in self.world.agents if
                              sum(agent.state.received_from) or agent.state.message_origin])
            if n_received == NUMBER_OF_AGENTS:
                print(
                    f"Every agent has received the message, terminating in {self.steps}, "
                    f"messages transmitted: {self.world.messages_transmitted}")
                for agent in self.agents:
                    self.terminations[agent] = True

            if self.render_mode == 'human':
                self.render()
        else:
            self._clear_rewards()

        self._cumulative_rewards[current_agent] = 0
        self._accumulate_rewards()

    def _execute_world_step(self):
        for i, agent in enumerate(self.world.agents):
            action = self.current_actions[i]
            scenario_action = [action]
            self._set_action(scenario_action, agent, self.action_spaces[agent.name])

        self.world.step()

        global_reward = 0.0
        # TODO: exp with local ratio
        if self.local_ratio is not None:
            pass
            # global_reward = float(self.global_reward())

        for agent in self.world.agents:
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

    def reward(self, agent):
        ##  Negative reward
        accumulated = 0
        alpha = 0.001
        for agent in self.world.agents:
            if sum(agent.state.received_from) or agent.state.message_origin:
                accumulated += 1
        completion = accumulated / len(self.world.agents)
        logging.debug(f"Agent {agent.name} received : {- 1 + completion}")
        reward = 1 if completion == 1 else math.log(completion) - math.log(self.world.messages_transmitted)
        # reward = - ((alpha * self.world.messages_transmitted) / accumulated)
        return reward


def make_env(raw_env):
    def env(**kwargs):
        env = raw_env(**kwargs)
        # env = MultiDiscreteToDiscreteWrapper(env)
        env = wrappers.AssertOutOfBoundsWrapper(env)
        env = wrappers.OrderEnforcingWrapper(env)
        return env

    return env


env = make_env(GraphEnv)
