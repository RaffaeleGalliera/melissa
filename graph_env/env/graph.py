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

from .utils.constants import RADIUS_OF_INFLUENCE, NUMBER_OF_AGENTS
from .utils.core import Agent, MprWorld, World

import matplotlib.pyplot as plt


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
            max_cycles=5,
            device='cuda',
            local_ratio=None,
            seed=9,
            py_game=False
    ):
        super().__init__()
        self.py_game = py_game

        if self.py_game:
            pygame.init()
            self.game_font = pygame.freetype.Font(None, 24)
            self.viewer = None
            self.width = 1024
            self.height = 1024
            self.screen = pygame.Surface([self.width, self.height])
            self.max_size = 1
            plt.ion()
            plt.show()

        self.np_random = None
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
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self._agent_selector = agent_selector(self.agents)
        self.max_cycles = max_cycles
        self.steps = 0
        self.current_actions = [None] * self.num_agents

        self.reset()

        # set spaces
        self.action_spaces = dict()
        self.observation_spaces = dict()

        actions_dim = np.zeros(NUMBER_OF_AGENTS)
        actions_dim.fill(2)

        for agent in self.world.agents:
            obs_dim = len(self.observe(agent.name)['observation'])
            mask_dim = len(self.observe(agent.name)['action_mask'])

            self.action_spaces[agent.name] = gymnasium.spaces.Discrete(2)
            self.observation_spaces[agent.name] = {
                'observation': gymnasium.spaces.Box(low=0, high=np.inf, shape=(obs_dim, )),
                'action_mask': gymnasium.spaces.Box(low=0, high=1,
                                                    shape=(mask_dim,),
                                                    dtype=np.int8)
            }


    def enable_render(self, mode="human"):
        if not self.renderOn and mode == "human":
            self.screen = pygame.display.set_mode(self.screen.get_size())
            self.renderOn = True

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if self.py_game:
            self.enable_render(self.render_mode)
            observation = np.array(pygame.surfarray.pixels3d(self.screen))

        if self.render_mode == "human" and self.world.agents:
            self.draw_graph()
            if self.py_game:
                self.draw()
                pygame.display.flip()
        return (
            np.transpose(observation, axes=(1, 0, 2))
            if self.render_mode == "rgb_array"
            else None
        )

    def draw(self, enable_aoi=False):
        # clear screen
        self.screen.fill((255, 255, 255))

        # update bounds to center around agent
        all_poses = [agent.pos for agent in self.world.agents]
        cam_range = np.max(np.abs(np.array(all_poses)))

        # update geometry and text positions
        text_line = 0
        for e, agent in enumerate(self.world.agents):
            # geometry
            x, y = agent.pos
            x_influence = ((x + RADIUS_OF_INFLUENCE) / cam_range) * self.width // 2 * 0.9
            x_influence += self.width // 2

            y *= (
                -1
            )  # this makes the display mimic the old pyglet setup (ie. flips image)
            x = (
                (x / cam_range) * self.width // 2 * 0.9
            )  # the .9 is just to keep entities from appearing "too" out-of-bounds
            y = (y / cam_range) * self.height // 2 * 0.9
            x += self.width // 2
            y += self.height // 2

            aoi_radius = math.dist([x, y], [x_influence, y])

            if enable_aoi:
                # Draw AoI
                aoi_color = (0, 255, 0, 128) if sum(agent.state.transmitted_to) else (255, 0, 0, 128)
                surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
                pygame.draw.circle(surface, aoi_color, (x, y), aoi_radius)
                self.screen.blit(surface, (0, 0))

            entity_color = np.array([78, 237, 105]) if sum(agent.state.received_from) or agent.state.message_origin else agent.color
            pygame.draw.circle(
                self.screen, entity_color, (x, y), agent.size * 350
            )  # 350 is an arbitrary scale factor to get pygame to render similar sizes as pyglet
            pygame.draw.circle(
                self.screen, (0, 0, 0), (x, y), agent.size * 350, 1
            )  # borders
            assert (
                0 < x < self.width and 0 < y < self.height
            ), f"Coordinates {(x, y)} are out of bounds."

            # Draw agent name
            # message = agent.name
            # self.game_font.render_to(
            #     self.screen, (x, y), message, (0, 0, 0)
            # )
            #
            # if isinstance(agent, MprAgent):
            #     if np.all(agent.state.transmitted_to == 0):
            #         word = "_"
            #     else:
            #         indices = [i for i, x in enumerate(agent.state.transmitted_to) if x == 1]
            #         word = str(indices)
            #
            #     message = agent.name + " Transmitted to " + word
            #     message_x_pos = self.width * 0.05
            #     message_y_pos = self.height * 0.95 - (self.height * 0.05 * text_line)
            #     self.game_font.render_to(
            #        self.screen, (message_x_pos, message_y_pos), message, (0, 0, 0), bgcolor=entity_color
            #     )
            #     text_line += 1

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
            pygame.event.pump()
            pygame.display.quit()
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
            if self.render_mode == 'human':
                self.render()
        else:
            self._clear_rewards()

        self._cumulative_rewards[current_agent] = 0
        self._accumulate_rewards()

        n_received = sum([1 for agent in self.world.agents if
                          sum(agent.state.received_from) or agent.state.message_origin])
        if n_received == NUMBER_OF_AGENTS:
            logging.debug(
                f"Every agent has received the message, terminating in {self.steps}, messages transmitted: {self.world.messages_transmitted}")
            for agent in self.agents:
                self.terminations[agent] = True

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
        accumulated = 0
        for agent in self.world.agents:
            if sum(agent.state.received_from) or agent.state.message_origin:
                accumulated += 1
        completion = accumulated / len(self.world.agents)
        logging.debug(f"Agent {agent.name} received : {- 1 + completion}")
        return (- 1 + completion) * self.world.messages_transmitted


def make_env(raw_env):
    def env(**kwargs):
        env = raw_env(**kwargs)
        # env = MultiDiscreteToDiscreteWrapper(env)
        env = wrappers.AssertOutOfBoundsWrapper(env)
        env = wrappers.OrderEnforcingWrapper(env)
        return env

    return env


env = make_env(GraphEnv)
