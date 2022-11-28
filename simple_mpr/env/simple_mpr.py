import math
import random

import numpy as np
import logging
import itertools
from gymnasium.utils import EzPickle

from pettingzoo.utils.conversions import parallel_wrapper_fn

from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv
import pettingzoo.utils.wrappers as wrappers

from simple_mpr.env.utils.wrappers.multi_discrete_to_discrete import MultiDiscreteToDiscreteWrapper

import pygame
from .utils.constants import RADIUS_OF_INFLUENCE, NUMBER_OF_AGENTS
from .utils.core import MprAgent, MprWorld
from gymnasium import spaces

from pettingzoo.utils import agent_selector

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


class RawEnv(SimpleEnv, EzPickle):
    def __init__(self, max_cycles=50, continuous_actions=False, render_mode=None):
        EzPickle.__init__(self, max_cycles, continuous_actions, render_mode)
        scenario = Scenario()
        world = scenario.make_world()

        super().__init__(
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions
        )
        self.metadata["name"] = "simple_mpr"
        self.width = 1000
        self.height = 1000
        self.screen = pygame.Surface([self.width, self.height])
        self.max_size = 1

        # self._agent_selector = MprSelector(self.agents, self.world.agents)
        self._agent_selector = agent_selector(self.agents)

        actions_dim = np.zeros(NUMBER_OF_AGENTS)
        actions_dim.fill(2)

        state_dim = 0
        for agent in world.agents:
            self.action_spaces[agent.name] = spaces.MultiDiscrete(actions_dim)
            obs_dim = len(self.scenario.observation(agent, self.world)['observation'])
            state_dim += obs_dim
            self.observation_spaces[agent.name] = {
                'observation': spaces.Box(low=1, high=1, shape=(obs_dim,), dtype=np.int8),
                'action_mask': spaces.Box(low=0, high=1, shape=(NUMBER_OF_AGENTS,), dtype=np.int8)
            }

        self.state_space = spaces.Box(
            low=0,
            high=1,
            shape=(state_dim,),
            dtype=np.int8
        )

    def draw(self):
        # clear screen
        self.screen.fill((255, 255, 255))

        # update bounds to center around agent
        all_poses = [entity.state.p_pos for entity in self.world.entities]
        cam_range = np.max(np.abs(np.array(all_poses)))

        # update geometry and text positions
        text_line = 0
        for e, entity in enumerate(self.world.entities):
            # geometry
            x, y = entity.state.p_pos
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

            # Draw AoI
            aoi_color = (0, 255, 0, 128) if sum(entity.state.transmitted_to) else (255, 0, 0, 128)
            surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            pygame.draw.circle(surface, aoi_color, (x, y), aoi_radius)
            self.screen.blit(surface, (0, 0))

            entity_color = np.array([78, 237, 105]) if sum(entity.state.received_from) or entity.state.message_origin else entity.color
            pygame.draw.circle(
                self.screen, entity_color, (x, y), entity.size * 350
            )  # 350 is an arbitrary scale factor to get pygame to render similar sizes as pyglet
            pygame.draw.circle(
                self.screen, (0, 0, 0), (x, y), entity.size * 350, 1
            )  # borders
            assert (
                0 < x < self.width and 0 < y < self.height
            ), f"Coordinates {(x, y)} are out of bounds."

            # Draw agent name
            message = str(entity.id)
            self.game_font.render_to(
                self.screen, (x, y), message, (0, 0, 0)
            )

            if isinstance(entity, MprAgent):
                if entity.silent:
                    continue
                if np.all(entity.state.relayed_by == 0):
                    word = "_"
                elif self.continuous_actions:
                    word = (
                        "[" + ",".join([f"{comm:.2f}" for comm in entity.state.c]) + "]"
                    )
                else:
                    indices = [i for i, x in enumerate(entity.state.relayed_by) if x == 1]
                    word = str(indices)

                message = entity.name + " chosen MPR " + word
                message_x_pos = self.width * 0.05
                message_y_pos = self.height * 0.95 - (self.height * 0.05 * text_line)
                self.game_font.render_to(
                   self.screen, (message_x_pos, message_y_pos), message, (0, 0, 0)
                )
                text_line += 1

    def observe(self, agent):
        return self.scenario.observation(
            self.world.agents[self._index_map[agent]], self.world
        )

    def step(self, action):
        if (
                self.terminations[self.agent_selection]
                or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        cur_agent = self.agent_selection
        current_idx = self._index_map[self.agent_selection]
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

        self._cumulative_rewards[cur_agent] = 0
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

        n_received = sum([1 for agent in self.world.agents if sum(agent.state.received_from) or agent.state.message_origin])
        if n_received == NUMBER_OF_AGENTS:
            logging.debug(f"Every agent has received the message, terminating in {self.steps}, messages transmitted: {self.world.messages_transmitted}")
            for agent in self.agents:
                self.terminations[agent] = True

    # Here actions are called from execute world step
    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)

        if not agent.silent:
            agent.action.c = action[0]
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0


class Scenario(BaseScenario):
    def make_world(self):
        world = MprWorld()
        # Set world properties
        # Communication dimension which will become also the agent's action
        world.dim_c = NUMBER_OF_AGENTS
        world.collaborative = True
        # add agents TODO: number i provisional
        world.agents = [MprAgent() for i in range(NUMBER_OF_AGENTS)]
        # Set agents properties, they are all the same for now
        for i, agent in enumerate(world.agents):
            agent.name = f"node_{i}"
            agent.collide = False
            agent.size = .075
            # Agents cannot move for now
            agent.movable = False
        # No Landmarks needed
        return world

    def reset_world(self, world, np_random):
        # Agents do not have explicit goals to reach
        for agent in world.agents:
            agent.goal_a = None
            agent.goal_b = None
        # Agent colors
        for i, agent in enumerate(world.agents):
            agent.color = np.array([237, 200, 78])
            # agent.color = np.array([0.25, 0.25, 0.25])

        random.shuffle(world.agents)
        previous_position = np_random.uniform(-1, +1, world.dim_p)
        for agent in world.agents:
            r = RADIUS_OF_INFLUENCE * 0.99
            theta = np_random.uniform() * 2 * math.pi
            agent.state.p_pos = [previous_position[0] + r * math.cos(theta), previous_position[1] + r * math.sin(theta)]
            previous_position = agent.state.p_pos
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.state.received_from = np.zeros(NUMBER_OF_AGENTS)
            agent.state.transmitted_to = np.zeros(NUMBER_OF_AGENTS)
            agent.state.relays_for = np.zeros(NUMBER_OF_AGENTS)
            agent.state.relayed_by = np.zeros(NUMBER_OF_AGENTS)
            agent.state.message_origin = 0
            agent.one_hop_neighbours_ids = None
            agent.two_hop_neighbours_ids = None
            agent.one_hop_neighbours_neighbours_ids = None
            agent.allowed_actions = None

            agent.id = int(agent.name.split('_')[1])

        assert (check_connected(world.agents))

        # As it's static for every agent calculate its one and two hop neighbours
        for agent in world.agents:
            one_hop_neighbours, one_hop_neighbours_ids = calculate_neighbours(agent, world.agents)
            assert len(one_hop_neighbours_ids) == NUMBER_OF_AGENTS

            two_hop_neighbours = []
            two_hop_neighbours_ids = []
            for possible_neighbour in world.agents:
                if possible_neighbour in one_hop_neighbours:
                    neighbour_neighbours, neighbour_neighbours_names = calculate_neighbours(possible_neighbour, world.agents)
                    two_hop_neighbours.append(neighbour_neighbours)
                    two_hop_neighbours_ids.append(neighbour_neighbours_names)
                elif possible_neighbour is not agent:
                    two_hop_neighbours_ids.append([0] * NUMBER_OF_AGENTS)

            assert np.array(two_hop_neighbours_ids).shape == (NUMBER_OF_AGENTS, NUMBER_OF_AGENTS)

            # Not used for now, the information is given in full about neighbourhoods
            final_two_hop = np.zeros((NUMBER_OF_AGENTS,))
            for sublist in two_hop_neighbours_ids:
                temp = []
                for x, y in zip(final_two_hop, sublist):
                    temp.append(x or y)
                final_two_hop = temp

            agent.one_hop_neighbours_ids = one_hop_neighbours_ids
            agent.two_hop_neighbours_ids = two_hop_neighbours_ids

            # Calc every combination of the agent's neighbours to get allowed actions
            neighbours_combinations = list(itertools.product([0, 1], repeat=sum(one_hop_neighbours_ids)))
            indices = [i for i, x in enumerate(one_hop_neighbours_ids) if x == 1]

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
        # Reset messages counter
        world.messages_transmitted = 0
        # Randomly select an agent with the message
        random_agent = np_random.choice(world.agents)
        random_agent.state.message_origin = 1

    def benchmark_data(self, agent, world):
        return self.reward(agent, world)

    def reward(self, agent, world):
        accumulated = 0
        for other_agent in world.agents:
            if other_agent is agent:
                continue
            if sum(other_agent.state.received_from):
                accumulated += 1
        completion = accumulated/len(world.agents) if accumulated > 0 else 0
        logging.debug(f"Agent {agent.name} received : { - 1 + completion}")
        return (- 1 + completion) * world.messages_transmitted

    def observation(self, agent, world):
        # communication of all other agents
        comm = []
        for other in world.agents:
            if other is agent or (other.state.c is None):
                continue
            comm.append(other.state.c)

        # communication status, received and message transmitted flag
        received = agent.state.received_from
        transmitted_to = agent.state.transmitted_to

        agent_observation = np.concatenate([agent.one_hop_neighbours_ids] + agent.two_hop_neighbours_ids + [transmitted_to] + [received])
        assert(sum(agent.allowed_actions) == 2 ** sum(agent.one_hop_neighbours_ids))
        return {"observation": agent_observation, "action_mask": agent.allowed_actions}


def calculate_neighbours(agent, possible_neighbours):
    neighbours = []
    neighbours_names = []
    for possible_neighbour in possible_neighbours:
        l2_distance = math.dist(possible_neighbour.state.p_pos, agent.state.p_pos)
        if l2_distance <= RADIUS_OF_INFLUENCE:
            neighbours_names.append(1)
            neighbours.append(possible_neighbour)
        else:
            neighbours_names.append(0)
            neighbours.append(None)

    return np.array(neighbours), np.array(neighbours_names)


def check_connected(agents):
    visited = set()  # Set to keep track of visited nodes of graph.

    def dfs(visited, graph, node):
        if node not in visited:
            visited.add(node)
            # TODO: use agent attribute instead of calculation
            neighbours, _ = calculate_neighbours(node, graph)
            for neighbour in neighbours:
                if neighbour is not None:
                    dfs(visited, graph, neighbour)

    dfs(visited, agents, agents[0])

    return True if len(visited) == len(agents) else False


def make_env(raw_env):
    def env(**kwargs):
        env = raw_env(**kwargs)
        if env.continuous_actions:
            env = wrappers.ClipOutOfBoundsWrapper(env)
        else:
            env = MultiDiscreteToDiscreteWrapper(env)
            env = wrappers.AssertOutOfBoundsWrapper(env)
        env = wrappers.OrderEnforcingWrapper(env)
        return env

    return env


env = make_env(RawEnv)
# parallel_env = parallel_wrapper_fn(env)
