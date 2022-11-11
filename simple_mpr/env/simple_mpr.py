import numpy as np
import logging

from gymnasium.utils import EzPickle

from pettingzoo.utils.conversions import parallel_wrapper_fn

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World, AgentState, Action
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env

import pygame
from .utils.constants import AREA_OF_INFLUENCE, NUMBER_OF_AGENTS
from gymnasium import spaces

from pettingzoo.utils import agent_selector

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


class MprSelector():
    """Outputs an agent in the given order whenever agent_select is called.

    Can reinitialize to a new order
    """

    def __init__(self, agent_order, agents):
        self.reinit(agent_order, agents)

    def reinit(self, agent_order, agents):
        self.agent_order = agent_order
        self.agents = agents
        self._current_agent = 0
        self.selected_agent = 0

    def reset(self):
        self.reinit(self.agent_order, self.agents)
        return self.next()

    def next(self):
        # while True:
        agents_received = [agent for agent in self.agents if agent.state.received and not agent.state.c]
        self._current_agent = (self._current_agent + 1) % len(self.agents)

        temp_index = self._current_agent % len(agents_received)
        self.selected_agent = self.agent_order[int(agents_received[temp_index].name.split('_')[1])]

        return self.selected_agent

    def is_last(self):
        """Does not work as expected if you change the order."""
        return self.selected_agent == self.agent_order[-1]

    def is_first(self):
        return self.selected_agent == self.agent_order[0]

    def __eq__(self, other):
        if not isinstance(other, agent_selector):
            return NotImplemented

        return (
            self.agent_order == other.agent_order
            and self._current_agent == other._current_agent
            and self.selected_agent == other.selected_agent
        )


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

        self._agent_selector = MprSelector(self.agents, self.world.agents)

        for agent in world.agents:
            self.action_spaces[agent.name] = spaces.Discrete(2)

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
            y *= (
                -1
            )  # this makes the display mimic the old pyglet setup (ie. flips image)
            x = (
                (x / cam_range) * self.width // 2 * 0.9
            )  # the .9 is just to keep entities from appearing "too" out-of-bounds
            y = (y / cam_range) * self.height // 2 * 0.9
            x += self.width // 2
            y += self.height // 2

            # Draw AoI
            aoi_color = (0, 255, 0, 128) if entity.state.c else (255, 0, 0, 128)
            surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            pygame.draw.circle(surface, aoi_color, (x, y), (entity.size + AREA_OF_INFLUENCE) * 350)
            self.screen.blit(surface, (0, 0))

            entity_color = np.array([78, 237, 105]) if entity.state.received else entity.color
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
            message = entity.name.split('_')[1]
            self.game_font.render_to(
                self.screen, (x, y), message, (0, 0, 0)
            )

            if isinstance(entity, MprAgent):
                if entity.silent:
                    continue
                if np.all(entity.state.c == 0):
                    word = "_"
                elif self.continuous_actions:
                    word = (
                        "[" + ",".join([f"{comm:.2f}" for comm in entity.state.c]) + "]"
                    )
                else:
                    if entity.state.c == 1:
                        indices = [i for i, x in enumerate(entity.one_hop_neighbours_ids) if x == 1]
                        word = str(indices)
                    else:
                        word = []

                message = entity.name + " sent to " + word
                message_x_pos = self.width * 0.05
                message_y_pos = self.height * 0.95 - (self.height * 0.05 * text_line)
                self.game_font.render_to(
                   self.screen, (message_x_pos, message_y_pos), message, (0, 0, 0)
                )
                text_line += 1

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        cur_agent = self.agent_selection
        current_idx = self._index_map[self.agent_selection]
        # next_idx = (current_idx + 1) % len([agent for agent in self.world.agents if agent.state.received and not agent.state.c])
        last = max([int(agent.name.split('_')[1]) for agent in self.world.agents if agent.state.received and not agent.state.c])
        self.agent_selection = self._agent_selector.next()

        self.current_actions[current_idx] = action
        if int(self.agent_selection.split('_')[1]) == last:
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

        n_received = sum([agent.state.received for agent in self.world.agents])
        if n_received == NUMBER_OF_AGENTS:
            logging.debug(f"Every agent has received the message, terminating in {self.steps}")
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


env = make_env(RawEnv)
parallel_env = parallel_wrapper_fn(env)


class MprAgentState(AgentState):
    def __int__(self):
        super().__init__()
        self.received = None


# Currently not used but might be needed for additional properties
class MprAgent(Agent):
    def __init__(self):
        super().__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = MprAgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        self.one_hop_neighbours_ids = None
        self.two_hop_neighbours_ids = None


class MprWorld(World):
    # update state of the world
    def step(self):
        # set actions for scripted agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # update agent state

        for agent in self.agents:
            logging.debug(f"Agent {agent.name} Action: {agent.action.c} with Neigh: {agent.one_hop_neighbours_ids}")
            self.update_agent_state(agent)
            for a in self.agents:
                logging.debug(f"Agents {a.name} R: {a.state.received}")

    def update_agent_state(self, agent):
        if agent.action.c and agent.state.received:
            logging.debug(f"Agent {agent.name} transmitted")
            agent.state.c = 1
            agent.state.fired = 1
            indices = [i for i, x in enumerate(agent.one_hop_neighbours_ids) if x == 1]
            for index in indices:
                self.agents[index].state.received = 1
        else:
            logging.debug("TRIED")


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


class Scenario(BaseScenario):
    def make_world(self):
        world = MprWorld()
        # Set world properties
        # Communication dimension which will become also the agent's action
        world.dim_c = 1
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

        # Set random initial state
        while True:
            for agent in world.agents:
                agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)
                agent.state.received = 0

            if check_connected(world.agents):
                logging.debug("Created a connected graph!")
                break

        # Randomly select an agent with the message
        random_agent = np_random.choice(world.agents)
        random_agent.state.received = 1

    def benchmark_data(self, agent, world):
        return self.reward(agent, world)

    def reward(self, agent, world):
        accumulated = 0
        for other_agent in world.agents:
            if other_agent is agent:
                continue
            if other_agent.state.received == 1:
                accumulated += 1
        completion = accumulated/len(world.agents) if accumulated > 0 else 0
        logging.debug(f"Agent {agent.name} received : { - 1 + completion}")
        return - 1 + completion

    def observation(self, agent, world):
        # Move neighbours as an agent property so we can query it easily
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

        final_two_hop = np.zeros((NUMBER_OF_AGENTS,))
        for sublist in two_hop_neighbours_ids:
            temp = []
            for x, y in zip(final_two_hop, sublist):
                temp.append(x or y)
            final_two_hop = temp

        # communication of all other agents
        comm = []
        for other in world.agents:
            if other is agent or (other.state.c is None):
                continue
            comm.append(other.state.c)

        # communication status, received and message transmitted flag
        r = agent.state.received if agent.state.received is not None else 0
        t = agent.state.c if agent.state.c is not None else 0

        agent.one_hop_neighbours_ids = one_hop_neighbours_ids
        agent.two_hop_neighbours_ids = final_two_hop

        transmitted = np.zeros(NUMBER_OF_AGENTS,)
        received = np.zeros(NUMBER_OF_AGENTS,)

        # TODO: see if there is a better way to do this also we might want an history
        transmitted.put(0, t)
        received.put(0, r)

        return np.concatenate([agent.one_hop_neighbours_ids] + [agent.two_hop_neighbours_ids] + [transmitted] + [received])


def calculate_neighbours(agent, possible_neighbours):
    neighbours = []
    neighbours_names = []
    for possible_neighbour in possible_neighbours:
        l2_distance = np.sum(np.square(possible_neighbour.state.p_pos - agent.state.p_pos))
        if l2_distance <= AREA_OF_INFLUENCE:
            neighbours_names.append(1)
            neighbours.append(possible_neighbour)
        else:
            neighbours_names.append(0)
            neighbours.append(None)

    return np.array(neighbours), np.array(neighbours_names)
