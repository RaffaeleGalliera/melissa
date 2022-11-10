import numpy as np
from gymnasium.utils import EzPickle

from pettingzoo.utils.conversions import parallel_wrapper_fn

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World, AgentState, Action
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env

import pygame
from .utils.constants import AREA_OF_INFLUENCE, NUMBER_OF_AGENTS
from gymnasium import spaces


class RawEnv(SimpleEnv, EzPickle):
    def __init__(self, max_cycles=25, continuous_actions=False, render_mode=None):
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
            surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            pygame.draw.circle(surface, (255, 0, 0, 128), (x, y), (entity.size + AREA_OF_INFLUENCE) * 350)
            self.screen.blit(surface, (0, 0))

            pygame.draw.circle(
                self.screen, entity.color, (x, y), entity.size * 350
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

            if isinstance(entity, Agent):
                if entity.silent:
                    continue
                if np.all(entity.state.c == 0):
                    word = "_"
                elif self.continuous_actions:
                    word = (
                        "[" + ",".join([f"{comm:.2f}" for comm in entity.state.c]) + "]"
                    )
                else:
                    indices = [i for i, x in enumerate(entity.state.c) if x == 1]
                    word = str(indices)

                message = entity.name + " sends " + word
                message_x_pos = self.width * 0.05
                message_y_pos = self.height * 0.95 - (self.height * 0.05 * text_line)
                self.game_font.render_to(
                    self.screen, (message_x_pos, message_y_pos), message, (0, 0, 0)
                )
                text_line += 1

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
        self.one_hop_neighbours_names = None
        self.two_hop_neighbours_names = None


class MprWorld(World):
    # update state of the world
    def step(self):
        # set actions for scripted agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # update agent state

        for agent in self.agents:
            print(f"Agnet {agent.name} Action: {agent.action.c} with Neigh: {agent.one_hop_neighbours_names}")
            self.update_agent_state(agent)
            for a in self.agents:
                print(f"Agents {a.name} R: {a.state.received}")


    def update_agent_state(self, agent):
        if agent.action.c:
            agent.state.c = 1
            indices = [i for i, x in enumerate(agent.one_hop_neighbours_names) if x == 1]
            for index in indices:
                self.agents[index].state.received = 1


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
            agent.color = np.random.choice(range(255), size=3)
            # agent.color = np.array([0.25, 0.25, 0.25])

        # Set random initial state
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.state.received = 0

    def benchmark_data(self, agent, world):
        return self.reward(agent, world)

    def reward(self, agent, world):
        return 1

    def observation(self, agent, world):
        # Move neighbours as an agent property so we can query it easily
        one_hop_neighbours, one_hop_neighbours_names = calculate_neighbours(agent, world.agents)
        assert len(one_hop_neighbours_names) == NUMBER_OF_AGENTS

        two_hop_neighbours = []
        two_hop_neighbours_names = []
        for possible_neighbour in world.agents:
            if possible_neighbour in one_hop_neighbours:
                neighbour_neighbours, neighbour_neighbours_names = calculate_neighbours(possible_neighbour, world.agents)
                two_hop_neighbours.append(neighbour_neighbours)
                two_hop_neighbours_names.append(neighbour_neighbours_names)
            elif possible_neighbour is not agent:
                two_hop_neighbours_names.append([0] * NUMBER_OF_AGENTS)

        assert np.array(two_hop_neighbours_names).shape == (NUMBER_OF_AGENTS, NUMBER_OF_AGENTS)

        final_two_hop = np.zeros((NUMBER_OF_AGENTS,))
        for sublist in two_hop_neighbours_names:
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

        agent.one_hop_neighbours_names = one_hop_neighbours_names
        agent.two_hop_neighbours_names = final_two_hop

        # print(f"Agent {agent.name} Comm: {agent.state.c}")
        # print(f"One hop {agent.name} : {agent.one_hop_neighbours_names}")
        # print(f"Two hop {agent.name} : {agent.two_hop_neighbours_names}")

        return np.concatenate([agent.one_hop_neighbours_names] + [agent.two_hop_neighbours_names])


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

from pettingzoo.test import parallel_api_test
if __name__ == "__main__":
    parallel_api_test(RawEnv)
