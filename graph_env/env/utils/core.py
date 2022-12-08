import itertools
import numpy as np
import logging

import networkx as nx
from torch_geometric.utils import from_networkx

from .constants import NUMBER_OF_AGENTS, RADIUS_OF_INFLUENCE


class MprAgentState:
    def __init__(self):
        self.received_from = None
        self.transmitted_to = None
        self.relays_for = None
        self.relayed_by = None
        self.message_origin = 0


class MprAgent:
    def __init__(self,
                 id,
                 local_view,
                 size=0.05,
                 color=(0, 0, 0),
                 state=None,
                 pos=None):
        # state
        self.id = id
        self.name = str(id)
        self.state = MprAgentState() if state is None else state
        self.local_view = local_view
        self.geometric_data = from_networkx(local_view)
        self.size = size
        self.pos = pos
        self.color = color
        self.one_hop_neighbours_ids = None
        self.one_hop_neighbours_neighbours_ids = None
        self.allowed_actions = None
        self.action = None

    def reset(self, local_view, pos):
        self.__init__(id=self.id, local_view=local_view, state=self.state, pos=pos)

    def has_received_from_relayed_node(self):
        return sum([received and relay for received, relay in
                    zip(self.state.received_from, self.state.relays_for)])


class MprWorld:
    # update state of the world
    def __init__(
            self,
            number_of_agents,
            radius,
            np_random,
            graph=None
    ):
        self.messages_transmitted = 0
        self.num_agents = number_of_agents
        self.radius = radius
        self.graph = nx.random_geometric_graph(n=number_of_agents,
                                               radius=radius) if graph is None else graph
        self.agents = [MprAgent(i, nx.ego_graph(self.graph, i, undirected=True))
                       for i in range(number_of_agents)]
        self.num_agents = number_of_agents
        self.np_random = np_random
        self.reset()

    def step(self):
        # set actions for scripted agents
        # for agent in self.scripted_agents:
        #     agent.action = agent.action_callback(agent, self)

        # Set MPRs
        for agent in self.agents:
            self.set_relays(agent)

        # Send message
        for agent in self.agents:
            logging.debug(f"Agent {agent.name} Action: {agent.action} with Neigh: {agent.one_hop_neighbours_ids}")
            self.update_agent_state(agent)

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
                self.agents[index].state.relays_for[agent.id] = value

    def update_agent_state(self, agent):
        # if it has received from a relay node or is message origin and has not already transmitted the message
        if (agent.has_received_from_relayed_node() or agent.state.message_origin)\
                and not sum(agent.state.transmitted_to):
            logging.debug(
                f"Agent {agent.name} sending to Neighs: {agent.one_hop_neighbours_ids}")

            agent.state.transmitted_to = agent.one_hop_neighbours_ids
            self.messages_transmitted += 1
            neighbour_indices = [i for i, x in
                                 enumerate(agent.one_hop_neighbours_ids) if
                                 x == 1]
            for index in neighbour_indices:
                self.agents[index].state.received_from[agent.id] = 1
        else:
            logging.debug(f"Agent {agent.name} could not send")

    def reset(self):
        while True:
            self.graph = nx.random_geometric_graph(n=self.num_agents, radius=self.radius)
            if nx.is_connected(self.graph):
                break
            else:
                logging.debug("Generated graph not connected, retry")

        for agent in self.agents:
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

        actions_dim = np.zeros(NUMBER_OF_AGENTS)
        actions_dim.fill(2)
        for agent in self.agents:
            agent.reset(local_view=nx.ego_graph(self.graph, agent.id, undirected=True), pos = self.graph.nodes[agent.id]['pos'])
            agent.one_hop_neighbours_ids = self.graph.nodes[agent.id]['features']

            # Calc every combination of the agent's neighbours to get allowed actions
            neighbours_combinations = list(itertools.product([0, 1], repeat=int(sum(agent.one_hop_neighbours_ids))))
            indices = [i for i, x in enumerate(agent.one_hop_neighbours_ids) if x == 1]

            allowed_actions_mask = [False] * int(np.prod(actions_dim))
            for element in neighbours_combinations:
                allowed_action_binary = [0] * NUMBER_OF_AGENTS
                for i in range(len(element)):
                    allowed_action_binary[indices[i]] = element[i]

                res = int("".join(str(x) for x in allowed_action_binary), 2)
                allowed_actions_mask[res] = True

            agent.allowed_actions = allowed_actions_mask

        random_agent = self.np_random.choice(self.agents)
        random_agent.state.message_origin = 1


