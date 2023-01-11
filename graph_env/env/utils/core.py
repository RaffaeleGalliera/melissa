import itertools
import pickle

import numpy as np
import logging

import networkx as nx
from torch_geometric.utils import from_networkx


# OLSRv1 MPR computation https://www.rfc-editor.org/rfc/rfc3626.html#section-8.3.1
def mpr_heuristic(one_hop_neighbours_ids,
                  two_hop_neighbours_ids,
                  agent_id,
                  local_view
                  ):
    d_y = dict()
    two_hop_coverage_summary = {index: [] for index, value in
                                enumerate(two_hop_neighbours_ids) if
                                value == 1}
    covered = np.zeros(len(one_hop_neighbours_ids))
    mpr = np.zeros(len(one_hop_neighbours_ids))

    for neighbor in local_view.neighbors(agent_id):
        clean_neighbor_neighbors = local_view.nodes[neighbor]['one_hop'].copy()
        # Exclude from the list the 2hop neighbours already reachable by self
        clean_neighbor_neighbors[agent_id] = 0
        for index in [i for i, x in enumerate(one_hop_neighbours_ids) if
                      x == 1]:
            clean_neighbor_neighbors[index] = 0
        # Calculate the coverage for two hop neighbors
        for index in [i for i, x in enumerate(two_hop_neighbours_ids.astype(
                        int) & clean_neighbor_neighbors.astype(int)) if x == 1]:
            two_hop_coverage_summary[index].append(
                int(local_view.nodes[neighbor]['label']))
        d_y[int(local_view.nodes[neighbor]['label'])] = sum(
            clean_neighbor_neighbors)

    # Add to covered if the cleaned 2hop neighbors are the only ones providing that link so far
    for key, candidates in two_hop_coverage_summary.items():
        if len(candidates) == 1:
            mpr[candidates[0]] = 1
            covered[key] = 1

    reachable_uncovered = np.array(
        [1 if (value_2h and not value_c) else 0 for value_2h, value_c in
         zip(two_hop_neighbours_ids, covered)])
    while (reachable_uncovered != 0).any():
        reachability = dict()
        for neighbor in local_view.neighbors(agent_id):
            reachability[int(local_view.nodes[neighbor]['label'])] = sum(
                local_view.nodes[neighbor]['one_hop'].astype(int) & reachable_uncovered.astype(int))
        max_reachability = [k for k, v in reachability.items() if v == max(reachability.values())]
        if len(max_reachability) == 1:
            key_to_add = max_reachability[0]
            mpr[key_to_add] = 1
            reachable_uncovered = np.array([
                0 if (value_n and value_unc) else value_unc for
                value_n, value_unc in
                zip(local_view.nodes[key_to_add]['one_hop'], reachable_uncovered)])

        elif len(max_reachability) > 1:
            key_to_add = max({k: d_y[k] for k in max_reachability})
            mpr[key_to_add] = 1
            reachable_uncovered = np.array([
                0 if (value_n and value_unc) else value_unc for
                value_n, value_unc in
                zip(local_view.nodes[key_to_add]['one_hop'], reachable_uncovered)])

    return mpr


class State:
    def __init__(self):
        self.received_from = None
        self.transmitted_to = None
        self.relays_for = None
        self.relayed_by = None
        self.message_origin = 0

    def reset(self, num_agents):
        self.received_from = np.zeros(num_agents)
        self.transmitted_to = np.zeros(num_agents)
        self.relays_for = np.zeros(num_agents)
        self.relayed_by = np.zeros(num_agents)
        self.message_origin = 0

class Agent:
    def __init__(self,
                 agent_id,
                 local_view,
                 size=0.05,
                 color=(0, 0, 0),
                 state=None,
                 pos=None,
                 is_scripted=False):
        # state
        self.id = agent_id
        self.name = str(agent_id)
        self.state = State() if state is None else state
        self.local_view = local_view
        self.geometric_data = from_networkx(local_view)
        self.size = size
        self.pos = pos
        self.color = color
        self.one_hop_neighbours_ids = None
        self.two_hop_neighbours_ids = None
        self.allowed_actions = None
        self.action = None
        self.is_scripted = is_scripted
        self.action_callback = mpr_heuristic if self.is_scripted else None
        self.suppl_mpr = None

    def reset(self, local_view, pos):
        self.__init__(agent_id=self.id, local_view=local_view, state=self.state, pos=pos)

    def has_received_from_relayed_node(self):
        return sum([received and relay for received, relay in
                    zip(self.state.received_from, self.state.relays_for)])


class World:
    # update state of the world
    def __init__(
            self,
            number_of_agents,
            radius,
            np_random,
            seed=9,
            graph=None,
            is_scripted=False
    ):
        # Includes origin message
        self.messages_transmitted = 1
        self.origin_agent = None

        self.num_agents = number_of_agents
        self.radius = radius
        self.graph = create_connected_graph(n=self.num_agents, radius=self.radius, seed=seed) if graph is None else graph
        self.is_scripted = is_scripted
        self.random_structure = False if graph else True
        self.agents = [Agent(i, nx.ego_graph(self.graph, i, undirected=True), is_scripted=self.is_scripted)
                       for i in range(number_of_agents)]
        self.np_random = np_random
        self.reset(seed)

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if
                agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if
                agent.action_callback is not None]

    def step(self):
        # set actions for scripted agents
        for agent in self.scripted_agents:
            agent.suppl_mpr = agent.action_callback(agent.one_hop_neighbours_ids,
                                                 agent.two_hop_neighbours_ids,
                                                 agent.id,
                                                 agent.local_view)
            relay_indices = [i for i, x in
                             enumerate(agent.suppl_mpr) if
                             x == 1]

            for index in relay_indices:
                self.agents[index].state.relays_for[agent.id] = 1

        for agent in self.scripted_agents:
            if (agent.has_received_from_relayed_node() or agent.state.message_origin) \
                    and not sum(agent.state.transmitted_to):
                agent.action = 1
            else:
                agent.action = 0

        # Send message
        for agent in self.agents:
            logging.debug(f"Agent {agent.name} Action: {agent.action} with Neigh: {agent.one_hop_neighbours_ids}")
            self.update_agent_state(agent) if agent.action else None

    def update_agent_state(self, agent):
        # if it has received from a relay node or is message origin
        # and has not already transmitted the message
        if (sum(agent.state.received_from) or agent.state.message_origin)\
                and not sum(agent.state.transmitted_to):
            logging.debug(
                f"Agent {agent.name} sending to Neighs: {agent.one_hop_neighbours_ids}")

            agent.state.transmitted_to = agent.one_hop_neighbours_ids
            self.messages_transmitted += 1
            neighbour_indices = [i for i, x in
                                 enumerate(agent.one_hop_neighbours_ids) if
                                 x == 1]
            for index in neighbour_indices:
                self.agents[index].state.received_from[agent.id] = agent.action
        else:
            logging.debug(f"Agent {agent.name} could not send")

    def reset(self, seed):
        if self.random_structure:
            self.graph = create_connected_graph(n=self.num_agents, radius=self.radius, seed=seed)

        # Includes origin message
        self.messages_transmitted = 1
        random_agent = self.np_random.choice(
            self.agents) if self.random_structure else self.agents[0]
        self.origin_agent = random_agent.id

        for agent in self.agents:
            agent.state.reset(self.num_agents)
            one_hop_neighbours_ids = np.zeros(self.num_agents)

            for agent_index in self.graph.neighbors(agent.id):
                one_hop_neighbours_ids[agent_index] = 1
            self.graph.nodes[agent.id]['one_hop'] = one_hop_neighbours_ids
            self.graph.nodes[agent.id]['features'] = np.append(one_hop_neighbours_ids, [sum(one_hop_neighbours_ids), self.origin_agent])
            self.graph.nodes[agent.id]['label'] = agent.id

        actions_dim = np.ones(2)
        for agent in self.agents:
            agent.reset(local_view=nx.ego_graph(self.graph, agent.id, undirected=True),
                        pos=self.graph.nodes[agent.id]['pos'])
            agent.one_hop_neighbours_ids = self.graph.nodes[agent.id]['one_hop']
            agent.two_hop_neighbours_ids = np.zeros(self.num_agents)
            for agent_index in self.graph.neighbors(agent.id):
                agent.two_hop_neighbours_ids = self.graph.nodes[agent_index]['one_hop'].astype(int) | agent.two_hop_neighbours_ids.astype(int)
            agent.two_hop_neighbours_ids[agent.id] = 0

            agent.allowed_actions = [True] * int(np.sum(actions_dim))

        random_agent.state.message_origin = 1


class MprWorld(World):
    def step(self):
        # set actions for scripted agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent.one_hop_neighbours_ids,
                                                 agent.two_hop_neighbours_ids,
                                                 agent.id,
                                                 agent.local_view)

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
        # if it has received from a relay node or is message origin
        # and has not already transmitted the message
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

    def reset(self, seed):
        if self.random_structure:
            self.graph = create_connected_graph(n=self.num_agents, radius=self.radius, seed=seed)

        # Includes origin message
        self.messages_transmitted = 1

        for agent in self.agents:
            agent.state.reset(self.num_agents)

            one_hop_neighbours_ids = np.zeros(self.num_agents)
            for agent_index in self.graph.neighbors(agent.id):
                one_hop_neighbours_ids[agent_index] = 1
            self.graph.nodes[agent.id]['features'] = one_hop_neighbours_ids
            self.graph.nodes[agent.id]['label'] = agent.id

        actions_dim = np.zeros(self.num_agents)
        actions_dim.fill(2)
        for agent in self.agents:
            agent.reset(local_view=nx.ego_graph(self.graph, agent.id, undirected=True),
                        pos=self.graph.nodes[agent.id]['pos'])
            agent.one_hop_neighbours_ids = self.graph.nodes[agent.id]['features']
            agent.two_hop_neighbours_ids = np.zeros(self.num_agents)
            for agent_index in self.graph.neighbors(agent.id):
                agent.two_hop_neighbours_ids = self.graph.nodes[agent_index]['features'].astype(int) | agent.two_hop_neighbours_ids.astype(int)
            agent.two_hop_neighbours_ids[agent.id] = 0

            # Calc every combination of the agent's neighbours to get allowed actions
            neighbours_combinations = list(itertools.product([0, 1], repeat=int(sum(agent.one_hop_neighbours_ids))))
            indices = [i for i, x in enumerate(agent.one_hop_neighbours_ids) if x == 1]

            allowed_actions_mask = [False] * 2**self.num_agents
            for element in neighbours_combinations:
                allowed_action_binary = [0] * self.num_agents
                for i in range(len(element)):
                    allowed_action_binary[indices[i]] = element[i]

                res = int("".join(str(x) for x in allowed_action_binary), 2)
                allowed_actions_mask[res] = True

            agent.allowed_actions = allowed_actions_mask

        random_agent = self.np_random.choice(self.agents) if self.random_structure else self.agents[0]
        random_agent.state.message_origin = 1


def create_connected_graph(n, radius, seed):
    while True:
        graph = nx.random_geometric_graph(n=n, radius=radius, seed=seed)
        if nx.is_connected(graph):
            break
        else:
            logging.debug("Generated graph not connected, retry")

    return graph


def load_testing_graph(path="testing_graph.gpickle"):
    with open(path, "rb") as input_file:
        graph = pickle.load(input_file)
    return graph


def save_testing_graph(graph, path="testing_graph.gpickle"):
    with open(path, "wb") as output_file:
        pickle.dump(graph, output_file)
