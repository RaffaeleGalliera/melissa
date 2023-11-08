import copy
from itertools import cycle
import glob
import pickle
import numpy as np
import logging
import networkx as nx
from torch_geometric.utils import from_networkx
from . import constants


# OLSRv1 MPR computation https://www.rfc-editor.org/rfc/rfc3626.html
def mpr_heuristic(one_hop_neighbours_ids,
                  two_hop_neighbours_ids,
                  agent_id,
                  local_view
                  ):
    two_hop_neighbours_ids = two_hop_neighbours_ids - one_hop_neighbours_ids
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
        for index in np.where(one_hop_neighbours_ids)[0]:
            clean_neighbor_neighbors[index] = 0
        # Calculate the coverage for two hop neighbors
        for index in [i for i, x in enumerate(two_hop_neighbours_ids.astype(
                int) & clean_neighbor_neighbors.astype(int)) if x == 1]:
            two_hop_coverage_summary[index].append(
                int(local_view.nodes[neighbor]['label']))
        d_y[int(local_view.nodes[neighbor]['label'])] = sum(
            clean_neighbor_neighbors)

    # Add to covered if the cleaned 2hop neighbors are the only ones providing
    # that link so far
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
        # Used for memory-less MPR heuristic
        self.has_taken_action = 0

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
                 one_hop_neighbors_ids=None,
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
        self.one_hop_neighbours_ids = one_hop_neighbors_ids
        self.two_hop_neighbours_ids = None
        self.two_hop_cover = 0
        self.gained_two_hop_cover = 0
        self.allowed_actions = None
        self.action = None
        self.is_scripted = is_scripted
        self.action_callback = mpr_heuristic if self.is_scripted else None
        self.suppl_mpr = None
        self.steps_taken = None
        self.truncated = None
        self.messages_transmitted = 0
        self.actions_history = np.zeros((4,))

    def reset(self, local_view, pos, one_hop_neighbours_ids):
        self.__init__(agent_id=self.id,
                      local_view=local_view,
                      state=self.state,
                      pos=pos,
                      one_hop_neighbors_ids=one_hop_neighbours_ids,
                      is_scripted=self.is_scripted)

    def update_local_view(self, local_view):
        # local_view is updated
        self.local_view = local_view
        # Convert nx graph into torch tensor and save it in geometric_data param
        self.geometric_data = from_networkx(local_view)

    def has_received_from_relayed_node(self):
        return sum([received and relay for received, relay in
                    zip(self.state.received_from, self.state.relays_for)])

    def update_two_hop_cover_from_one_hopper(self, agents):
        two_hop_neighbor_indices = np.where(self.two_hop_neighbours_ids)[0]

        new_two_hop_cover = sum(
            [1 for index in two_hop_neighbor_indices
             if sum(list(agents[index].state.received_from))
             or agents[index].state.message_origin
             ]
        )

        self.gained_two_hop_cover = new_two_hop_cover - self.two_hop_cover
        self.two_hop_cover = new_two_hop_cover


# In this world the agents select if they are on the MPR set or not
class World:
    # update state of the world
    def __init__(
            self,
            number_of_agents,
            radius,
            np_random,
            graph=None,
            is_scripted=False,
            is_testing=False,
            random_graph=False,
            dynamic_graph=False
    ):
        # Includes origin message
        self.messages_transmitted = 1
        self.origin_agent = None
        self.num_agents = number_of_agents
        self.radius = radius
        self.graph = graph
        self.is_graph_fixed = True if graph else False
        self.is_scripted = is_scripted
        self.random_graph = random_graph
        self.dynamic_graph = dynamic_graph
        self.agents = None
        self.np_random = np_random
        self.is_testing = is_testing
        self.pre_move_graph = None
        self.pre_move_agents = None
        if self.is_testing:
            self.graphs = cycle(glob.glob(f"graph_topologies/testing_{self.num_agents}/*"))
        else:
            self.graphs = glob.glob(f"graph_topologies/training_{self.num_agents}/*")
        self.tested_agent = 0
        self.reset()

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
            relay_indices = np.where(agent.suppl_mpr)[0]

            for index in relay_indices:
                self.agents[index].state.relays_for[agent.id] = 1

        for agent in self.scripted_agents:
            agent.action = 0

            # Check if the agent has taken an action or received a message
            if not agent.state.has_taken_action and (
                    sum(agent.state.received_from) or agent.state.message_origin):
                # Check the specific conditions to update the action to 1
                if agent.has_received_from_relayed_node() or agent.state.message_origin:
                    agent.action = 1
                agent.state.has_taken_action = 1

        # Send message
        for agent in self.agents:
            logging.debug(f"Agent {agent.name} Action: {agent.action} "
                          f"with Neigh: {agent.one_hop_neighbours_ids}")
            self.update_agent_state(agent)

        # Updates nodes positions and edges if the graph is dynamic
        if self.dynamic_graph:
            self.move_graph()

        # Features of agents are updated
        for agent in self.agents:
            self.update_agent_features(agent)

        # Local graph of every agent is updated
        for agent in self.agents:
            self.update_local_graph(agent)

        # Reset MPR Policies
        for agent in self.scripted_agents:
            agent.state.relays_for = np.zeros(self.num_agents)

    def update_agent_state(self, agent):
        # If it has received from a relay node or is message origin
        # and has not already transmitted the message
        if agent.action:
            agent.state.transmitted_to += agent.one_hop_neighbours_ids
            self.messages_transmitted += 1
            agent.messages_transmitted += 1
            agent.actions_history[agent.steps_taken - 1] = agent.action
            neighbour_indices = np.where(agent.one_hop_neighbours_ids)[0]
            for index in neighbour_indices:
                self.agents[index].state.received_from[agent.id] += 1

    def update_agent_features(self, agent):
        # Update graph
        self.graph.nodes[agent.id]['features_critic'] = [
            sum(agent.one_hop_neighbours_ids),
            agent.messages_transmitted,
            agent.steps_taken
        ]

        self.graph.nodes[agent.id]['features_actor'] = [
            sum(agent.one_hop_neighbours_ids),
            agent.messages_transmitted,
            agent.action if agent.action is not None else 0
        ]


    def update_local_graph(self, agent):
        agent.update_local_view(
            local_view=nx.ego_graph(self.graph, agent.id,
                                    undirected=True))

        agent.update_two_hop_cover_from_one_hopper(self.agents)

    def move_graph(self):
        # Graph and agent state is saved for visualization
        self.pre_move_graph = self.graph.copy()
        self.pre_move_agents = copy.deepcopy(self.agents)

        # Move nodes and compute new edges
        self.update_position(step=constants.NODES_MOVEMENT_STEP)

        # Update agent embedded data after the graph movement
        for agent in self.agents:
            self.update_one_hop_neighbors(agent)
        for agent in self.agents:
            self.update_two_hop_neighbors(agent)

    def update_position(self, step):
        # Get current node positions
        pos = nx.get_node_attributes(self.graph, "pos")

        # Given the step size, compute the x and y movement for each agent
        offset_x, offset_y = self.compute_random_movement(step)

        # Update positions of the agents
        pos = {k: [v[0] + offset_x[k], v[1] + offset_y[k]] for k, v in pos.items()}
        nx.set_node_attributes(self.graph, pos, "pos")
        for i in range(self.num_agents):
            self.agents[i].pos[0] += offset_x[i]
            self.agents[i].pos[1] += offset_y[i]

        # Given the new positions, calculate the edges and update the graph
        new_edges = nx.geometric_edges(self.graph, radius=constants.RADIUS_OF_INFLUENCE)
        old_edges = list(self.graph.edges())
        self.graph.remove_edges_from(old_edges)
        self.graph.add_edges_from(new_edges)

    def update_one_hop_neighbors(self, agent):
        # Initialize one hop neighbors to zeros
        one_hop_neighbours_ids = np.zeros(self.num_agents)

        # Compute the neighbors one hop and save the information in an array
        for agent_index in self.graph.neighbors(agent.id):
            one_hop_neighbours_ids[agent_index] = 1

        # One hop neighbors information is stored into the nodes of the graph
        self.graph.nodes[agent.id]['one_hop'] = one_hop_neighbours_ids
        self.graph.nodes[agent.id]['one_hop_list'] = [x for x in self.graph.neighbors(agent.id)]

        # One hop neigh field is updated here
        agent.one_hop_neighbours_ids = one_hop_neighbours_ids

    def update_two_hop_neighbors(self, agent):
        # One hop neighbors are two hop neighbors as well
        agent.two_hop_neighbours_ids = agent.one_hop_neighbours_ids

        # Calculate two hop neighbors
        for agent_index in self.graph.neighbors(agent.id):
            agent.two_hop_neighbours_ids = np.logical_or(
                self.graph.nodes[agent_index]['one_hop'].astype(int),
                agent.two_hop_neighbours_ids.astype(int)
            ).astype(int)

        # Agent can't be two hop neighbor of himself
        agent.two_hop_neighbours_ids[agent.id] = 0

    # Method that calculate random movement for the agents if the graph is dynamic
    def compute_random_movement(self, step):
        ox = [step * self.np_random.uniform(-1, 1) for _ in range(self.num_agents)]
        oy = [step * self.np_random.uniform(-1, 1) for _ in range(self.num_agents)]
        return ox, oy

    def reset(self):
        if self.random_graph:
            self.graph = create_connected_graph(n=self.num_agents, radius=self.radius)
        elif not self.is_graph_fixed:
            if self.is_testing:
                if self.tested_agent == 0:
                    self.graph = load_graph(next(self.graphs))
            else:
                self.graph = load_graph(
                    self.np_random.choice(self.graphs, replace=True)
                )

        self.agents = [Agent(i,
                             nx.ego_graph(self.graph, i, undirected=True),
                             is_scripted=self.is_scripted)
                       for i in range(self.num_agents)]
        # Includes origin message
        self.messages_transmitted = 0
        random_agent = self.agents[self.tested_agent] if self.is_testing else self.np_random.choice(self.agents)
        self.origin_agent = random_agent.id

        if not self.is_graph_fixed and self.is_testing:
            self.tested_agent += 1
            if self.tested_agent == self.num_agents:
                self.tested_agent = 0

        for agent in self.agents:
            agent.state.reset(self.num_agents)
            self.update_one_hop_neighbors(agent)
            self.graph.nodes[agent.id]['features_actor'] = np.zeros((6,))
            self.graph.nodes[agent.id]['features_critic'] = np.zeros((7,))
            self.graph.nodes[agent.id]['label'] = agent.id

        actions_dim = np.ones(2)
        for agent in self.agents:
            agent.reset(local_view=nx.ego_graph(self.graph,
                                                agent.id,
                                                undirected=True),
                        pos=self.graph.nodes[agent.id]['pos'],
                        one_hop_neighbours_ids=agent.one_hop_neighbours_ids
                        )

            self.update_two_hop_neighbors(agent)

            agent.allowed_actions = [True] * int(np.sum(actions_dim))
            agent.steps_taken = 0
            agent.truncated = False

            self.update_agent_features(agent)

        for agent in self.agents:
            self.update_local_graph(agent)

        random_agent.state.message_origin = 1
        random_agent.action = 1
        random_agent.steps_taken += 1

        self.step()


def create_connected_graph(n, radius):
    while True:
        graph = nx.random_geometric_graph(n=n, radius=radius)
        if nx.is_connected(graph):
            break
        else:
            logging.debug("Generated graph not connected, retry")

    return graph


def load_graph(path="testing_graph.gpickle"):
    with open(path, "rb") as input_file:
        graph = pickle.load(input_file)
    return graph


def save_graph(graph, path="testing_graph.gpickle"):
    with open(path, "wb") as output_file:
        pickle.dump(graph, output_file)
