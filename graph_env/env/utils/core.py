import copy
import glob
import pickle
import numpy as np
import logging
import networkx as nx
from . import constants


# OLSRv1 MPR computation https://www.rfc-editor.org/rfc/rfc3626.html
def mpr_heuristic(
        one_hop_neighbours_ids,
        two_hop_neighbours_ids,
        agent_id,
        local_view
):
    """
    OLSRv1 MPR computation as before.
    This is invoked for scripted agents in World.step().
    """
    two_hop_neighbours_ids = two_hop_neighbours_ids - one_hop_neighbours_ids
    d_y = dict()
    two_hop_coverage_summary = {
        index: [] for index, value in enumerate(two_hop_neighbours_ids) if value == 1
    }
    covered = np.zeros(len(one_hop_neighbours_ids))
    mpr = np.zeros(len(one_hop_neighbours_ids))

    for neighbor in local_view.neighbors(agent_id):
        clean_neighbor_neighbors = local_view.nodes[neighbor]['one_hop'].copy()
        # Exclude from the list the 2-hop neighbours already reachable by self
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

    # Add to covered if that neighbor is the unique provider
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
        self.has_taken_action = 0
        self.has_message = 0

    def reset(self, num_agents):
        self.received_from = np.zeros(num_agents)
        self.transmitted_to = np.zeros(num_agents)
        self.relays_for = np.zeros(num_agents)
        self.relayed_by = np.zeros(num_agents)
        self.message_origin = 0
        self.has_taken_action = 0
        self.has_message = 0


class Agent:
    def __init__(
            self,
            agent_id,
            local_view,
            size=0.05,
            color=(0, 0, 0),
            state=None,
            pos=None,
            one_hop_neighbors_ids=None,
            is_scripted=False,
            is_interested=False
    ):
        self.id = agent_id
        self.name = str(agent_id)
        self.state = State() if state is None else state
        # local_view is used by scripted heuristics
        self.local_view = local_view

        self.size = size
        self.pos = pos
        self.color = color

        self.one_hop_neighbours_ids = one_hop_neighbors_ids
        self.two_hop_neighbours_ids = None
        self.two_hop_cover = 0
        self.gained_two_hop_cover = 0

        self.action = None
        self.is_scripted = is_scripted
        self.action_callback = mpr_heuristic if self.is_scripted else None
        self.suppl_mpr = None

        self.steps_taken = None
        self.truncated = None
        self.messages_transmitted = 0
        self.actions_history = np.zeros((4,))

        self.is_interested = is_interested

    def reset(self, local_view, pos, one_hop_neighbors_ids):
        self.__init__(
            agent_id=self.id,
            local_view=local_view,
            state=self.state,
            pos=pos,
            one_hop_neighbors_ids=one_hop_neighbors_ids,
            is_scripted=self.is_scripted,
            is_interested=self.is_interested
        )

    def update_local_view(self, local_view):
        self.local_view = local_view

    def has_received_from_relayed_node(self):
        return sum([
            1 for received, relay in zip(self.state.received_from, self.state.relays_for)
            if received and relay
        ])

    def update_two_hop_cover_from_one_hopper(self, agents):
        """Calculate how many two-hop neighbors have the message."""
        two_hop_neighbor_indices = np.where(self.two_hop_neighbours_ids)[0]
        new_two_hop_cover = sum(
            1 for idx in two_hop_neighbor_indices
            if (agents[idx].state.has_message == 1 or agents[idx].state.message_origin == 1)
        )
        self.gained_two_hop_cover = new_two_hop_cover - self.two_hop_cover
        self.two_hop_cover = new_two_hop_cover


class World:
    def __init__(
            self,
            number_of_agents,
            radius,
            np_random,
            graph=None,
            is_scripted=False,
            is_testing=False,
            random_graph=False,
            dynamic_graph=False,
            all_agents_source=False,
            num_test_episodes=10,
            fixed_interest_density=None
    ):
        self.selected_graph = None
        self.messages_transmitted = 1
        self.origin_agent = None
        self.num_agents = number_of_agents
        self.radius = radius
        self.graph = graph
        self.all_agents_source = all_agents_source
        self.is_graph_fixed = (graph is not None)
        self.is_scripted = is_scripted
        self.random_graph = random_graph
        self.dynamic_graph = dynamic_graph
        self.agents = None
        self.np_random = np_random
        self.is_testing = is_testing
        self.pre_move_graph = None
        self.pre_move_agents = None
        self.movement_np_random = None
        self.movement_np_random_state = None
        self.max_node_degree = constants.MAX_NODE_DEGREE

        self.fixed_interest_density = fixed_interest_density

        if self.is_testing:
            if self.max_node_degree:
                self.test_graphs = sorted(glob.glob(
                    f"graph_topologies/testing_{self.num_agents}_{self.max_node_degree}max/*"
                ))
            else:
                self.test_graphs = sorted(glob.glob(
                    f"graph_topologies/testing_{self.num_agents}/*"
                ))
        else:
            self.train_graphs = glob.glob(f"graph_topologies/training_{self.num_agents}/*")

        # We store a sequence of test seeds for reproducibility
        self.num_test_episodes = num_test_episodes
        self.test_seeds_list = []
        self.test_episode_index = 0

        if self.is_testing:
            # Generate a list of per-episode seeds from the main environment RNG
            testing_generator = np.random.RandomState(17)
            self.test_seeds_list = [
                testing_generator.randint(0, 1e9) for _ in range(num_test_episodes)
            ]

        # Do the initial reset
        self.reset()

    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    def step(self):
        # Scripted agents pick actions
        for agent in self.scripted_agents:
            agent.suppl_mpr = agent.action_callback(
                agent.one_hop_neighbours_ids,
                agent.two_hop_neighbours_ids,
                agent.id,
                agent.local_view
            )
            relay_indices = np.where(agent.suppl_mpr)[0]
            for idx in relay_indices:
                self.agents[idx].state.relays_for[agent.id] = 1

        for agent in self.scripted_agents:
            agent.action = 0
            if (not agent.state.has_taken_action) and (
                agent.state.has_message == 1 or agent.state.message_origin
            ):
                if agent.has_received_from_relayed_node() or agent.state.message_origin:
                    agent.action = 1
                agent.state.has_taken_action = 1

        # Update environment logic
        for agent in self.agents:
            logging.debug(
                f"Agent {agent.name} Action: {agent.action} with Neigh: {agent.one_hop_neighbours_ids}"
            )
            self.update_agent_state(agent)

        if self.dynamic_graph:
            self.move_graph()

        # Update local graphs
        for agent in self.agents:
            self.update_local_graph(agent)

        # Clear MPR for next step
        for agent in self.scripted_agents:
            agent.state.relays_for = np.zeros(self.num_agents)

    def update_agent_state(self, agent):
        if agent.action:
            agent.state.transmitted_to += agent.one_hop_neighbours_ids
            self.messages_transmitted += 1
            agent.messages_transmitted += 1
            if agent.steps_taken is not None and agent.steps_taken > 0:
                agent.actions_history[agent.steps_taken - 1] = agent.action

            neighbor_indices = np.where(agent.one_hop_neighbours_ids)[0]
            for idx in neighbor_indices:
                self.agents[idx].state.received_from[agent.id] += 1
                self.agents[idx].state.has_message = 1

    def move_graph(self):
        self.pre_move_graph = self.graph.copy()
        self.pre_move_agents = copy.deepcopy(self.agents)
        self.update_position(step=constants.NODES_MOVEMENT_STEP)

        for agent in self.agents:
            self.update_one_hop_neighbors(agent)
        for agent in self.agents:
            self.update_two_hop_neighbors(agent)

    def update_local_graph(self, agent):
        local_graph = nx.ego_graph(self.graph, agent.id, undirected=True)
        edges = list(local_graph.edges())
        for edge in edges:
            if agent.id not in edge:
                local_graph.remove_edge(*edge)
        agent.update_local_view(local_graph)
        agent.update_two_hop_cover_from_one_hopper(self.agents)

    def update_position(self, step):
        pos = nx.get_node_attributes(self.graph, "pos")
        offset_x, offset_y = self.compute_random_movement(step)
        new_pos = {}
        for k, v in pos.items():
            new_pos[k] = [v[0] + offset_x[k], v[1] + offset_y[k]]
        nx.set_node_attributes(self.graph, new_pos, "pos")

        for i in range(self.num_agents):
            self.agents[i].pos = new_pos[i]

        new_edges = nx.geometric_edges(self.graph, radius=constants.RADIUS_OF_INFLUENCE)
        old_edges = list(self.graph.edges())
        self.graph.remove_edges_from(old_edges)
        self.graph.add_edges_from(new_edges)

    def compute_random_movement(self, step):
        ox = [step * self.movement_np_random.uniform(-1, 1) for _ in range(self.num_agents)]
        oy = [step * self.movement_np_random.uniform(-1, 1) for _ in range(self.num_agents)]
        return ox, oy

    def update_one_hop_neighbors(self, agent):
        one_hop_neighbours_ids = np.zeros(self.num_agents)
        for agent_index in self.graph.neighbors(agent.id):
            one_hop_neighbours_ids[agent_index] = 1
        self.graph.nodes[agent.id]['one_hop'] = one_hop_neighbours_ids
        self.graph.nodes[agent.id]['one_hop_list'] = list(self.graph.neighbors(agent.id))
        agent.one_hop_neighbours_ids = one_hop_neighbours_ids

    def update_two_hop_neighbors(self, agent):
        agent.two_hop_neighbours_ids = agent.one_hop_neighbours_ids.copy()
        for agent_index in self.graph.neighbors(agent.id):
            neighbor_1hop = self.agents[agent_index].one_hop_neighbours_ids
            agent.two_hop_neighbours_ids = np.logical_or(
                agent.two_hop_neighbours_ids, neighbor_1hop
            ).astype(int)
        agent.two_hop_neighbours_ids[agent.id] = 0

    def reset(self):
        """
        If in testing mode, pick from `test_seeds_list` in strict order.
        If in training mode, sample a new seed from self.np_random.
        """
        if self.is_testing:
            if not self.test_seeds_list:
                raise ValueError("No test seeds have been generated! Check num_test_episodes.")
            episode_seed = self.test_seeds_list[self.test_episode_index]
            self.test_episode_index = (self.test_episode_index + 1) % self.num_test_episodes
            ep_rng = np.random.RandomState(episode_seed)

            if not self.test_graphs:
                raise ValueError("No test graphs found!")
            selected_graph_path = ep_rng.choice(self.test_graphs)
            self.selected_graph = load_graph(selected_graph_path)
            self.graph = self.selected_graph

            movement_seed = ep_rng.randint(0, 1e9)
            self.movement_np_random = np.random.RandomState(movement_seed)

            chosen_source_id = ep_rng.randint(0, self.num_agents)
        else:
            episode_seed = self.np_random.integers(0, 1e9)
            ep_rng = np.random.RandomState(episode_seed)

            if self.random_graph:
                self.graph = create_connected_graph(n=self.num_agents, radius=self.radius)
            elif not self.is_graph_fixed:
                self.selected_graph = self.np_random.choice(self.train_graphs, replace=True)
                self.graph = load_graph(self.selected_graph)

            movement_seed = ep_rng.randint(0, 1e9)
            self.movement_np_random = np.random.RandomState(movement_seed)

            chosen_source_id = ep_rng.randint(0, self.num_agents)

        self.agents = []
        self.messages_transmitted = 0
        self.origin_agent = chosen_source_id

        # Assign interest to a fraction of agents
        interest_density = ep_rng.uniform(0.0, 1.0) if self.fixed_interest_density is None else self.fixed_interest_density
        num_interested = int(interest_density * self.num_agents)
        interested_indices = ep_rng.choice(self.num_agents, size=num_interested, replace=False)

        # Build agents
        for i in range(self.num_agents):
            local_graph = nx.ego_graph(self.graph, i, undirected=True)
            is_int = (i in interested_indices)
            new_agent = Agent(
                i,
                local_graph,
                is_scripted=self.is_scripted,
                is_interested=is_int
            )
            self.agents.append(new_agent)

        # Initialize each agent's arrays and neighbors
        for agent in self.agents:
            agent.state.reset(self.num_agents)
            self.update_one_hop_neighbors(agent)
            self.graph.nodes[agent.id]['label'] = agent.id

        for agent in self.agents:
            agent.reset(
                local_view=nx.ego_graph(self.graph, agent.id, undirected=True),
                pos=self.graph.nodes[agent.id]['pos'],
                one_hop_neighbors_ids=agent.one_hop_neighbours_ids
            )
            self.update_two_hop_neighbors(agent)
            agent.steps_taken = 0
            agent.truncated = False

        # Mark the source agent: it originates the message
        source_agent = self.agents[self.origin_agent]
        source_agent.state.message_origin = 1
        source_agent.state.has_message = 1
        source_agent.action = 1
        source_agent.steps_taken = 1

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
    with open(path, "rb") as f:
        return pickle.load(f)


def save_graph(graph, path="testing_graph.gpickle"):
    with open(path, "wb") as f:
        pickle.dump(graph, f)
