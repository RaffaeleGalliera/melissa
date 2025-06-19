import copy
import functools
import glob
import pickle
import numpy as np
import logging
import networkx as nx
from . import constants


class State:
    def __init__(self):
        self.received_from = None
        self.transmitted_to = None
        self.relays_for = None
        self.relayed_by = None
        self.message_origin = 0
        self.has_taken_action = False
        self.has_message = False

    def reset(self, num_agents):
        self.received_from = np.zeros(num_agents)
        self.transmitted_to = np.zeros(num_agents)
        self.relays_for = np.zeros(num_agents)
        self.relayed_by = np.zeros(num_agents)
        self.message_origin = 0
        self.has_taken_action = False
        self.has_message = False


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
            heuristic_fn=None,
            is_interested=False,
            is_scripted=False
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
        self.heuristic_fn = heuristic_fn
        self.action_callback = None
        self.suppl_mpr = None
        self.steps_taken = None
        self.truncated = None
        self.messages_transmitted = 0
        self.number_interested_neighbors = 0
        self.actions_history = np.zeros((4,))
        self.is_scripted = is_scripted
        self.is_interested = is_interested

    def reset(self, local_view, pos, one_hop_neighbors_ids):
        self.__init__(
            agent_id=self.id,
            local_view=local_view,
            state=self.state,
            pos=pos,
            one_hop_neighbors_ids=one_hop_neighbors_ids,
            heuristic_fn=self.heuristic_fn,
            is_interested=self.is_interested,
            is_scripted=self.is_scripted
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
            if (agents[idx].state.has_message or agents[idx].state.message_origin == 1)
        )
        self.gained_two_hop_cover = new_two_hop_cover - self.two_hop_cover
        self.two_hop_cover = new_two_hop_cover

from .heuristics import HEURISTIC_REGISTRY, HeuristicResult

class World:
    def __init__(
            self,
            number_of_agents,
            radius,
            np_random,
            graph=None,
            is_testing=False,
            random_graph=False,
            dynamic_graph=False,
            all_agents_source=False,
            num_test_episodes=10,
            fixed_interest_density=None,
            heuristic: str = None,
            heuristic_params: dict = None,
            scripted_agents_ratio: float = 0.0,
            **kwargs
    ):
        self.selected_graph = None
        self.messages_transmitted = 0
        self.origin_agent = None
        self.num_agents = number_of_agents
        self.radius = radius
        self.graph = graph
        self.all_agents_source = all_agents_source
        self.is_graph_fixed = (graph is not None)
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
        if not (0.0 <= scripted_agents_ratio <= 1.0):
            raise ValueError("`scripted_agents_ratio` must be in [0.0, 1.0].")
        elif scripted_agents_ratio == 0.0 and heuristic is not None:
            raise ValueError("If `scripted_agents_ratio` is 0.0, no heuristic can be set.")
        self.scripted_agents_ratio = scripted_agents_ratio
        self.scripted_indices = set()

        if heuristic:
            if heuristic not in HEURISTIC_REGISTRY:
                raise ValueError(f"Unknown heuristic policy: {heuristic}")
            if not callable(HEURISTIC_REGISTRY[heuristic]):
                raise ValueError(f"Policy {heuristic} is not callable.")
            if heuristic_params is not None and not isinstance(heuristic_params, dict):
                raise ValueError("Heuristic parameters must be a dictionary.")
            else:
                self.heuristic_fn = functools.partial(
                    HEURISTIC_REGISTRY[heuristic],
                    **(heuristic_params or {})
                )
        else:
            self.heuristic_fn = None

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

    def _sample_scripted_agents(self) -> np.ndarray:
        """
        Return an array of vertex indices that should be scripted.
        """
        n_scripted = int(round(self.scripted_agents_ratio * self.num_agents))
        scripted = set(
            self.np_random.choice(
                self.num_agents,
                size=n_scripted,
                replace=False
            )
        )

        if self.scripted_agents_ratio < 1.0:
            # Make sure origin agent is not scripted
            scripted.discard(self.origin_agent)

        return np.fromiter(scripted, dtype=int)

    def _apply_scripted_mask(self):
        """
        Annotate every ``Agent`` with ``is_scripted``.
        """
        self.scripted_indices = set(int(i) for i in self._sample_scripted_agents())

    def step(self):
        for agent in self.scripted_agents:
            result: HeuristicResult = agent.action_callback(agent=agent)

            if result.relay_mask is not None:
                for nbr in np.where(result.relay_mask)[0]:
                    self.agents[nbr].state.relays_for[agent.id] = 1

            if result.action is not None:
                agent.action = result.action

        for agent in self.scripted_agents:
            if np.where(agent.state.relays_for)[0].size > 0:
                agent.action = 0
                if (not agent.state.has_taken_action) and (
                        agent.state.has_message or agent.state.message_origin
                ):
                    if agent.has_received_from_relayed_node() or agent.state.message_origin:
                        agent.action = 1

        # Overwrite action for source agent at first round: it will always broadcast
        self.agents[self.origin_agent].action = 1 if self.agents[self.origin_agent].messages_transmitted == 0 else self.agents[self.origin_agent].action

        # Update environment logic
        for agent in self.agents:
            logging.debug(
                f"Agent {agent.name} Action: {agent.action} with Neigh: {agent.one_hop_neighbours_ids}"
            )
            if agent.action and agent.state.has_message:
                self.relay_message(agent)

        if self.dynamic_graph:
            self.move_graph()

        # Update local graphs
        for agent in self.agents:
            self.update_local_graph(agent)

        # Clear MPR for next step and action
        for agent in self.scripted_agents:
            agent.state.relays_for = np.zeros(self.num_agents)
            agent.action = 0

    def relay_message(self, agent):
        agent.state.transmitted_to += agent.one_hop_neighbours_ids
        self.messages_transmitted += 1
        agent.messages_transmitted += 1
        agent.state.has_taken_action = True
        if agent.steps_taken is not None and agent.steps_taken > 0:
            agent.actions_history[agent.steps_taken - 1] = agent.action

        neighbor_indices = np.where(agent.one_hop_neighbours_ids)[0]
        for idx in neighbor_indices:
            self.agents[idx].state.received_from[agent.id] += 1
            self.agents[idx].state.has_message = True

    def move_graph(self):
        self.pre_move_graph = self.graph.copy()
        self.pre_move_agents = copy.deepcopy(self.agents)
        self.update_position(step=constants.NODES_MOVEMENT_STEP)

        for agent in self.agents:
            self.update_one_hop_neighbors_info(agent)
        for agent in self.agents:
            self.update_two_hop_neighbors_info(agent)

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

    def update_one_hop_neighbors_info(self, agent):
        one_hop_neighbours_ids = np.zeros(self.num_agents)
        agent.number_interested_neighbors = 0

        for agent_index in self.graph.neighbors(agent.id):
            one_hop_neighbours_ids[agent_index] = 1
            if self.agents[agent_index].is_interested:
                agent.number_interested_neighbors += 1

        self.graph.nodes[agent.id]['one_hop'] = one_hop_neighbours_ids
        self.graph.nodes[agent.id]['one_hop_list'] = list(self.graph.neighbors(agent.id))
        agent.one_hop_neighbours_ids = one_hop_neighbours_ids

    def update_two_hop_neighbors_info(self, agent):
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
            fixed_interest_densities = [i / 10.0 for i in range(1, 11)]
            interest_density = fixed_interest_densities[self.test_episode_index % len(fixed_interest_densities)] if self.fixed_interest_density is None else self.fixed_interest_density
            print(
                f"Testing episode {self.test_episode_index}, seed {episode_seed}, "
                f"graph {selected_graph_path}, interest density {interest_density}"
            )
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
            interest_density = ep_rng.uniform(0.1, 1.0) if self.fixed_interest_density is None else self.fixed_interest_density


        self.agents = []
        self.messages_transmitted = 0
        self.origin_agent = chosen_source_id

        # Assign interest to a fraction of agents
        num_interested = int(interest_density * self.num_agents)
        interested_indices = ep_rng.choice(self.num_agents, size=num_interested, replace=False)
        self._apply_scripted_mask()

        # Build agents
        for i in range(self.num_agents):
            local_graph = nx.ego_graph(self.graph, i, undirected=True)
            is_int = (i in interested_indices)
            is_scripted = (i in self.scripted_indices)
            new_agent = Agent(
                i,
                local_graph,
                heuristic_fn=self.heuristic_fn if is_scripted else None,
                is_interested=is_int,
                is_scripted=is_scripted,
            )
            self.agents.append(new_agent)

        # Initialize each agent's arrays and neighbors
        for agent in self.agents:
            agent.state.reset(self.num_agents)
            self.update_one_hop_neighbors_info(agent)
            self.graph.nodes[agent.id]['label'] = agent.id

        for agent in self.agents:
            agent.reset(
                local_view=nx.ego_graph(self.graph, agent.id, undirected=True),
                pos=self.graph.nodes[agent.id]['pos'],
                one_hop_neighbors_ids=agent.one_hop_neighbours_ids
            )
            self.update_two_hop_neighbors_info(agent)
            agent.steps_taken = 0
            agent.truncated = False

        # Set the selected heuristic function for scripted agents
        for agent in self.agents:
            agent.action_callback = self.heuristic_fn if agent.is_scripted else None

        # Mark the source agent: it originates the message
        source_agent = self.agents[self.origin_agent]
        source_agent.state.message_origin = True
        source_agent.state.has_message = True
        source_agent.steps_taken = 1

        self.step()


class InfluenceWorld(World):
    def __init__(self, model: str = "LT", **kwargs):
        self.model = model  # "LT" or "IC"
        self.theta: np.ndarray | None = None          # (N,)
        self.influence_weights: np.ndarray | None = None  # (N,N)
        self.infl_class: np.ndarray | None = None     # (N,N) int8
        super().__init__(**kwargs)

    def _sample_weights_and_thresholds(self):

        """Sample θ and W for the Linear‑Threshold model.

        Steps
        -----
        1. θ_v ∼ Beta(1, 3)  (skew toward easier activation).
        2. Choose for every node **one neighbour** to be its guaranteed
           first‑hop target.  The algorithm tries to keep these targets
           unique; duplicates are allowed only if a node’s entire
           neighbourhood is already claimed.
        3. Draw each column of **W** from a Dirichlet(α=0.2) – produces
           a spiky column that already sums to 1.
        4. Raise every picked edge (x → y) until `W[x,y] > θ_y`, scaling
           *other* entries in the same column so the column sum stays 1.
        5. Final adjustment: ensure both “can‑be‑activated” and
           “can‑activate” guarantees hold.
        """

        rng, G = self.np_random, self.graph
        n = self.num_agents

        # 1. θ ~ Beta(1,3)  (values mostly in 0–0.5)
        theta = rng.beta(1, 3, size=n).astype(np.float32)

        # 2. Pick activation pairs (unique when possible)
        unclaimed = set(range(n))
        pairs = []
        for x in rng.permutation(n):
            neigh = [int(getattr(v, "id", v)) for v in G.neighbors(x)]
            if not neigh:
                raise RuntimeError("LT infeasible: isolated node detected.")
            opts = [v for v in neigh if v in unclaimed]
            y = rng.choice(opts if opts else neigh)
            pairs.append((x, y))
            unclaimed.discard(y)

        # 3. Dirichlet(0.2)
        W = np.zeros((n, n), dtype=np.float32)
        alpha = 0.2
        for v in range(n):
            neigh = [int(getattr(u, "id", u)) for u in G.neighbors(v)]
            weights = rng.dirichlet(alpha=np.full(len(neigh), alpha))
            W[neigh, v] = weights

        # 4. Boost each picked edge above θ_y, preserve column sum = 1
        eps = 1e-6
        for x, y in pairs:
            required = theta[y] + eps
            if W[x, y] >= required:
                continue  # already strong enough

            other_ids = [int(getattr(u, "id", u)) for u in G.neighbors(y) if int(getattr(u, "id", u)) != x]
            other_total = W[other_ids, y].sum() if other_ids else 0.0

            if other_total > eps:
                # scale factor keeps column sum exactly 1 while raising W[x,y]
                diff = required - W[x, y]
                scale = (other_total - diff) / other_total
                W[other_ids, y] *= scale
                W[x, y] = required
                # column sum is now: required + scale*other_total = 1.0  (exact)
            else:
                # x is the only neighbour with weight: give it all
                W[:, y] = 0.0
                W[x, y] = 1.0
                theta[y] = 0.9 * W[x, y]  # ensure activatability to guarantee both directions of property
        for v in range(n):
            max_in = W[:, v].max()
            if max_in <= theta[v]:
                theta[v] = 0.9 * max_in
        for u in range(n):
            v_max = np.argmax(W[u])
            if W[u, v_max] <= theta[v_max]:
                theta[v_max] = 0.9 * W[u, v_max]

                # 6. Final per‑pair guarantee pass (lower θ if still violated)
        for x, y in pairs:
            if W[x, y] <= theta[y] - eps:
                theta[y] = 0.9 * W[x, y]

            # --- invariants ---
            assert W[x, y] > theta[y] - eps

        # 6. Store matrices
        self.influence_weights = W.astype(np.float32)
        self.theta = theta.astype(np.float32)

        # 7. Edge‑class matrix (terciles)
        nz = W[W > 0]
        q1, q2 = np.quantile(nz, [0.33, 0.67]) if nz.size >= 3 else (0, 0)
        infl = np.full_like(W, -1, dtype=np.int8)
        infl[W >= q2] = 2
        infl[(W >= q1) & (W < q2)] = 1
        infl[(W > 0) & (W < q1)] = 0
        np.fill_diagonal(infl, -1)
        self.infl_class = infl

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
            fixed_interest_densities = [i / 10.0 for i in range(1, 11)]
            interest_density = fixed_interest_densities[self.test_episode_index % len(
                fixed_interest_densities)] if self.fixed_interest_density is None else self.fixed_interest_density
            print(
                f"Testing episode {self.test_episode_index}, seed {episode_seed}, "
                f"graph {selected_graph_path}, interest density {interest_density}"
            )
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
            interest_density = ep_rng.uniform(0.1,
                                              1.0) if self.fixed_interest_density is None else self.fixed_interest_density

        self.agents = []
        self.messages_transmitted = 0
        self.origin_agent = chosen_source_id

        # Assign interest to a fraction of agents
        num_interested = int(interest_density * self.num_agents)
        interested_indices = ep_rng.choice(self.num_agents, size=num_interested, replace=False)
        self._apply_scripted_mask()

        # Build agents
        for i in range(self.num_agents):
            local_graph = nx.ego_graph(self.graph, i, undirected=True)
            is_int = (i in interested_indices)
            is_scripted = (i in self.scripted_indices)
            new_agent = Agent(
                i,
                local_graph,
                heuristic_fn=self.heuristic_fn if is_scripted else None,
                is_interested=is_int,
                is_scripted=is_scripted,
            )
            self.agents.append(new_agent)

        # Initialize each agent's arrays and neighbors
        for agent in self.agents:
            agent.state.reset(self.num_agents)
            self.update_one_hop_neighbors_info(agent)
            self.graph.nodes[agent.id]['label'] = agent.id

        for agent in self.agents:
            agent.reset(
                local_view=nx.ego_graph(self.graph, agent.id, undirected=True),
                pos=self.graph.nodes[agent.id]['pos'],
                one_hop_neighbors_ids=agent.one_hop_neighbours_ids
            )
            self.update_two_hop_neighbors_info(agent)
            agent.steps_taken = 0
            agent.truncated = False

        # Set the selected heuristic function for scripted agents
        for agent in self.agents:
            agent.action_callback = self.heuristic_fn if agent.is_scripted else None

        # Mark the source agent: it originates the message
        source_agent = self.agents[self.origin_agent]
        source_agent.state.message_origin = True
        source_agent.state.has_message = True
        source_agent.steps_taken = 1

        # Sample influence weights and thresholds
        self._sample_weights_and_thresholds()

        self.step()

    def step(self):
        """One environment tick implementing LT / IC diffusion on the
        *current* graph snapshot.  Agents' `action` ∈ {0 (silent), 1 (forward)}
        is set by external RL policy or scripted logic.
        """
        for agent in self.scripted_agents:
            pass  # placeholder – keep whatever the parent does

        self.agents[self.origin_agent].action = 1 if self.agents[self.origin_agent].messages_transmitted == 0 else self.agents[self.origin_agent].action
        broadcasters = [a.id for a in self.agents if a.state.has_message and a.action == 1]

        if self.dynamic_graph:
            self.move_graph()

        for ag in self.agents:
            v = ag.id
            # sum weights from visible broadcasters
            inc = [u for u in broadcasters if self.graph.has_edge(u, v)]
            if not inc:
                continue
            if self.model == "LT":
                total_inf = self.influence_weights[inc, v].sum()
                for u in inc:
                    if total_inf >= self.theta[v]:
                        ag.state.received_from[u] += 1
                        ag.state.has_message = True
                        self.agents[u].state.transmitted_to[v] += 1
                    self.agents[u].messages_transmitted += 1
                self.messages_transmitted += 1
            else:  # IC
                for u in inc:
                    if self.np_random.random() < self.influence_weights[u, v]:
                        ag.state.has_message = True
                        break

        for agent in self.agents:
            self.update_local_graph(agent)

        for agent in self.scripted_agents:
            agent.state.relays_for.fill(0)
            agent.action = 0


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
