import pytest
import networkx as nx
from gymnasium.utils import seeding
from graph_env.env.utils.core import Agent, World, State
import numpy as np
import unittest.mock as mock


# Initialize world
@pytest.fixture
def world():
    def _world(graph):
        np_random, seed = seeding.np_random(9)
        return World(number_of_agents=12,
                     radius=.20,
                     np_random=np_random,
                     graph=graph)

    return _world


# Initialize graph
@pytest.fixture
def graph():
    graph = nx.Graph()
    graph.add_edges_from(
        [(0, 1), (0, 2), (0, 3), (0, 4), (3, 4), (2, 5), (2, 6), (3, 7), (7, 8), (7, 9), (8, 9), (4, 11), (3, 10)])
    for node in graph.nodes:
        graph.nodes[node]['pos'] = (0, 0)

    return graph


# Initialize agent state
@pytest.fixture
def agent_state():
    state = State()
    state.received_from = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    state.transmitted_to = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    state.relays_for = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    state.relayed_by = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    return state


# Factory as a fixture paradigm
@pytest.fixture
def relaying_agent_state():
    def _relaying_agent_state(state: State,
                              relays_for: list,
                              received_from: list):
        state.relays_for = relays_for
        state.received_from = received_from

        return state

    return _relaying_agent_state


# Initialize agent
@pytest.fixture
def agent():
    def _agent(state, graph):
        return Agent(agent_id=0,
                     local_view=nx.ego_graph(graph, 0, undirected=True),
                     state=state)

    return _agent


# Mock method to replace update_position()
def new_update_position(self, step):
    self.graph.remove_edges_from(list(self.graph.edges()))
    self.graph.add_edges_from(
        [(0, 1), (0, 3), (1, 5), (2, 3), (2, 5), (2, 6), (5, 6), (3, 4), (3, 7), (7, 8), (4, 11), (3, 10), (10, 11)])


# Agent test suite
class TestAgent:
    # test case for agent receiving from relayed node
    def test_has_received_form_relayed_node(self,
                                            agent,
                                            agent_state,
                                            relaying_agent_state,
                                            graph):
        state = relaying_agent_state(agent_state, [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        assert agent(state, graph).has_received_from_relayed_node()

        state = relaying_agent_state(agent_state, [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        assert not agent(state, graph).has_received_from_relayed_node()


# World test suite
class TestWorld:
    # Testing one_hop_neighbours method
    def test_one_hop_neighbours(self, world, graph):
        world = world(graph)
        assert (world.agents[0].one_hop_neighbours_ids == np.array([0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])).all()
        assert (world.agents[1].one_hop_neighbours_ids == np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])).all()
        assert (world.agents[2].one_hop_neighbours_ids == np.array([1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0])).all()
        assert (world.agents[3].one_hop_neighbours_ids == np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0])).all()
        assert (world.agents[4].one_hop_neighbours_ids == np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1])).all()
        assert (world.agents[5].one_hop_neighbours_ids == np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])).all()
        assert (world.agents[6].one_hop_neighbours_ids == np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])).all()
        assert (world.agents[7].one_hop_neighbours_ids == np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0])).all()
        assert (world.agents[8].one_hop_neighbours_ids == np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0])).all()
        assert (world.agents[9].one_hop_neighbours_ids == np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0])).all()
        assert (world.agents[10].one_hop_neighbours_ids == np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])).all()
        assert (world.agents[11].one_hop_neighbours_ids == np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])).all()

    # Testing update_one_hop_neighbours method
    def test_update_one_hop_neighbours(self, world, graph):
        with mock.patch.object(World, 'update_position', new=new_update_position):
            world = world(graph)
            world.update_position(step=1)
            for agent in world.agents:
                world.update_one_hop_neighbors_info(agent)
            assert (world.agents[0].one_hop_neighbours_ids == np.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])).all()
            assert (world.agents[1].one_hop_neighbours_ids == np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])).all()
            assert (world.agents[2].one_hop_neighbours_ids == np.array([0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0])).all()
            assert (world.agents[3].one_hop_neighbours_ids == np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0])).all()
            assert (world.agents[4].one_hop_neighbours_ids == np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1])).all()
            assert (world.agents[5].one_hop_neighbours_ids == np.array([0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0])).all()
            assert (world.agents[6].one_hop_neighbours_ids == np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0])).all()
            assert (world.agents[7].one_hop_neighbours_ids == np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0])).all()
            assert (world.agents[8].one_hop_neighbours_ids == np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])).all()
            assert (world.agents[9].one_hop_neighbours_ids == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])).all()
            assert (world.agents[10].one_hop_neighbours_ids == np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1])).all()
            assert (world.agents[11].one_hop_neighbours_ids == np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0])).all()

    # Testing two_hop_neighbours method
    def test_two_hop_neighbours(self, world, graph):
        world = world(graph)
        assert (world.agents[0].two_hop_neighbours_ids == np.array([0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1])).all()
        assert (world.agents[1].two_hop_neighbours_ids == np.array([1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])).all()
        assert (world.agents[2].two_hop_neighbours_ids == np.array([1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0])).all()
        assert (world.agents[3].two_hop_neighbours_ids == np.array([1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1])).all()
        assert (world.agents[4].two_hop_neighbours_ids == np.array([1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1])).all()
        assert (world.agents[5].two_hop_neighbours_ids == np.array([1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0])).all()
        assert (world.agents[6].two_hop_neighbours_ids == np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0])).all()
        assert (world.agents[7].two_hop_neighbours_ids == np.array([1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0])).all()
        assert (world.agents[8].two_hop_neighbours_ids == np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0])).all()
        assert (world.agents[9].two_hop_neighbours_ids == np.array([0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0])).all()
        assert (world.agents[10].two_hop_neighbours_ids == np.array([1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0])).all()
        assert (world.agents[11].two_hop_neighbours_ids == np.array([1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0])).all()

    # Testing update_two_hop_neighbours method
    def test_update_two_hop_neighbours(self, world, graph):
        with mock.patch.object(World, 'update_position', new=new_update_position):
            world = world(graph)
            world.update_position(step=1)
            for agent in world.agents:
                world.update_one_hop_neighbors_info(agent)
            for agent in world.agents:
                world.update_two_hop_neighbors_info(agent)
            assert (world.agents[0].one_hop_neighbours_ids == np.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])).all()
            assert (world.agents[0].two_hop_neighbours_ids == np.array([0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0])).all()
            assert (world.agents[1].two_hop_neighbours_ids == np.array([1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0])).all()
            assert (world.agents[2].two_hop_neighbours_ids == np.array([1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0])).all()
            assert (world.agents[3].two_hop_neighbours_ids == np.array([1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1])).all()
            assert (world.agents[4].two_hop_neighbours_ids == np.array([1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1])).all()
            assert (world.agents[5].two_hop_neighbours_ids == np.array([1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0])).all()
            assert (world.agents[6].two_hop_neighbours_ids == np.array([0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0])).all()
            assert (world.agents[7].two_hop_neighbours_ids == np.array([1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0])).all()
            assert (world.agents[8].two_hop_neighbours_ids == np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])).all()
            assert (world.agents[9].two_hop_neighbours_ids == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])).all()
            assert (world.agents[10].two_hop_neighbours_ids == np.array([1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1])).all()
            assert (world.agents[11].two_hop_neighbours_ids == np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0])).all()


class TestWorldHeuristics:
    def test_world_simple_broadcast(self, graph):
        # seed for reproducibility
        np_random, _ = seeding.np_random(42)
        world = World(
            number_of_agents=graph.number_of_nodes(),
            radius=0.2,
            np_random=np_random,
            graph=graph,
            scripted_agents_ratio=1.0,
            heuristic="simple_broadcast"
        )

        src = world.origin_agent
        first_wave = world.agents[src].one_hop_neighbours_ids.copy()
        # it MUST have transmitted exactly to all its one‐hop neighbours
        assert np.array_equal(world.agents[src].state.transmitted_to, first_wave)
        # and each such neighbour must have received exactly one message from src
        for nbr in np.where(first_wave)[0]:
            assert world.agents[nbr].state.received_from[src] == 1

        # --- STEP 2: drive a second propagation ---
        world.step()

        # now **every** agent that got the message in wave 1 should forward it in wave 2
        for relay_agent in np.where(first_wave)[0]:
            transmitted_to = world.agents[relay_agent].state.transmitted_to
            mask = world.agents[relay_agent].one_hop_neighbours_ids
            # that agent's own transmitted_to must equal its one‐hop mask
            # 1) Wherever mask==0, tt must be exactly 0
            assert np.all(transmitted_to[mask == 0] == 0), (
                f"Agent {relay_agent} forwarded to non-neighbours: "
                f"{np.where((mask == 0) & (transmitted_to != 0))[0]}"
            )

            # 2) Wherever mask==1, tt must be ≥1
            assert np.all(transmitted_to[mask == 1] >= 1), (
                f"Agent {relay_agent} failed to forward to some neighbour(s): "
                f"{np.where((mask == 1) & (transmitted_to < 1))[0]}"
            )            # and each of **its** neighbours must have received from relay_agent once

            for nbr2 in np.where(mask)[0]:
                assert world.agents[nbr2].state.received_from[relay_agent] >= 1, \
                    f"Neighbour {nbr2} did not receive from {relay_agent}"

    def test_world_probabilistic_relay_zero(self, graph):
        # with prob=0 nobody beyond the origin should relay
        np_random, _ = seeding.np_random(123)
        world = World(
            number_of_agents=graph.number_of_nodes(),
            radius=0.2,
            np_random=np_random,
            graph=graph,
            is_scripted=True,
            scripted_agents_ratio=1.0,
            heuristic="probabilistic_relay",
            heuristic_params={"prob": 0.0}
        )

        src = world.origin_agent
        # who got the origin’s message in the built-in reset()/step()
        first_wave = world.agents[src].one_hop_neighbours_ids

        # Make two steps such that the first wave of relays is done
        world.step()
        world.step()

        # now each of those first-wave recipients must *not* relay
        for relay_agent in np.where(first_wave)[0]:
            tt = world.agents[relay_agent].state.transmitted_to
            assert np.all(tt == 0), (
                f"Agent {relay_agent} unexpectedly relayed to "
                f"{np.where(tt != 0)[0].tolist()}"
            )

    def test_world_probabilistic_relay_one(self, graph):
        # prob=1 ⇒ first-wave recipients MUST forward to all their neighbours
        np_random, _ = seeding.np_random(999)
        world = World(
            number_of_agents=graph.number_of_nodes(),
            radius=0.2,
            np_random=np_random,
            graph=graph,
            is_scripted=True,
            scripted_agents_ratio=1.0,
            heuristic="probabilistic_relay",
            heuristic_params={"prob": 1.0}
        )

        src = world.origin_agent
        first_wave = world.agents[src].one_hop_neighbours_ids.copy()

        # step 2: first wave -> second wave
        world.step()

        for relay_agent in np.where(first_wave)[0]:
            mask = world.agents[relay_agent].one_hop_neighbours_ids
            tt   = world.agents[relay_agent].state.transmitted_to

            # no spurious forwards
            assert np.all(tt[mask == 0] == 0), (
                f"Agent {relay_agent} forwarded to non-neighbours: "
                f"{np.where((mask==0)&(tt!=0))[0].tolist()}"
            )
            # at least one forward to each true neighbour
            assert np.all(tt[mask == 1] >= 1), (
                f"Agent {relay_agent} failed to forward to neighbours: "
                f"{np.where((mask==1)&(tt<1))[0].tolist()}"
            )
            # and those neighbours must have recorded a receipt
            for nbr in np.where(mask)[0]:
                assert world.agents[nbr].state.received_from[relay_agent] >= 1, (
                    f"Neighbour {nbr} did not receive from {relay_agent}"
                )
