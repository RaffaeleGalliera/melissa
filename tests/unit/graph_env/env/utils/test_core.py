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

    # Testing update_local_graph method
    def test_update_local_graph(self, world, graph):
        with mock.patch.object(World, 'update_position', new=new_update_position):
            # Initialize world and set agent
            world = world(graph)
            agent = world.agents[0]

            # Check if local graph of agent chosen is correct
            world.update_local_graph(agent)
            g1 = nx.Graph()
            g1.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), (3, 4)])
            assert (nx.is_isomorphic(agent.local_view, g1))

            # Move the graph
            world.move_graph()

            # Check if local graph of agent chosen is still correct
            world.update_local_graph(agent)
            g2 = nx.Graph()
            g2.add_edges_from([(0, 1), (0, 3)])
            assert (nx.is_isomorphic(agent.local_view, g2))

    # Testing update_agent_state method when graph is static and agent doesn't cast message
    def test_update_agent_static_noaction(self, world, graph):
        with mock.patch.object(World, 'update_position', new=new_update_position):
            # Initialize world and set agent
            world = world(graph)
            agent = world.agents[0]
            # Agent doesn't act
            agent.action = 0

            # Update agent state and features
            world.update_agent_state(agent)
            world.update_agent_features(agent)

            # Check agent state is correct after the update
            assert (agent.state.transmitted_to == [0] * graph.number_of_nodes()).all()
            assert (agent.messages_transmitted == 0)
            assert (agent.actions_history[agent.steps_taken - 1] == 0)
            neighbour_indices = np.where(agent.one_hop_neighbours_ids)[0]
            for index in neighbour_indices:
                assert (world.agents[index].state.received_from[agent.id] == 0)
            assert (graph.nodes[agent.id]['features'] == [4, 0, 0, 0, 0, 0, 0]).all()

    # Testing update_agent_state method when graph is dynamic and agent doesn't cast message
    def test_update_agent_state_dynamic_noaction(self, world, graph):
        with mock.patch.object(World, 'update_position', new=new_update_position):
            # Initialize world and set first agent
            world = world(graph)
            agent = world.agents[0]
            # First agent act
            agent.action = 1

            # Update agent state and features, graph is moved
            world.update_agent_state(agent)
            world.move_graph()
            world.update_agent_features(agent)

            # Agent is set
            agent = world.agents[1]
            # Agent doesn't act
            agent.action = 0

            # Update agent state and features again
            world.update_agent_state(agent)
            world.update_agent_features(agent)

            # Check agent state is correct after the update
            assert (agent.state.transmitted_to == [0] * graph.number_of_nodes()).all()
            assert (agent.messages_transmitted == 0)
            assert (agent.actions_history[agent.steps_taken - 1] == 0)
            neighbour_indices = np.where(agent.one_hop_neighbours_ids)[0]
            for index in neighbour_indices:
                assert (world.agents[index].state.received_from[agent.id] == 0)
            assert (graph.nodes[agent.id]['features'] == [2, 0, 0, 0, 0, 0, 0]).all()

    # Testing update_agent_state method when graph is static and agent casts message
    def test_update_agent_state_static_action(self, world, graph):
        with mock.patch.object(World, 'update_position', new=new_update_position):
            # Initialize world and set agent
            world = world(graph)
            agent = world.agents[0]
            # Agent acts
            agent.action = 1

            # Update agent state and features
            world.update_agent_state(agent)
            world.update_agent_features(agent)

            # Check agent state is correct after the update
            assert (agent.state.transmitted_to == [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]).all()
            assert (agent.messages_transmitted == 1)
            assert (agent.actions_history[agent.steps_taken - 1] == 1)
            neighbour_indices = np.where(agent.one_hop_neighbours_ids)[0]
            for index in neighbour_indices:
                assert (world.agents[index].state.received_from[agent.id] == 1)
            assert (graph.nodes[agent.id]['features'] == [4, 1, 0, 0, 0, 0, 1]).all()

    # Testing update_agent_state method when graph is dynamic and agent casts message
    def test_update_agent_state_dynamic_action(self, world, graph):
        with mock.patch.object(World, 'update_position', new=new_update_position):
            # Initialize world and set first agent
            world = world(graph)
            agent = world.agents[0]
            # First agent act
            agent.action = 1

            # Update agent state and features, graph is moved
            world.update_agent_state(agent)
            world.move_graph()
            world.update_agent_features(agent)

            # Agent is set
            agent = world.agents[1]
            # Agent acts
            agent.action = 1

            # Update agent state and features again, graph is moved
            world.update_agent_state(agent)
            world.update_agent_features(agent)

            # Check agent state is correct after the update
            assert (agent.state.transmitted_to == [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]).all()
            assert (agent.messages_transmitted == 1)
            assert (agent.actions_history[agent.steps_taken - 1] == 1)
            neighbour_indices = np.where(agent.one_hop_neighbours_ids)[0]
            for index in neighbour_indices:
                assert (world.agents[index].state.received_from[agent.id] == 1)
            assert (graph.nodes[agent.id]['features'] == [2, 1, 0, 0, 0, 0, 1]).all()

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
                world.update_one_hop_neighbors(agent)
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
                world.update_one_hop_neighbors(agent)
            for agent in world.agents:
                world.update_two_hop_neighbors(agent)
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

    # Testing scenario in which graph updates and changes connections before agent sends message
    def test_send_while_graph_moves(self, world, graph):
        with mock.patch.object(World, 'update_position', new=new_update_position):
            # Initialize world
            world = world(graph)
            # Set first agent casting message
            agent = world.agents[7]
            agent.action = 1

            # Update agent state and features, graph is moved
            world.update_agent_state(agent)
            world.move_graph()
            world.update_agent_features(agent)

            # Check states and features of transmitter and receivers
            assert (world.agents[7].state.transmitted_to == [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0]).all()
            assert (graph.nodes[agent.id]['features'] == [2, 1, 0, 0, 0, 0, 1]).all()
            assert (world.agents[3].state.received_from == [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]).all()
            assert (world.agents[8].state.received_from == [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]).all()
            assert (world.agents[9].state.received_from == [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]).all()

            # Set second agent not casting any message
            agent = world.agents[8]
            agent.action = 0
            # Update agent state and features
            world.update_agent_state(agent)
            world.update_agent_features(agent)

            # Second agent finally cast message
            agent.action = 1
            world.update_agent_state(agent)
            world.update_agent_features(agent)

            # Check states and features of transmitter and receivers
            assert (world.agents[8].state.transmitted_to == [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]).all()
            assert (graph.nodes[agent.id]['features'] == [1, 1, 0, 0, 0, 0, 1]).all()
            assert (world.agents[7].state.received_from == [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]).all()
            assert (world.agents[9].state.received_from == [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]).all()
