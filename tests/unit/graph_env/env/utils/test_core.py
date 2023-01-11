import pytest
import networkx as nx
from gymnasium.utils import seeding
from graph_env.env.utils.core import Agent, MprWorld, State
import numpy as np

@pytest.fixture
def world():
    def _world(graph):
        np_random, seed = seeding.np_random(9)
        return MprWorld(number_of_agents=4,
                        radius=.40,
                        np_random=np_random,
                        graph=graph)

    return _world


@pytest.fixture
def graph():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
    for node in graph.nodes:
        graph.nodes[node]['pos'] = (0, 0)

    return graph


@pytest.fixture
def agent_state():
    state = State()
    state.received_from = [0, 0, 0, 0]
    state.transmitted_to = [0, 0, 0, 0]
    state.relays_for = [0, 0, 0, 0]
    state.relayed_by = [0, 0, 0, 0]
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


# Factory as a fixture paradigm
@pytest.fixture
def agent():
    def _agent(state, graph):
        return Agent(agent_id=0,
                     local_view=nx.ego_graph(graph, 0, undirected=True),
                     state=state)
    return _agent


class TestMprAgent:
    # test case for agent receiving from relayed node
    def test_has_received_form_relayed_node(self,
                                            agent,
                                            agent_state,
                                            relaying_agent_state,
                                            graph):
        state = relaying_agent_state(agent_state, [0, 1, 0, 1], [0, 1, 0, 0])
        assert agent(state, graph).has_received_from_relayed_node()

        state = relaying_agent_state(agent_state, [0, 1, 0, 1], [0, 0, 0, 0])
        assert not agent(state, graph).has_received_from_relayed_node()


class TestMprWorld:
    def test_set_relays(self, world, graph):
        world = world(graph)
        agent = world.agents[0]
        agent.action = [0, 1, 0, 0]
        world.set_relays(agent)
        assert world.agents[1].state.relays_for[agent.id]

    def test_update_agent_state_relay_node(self, world, graph):
        world = world(graph)

        world.agents[0].state.received_from = [0, 0, 0, 1]
        world.agents[0].state.relays_for = [0, 1, 0, 1]

        world.update_agent_state(world.agents[0])
        assert world.agents[1].state.received_from[0]
        # Count origin message as well
        assert world.messages_transmitted == 2

    def test_update_agent_state_message_origin(self, world, graph):
        world = world(graph)
        for agent in world.agents:
            agent.state.message_origin = 0
        world.agents[0].state.message_origin = 1
        world.update_agent_state(world.agents[0])

    def test_one_hop_neighbours(self, world, graph):
        world = world(graph)

        assert (world.agents[0].one_hop_neighbours_ids == np.array([0, 1, 0, 1])).all()
        assert (world.agents[1].one_hop_neighbours_ids == np.array([1, 0, 1, 0])).all()
        assert (world.agents[2].one_hop_neighbours_ids == np.array([0, 1, 0, 1])).all()
        assert (world.agents[3].one_hop_neighbours_ids == np.array([1, 0, 1, 0])).all()

    def test_two_hop_neighbours(self, world, graph):
        world = world(graph)
        assert (world.agents[0].two_hop_neighbours_ids == np.array([0, 0, 1, 0])).all()
        assert (world.agents[1].two_hop_neighbours_ids == np.array([0, 0, 0, 1])).all()
        assert (world.agents[2].two_hop_neighbours_ids == np.array([1, 0, 0, 0])).all()
        assert (world.agents[3].two_hop_neighbours_ids == np.array([0, 1, 0, 0])).all()
