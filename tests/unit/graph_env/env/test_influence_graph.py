import copy
import numpy as np
import networkx as nx
import pytest

from graph_env.env.influence_graph import InfluenceGraph


@pytest.fixture
def env():
    """
    Fresh InfluenceGraph on the static 0-1-2 graph with three agents.
    We call `reset()` once so that θ and W are initialised.
    """
    e = InfluenceGraph(
        model="LT",
        radius=0.2,
        number_of_agents=20,
        dynamic_graph=False,
    )
    e.seed(42)
    e.reset()
    return e


def _world(env):
    """Shortcut to the internal InfluenceWorld."""
    return env.world


def _prep_broadcast(env, broadcasters, uninformed):
    """
    Mark `broadcasters` as having the message and broadcasting (action==1).
    All `uninformed` nodes start without the message.
    """
    world = _world(env)
    for a in world.agents:
        if a.id in broadcasters:
            a.state.has_message = True
            a.action = 1
        else:
            a.state.has_message = False
            a.action = 0
    assert all(world.agents[u].state.has_message for u in broadcasters)
    assert all(not world.agents[v].state.has_message for v in uninformed)


def test_theta_in_unit_interval(env):
    """All thresholds θ_v must lie in [0,1]."""
    theta = _world(env).theta
    assert theta.shape == (20,)
    assert np.all(theta >= 0.0)
    assert np.all(theta <= 1.0)


def test_weight_matrix_properties(env):
    """
    • W is (n,n) with zero diagonal
    • Each column sums ≤ 1  (normalised during reset)
    """
    world = _world(env)
    W = world.influence_weights
    n = world.num_agents

    assert W.shape == (n, n)
    assert np.allclose(np.diag(W), 0.0)

    col_sums = W.sum(axis=0)
    assert np.all(col_sums <= 1.0 + 1e-6)   # tiny FP slack


def test_reset_draws_fresh_parameters(env):
    """
    After a new `reset()` call, θ and W should almost surely change.
    """
    env.seed(42)

    env.reset()
    theta_first  = _world(env).theta.copy()
    W_first      = _world(env).influence_weights.copy()

    env.reset()
    theta_second = _world(env).theta
    W_second     = _world(env).influence_weights

    assert not np.allclose(theta_first, theta_second)
    assert not np.allclose(W_first,   W_second)


def test_same_seed_draws_same_parameters(env):
    """
    After a new `reset()` call, θ and W should almost surely change.
    """
    env.seed(42)

    env.reset()
    theta_first  = _world(env).theta.copy()
    W_first      = _world(env).influence_weights.copy()

    env.seed(42)
    env.reset()
    theta_second = _world(env).theta
    W_second     = _world(env).influence_weights

    assert np.allclose(theta_first, theta_second)
    assert np.allclose(W_first,   W_second)


def test_single_broadcaster_triggers_activation(env):
    """
    Node 0 broadcasts; Node 1 activates iff W[0,1] ≥ θ₁.
    """
    world = _world(env)

    neighbors = np.where(world.agents[0].one_hop_neighbours_ids)[0]
    world.theta[neighbors[0]] = 0.3
    world.influence_weights[0, neighbors[0]] = 0.5   # ≥ θ₁

    _prep_broadcast(env, broadcasters=[world.agents[0].id], uninformed=[neighbors[0], neighbors[1]])
    world.step()

    assert world.agents[neighbors[0]].state.has_message is True
    assert world.agents[neighbors[1]].state.has_message is False


def test_below_threshold_no_activation(env):
    """
    If ∑_u W[u,v] < θ_v, node v must remain uninformed.
    """
    world = _world(env)
    neighbors = np.where(world.agents[0].one_hop_neighbours_ids)[0]
    world.theta[neighbors[0]] = 0.8
    world.influence_weights[0, neighbors[0]] = 0.2  # < θ₁

    _prep_broadcast(env, broadcasters=[world.agents[0].id], uninformed=[neighbors[0], neighbors[1]])
    world.step()

    assert world.agents[1].state.has_message is False


def test_combined_influence_multiple_broadcasters(env):
    """
    Two broadcasters jointly exceed θ₂ → node 2 activates.
    """
    world = _world(env)
    neighbors = np.where(world.agents[0].one_hop_neighbours_ids)[0]
    neighbor_of_neighbor_one = None
    for neighbor in np.where(world.agents[neighbors[0]].one_hop_neighbours_ids)[0]:
        if neighbor == world.agents[0].id:
            continue
        else:
            neighbor_of_neighbor_one = neighbor
            break
    world.graph.add_edge(world.agents[0], neighbor_of_neighbor_one)
    world.theta[neighbor_of_neighbor_one] = 0.55
    world.influence_weights[world.agents[0].id, neighbor_of_neighbor_one] = 0.30
    world.influence_weights[world.agents[neighbors[0]].id, neighbor_of_neighbor_one] = 0.35  # total 0.65 ≥ θ₂

    _prep_broadcast(env, broadcasters=[world.agents[0].id, world.agents[neighbors[0]].id], uninformed=[neighbor_of_neighbor_one])
    world.step()

    assert world.agents[neighbor_of_neighbor_one].state.has_message is True


def test_scripted_agents_counters_cleared(env):
    """
    After `step()`, every scripted agent must have action==0 and
    relays_for cleared.
    """
    world = _world(env)
    if not world.scripted_agents:
        pytest.skip("No scripted agents in this configuration")

    for ag in world.scripted_agents:
        ag.state.has_message = True
        ag.action = 1
        ag.state.relays_for[:] = 1

    world.step()

    for ag in world.scripted_agents:
        assert ag.action == 0
        assert np.all(ag.state.relays_for == 0)



def test_influence_graph_internal_world(env):
    """
    InfluenceGraph should expose a `.world` carrying the 'LT' model.
    """
    assert hasattr(env, "world")
    assert getattr(env.world, "model", None) == "LT"


def test_static_graph_not_moved(env):
    """
    When `dynamic_graph=False`, the edge set must remain unchanged after step().
    """
    world = _world(env)
    before = copy.deepcopy(world.graph.edges())
    _prep_broadcast(env, broadcasters=[0], uninformed=[1, 2])
    world.step()
    after = world.graph.edges()
    assert set(before) == set(after)
