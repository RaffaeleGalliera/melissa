import copy
import numpy as np
import pytest

from graph_env.env.influence_graph import InfluenceGraph


# ---------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------
@pytest.fixture
def env():
    """
    Fresh InfluenceGraph with 20 agents on a static topology.
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


# ---------------------------------------------------------------------
# helper to prepare broadcaster masks
# ---------------------------------------------------------------------
def _prep_broadcast(env, broadcasters, uninformed):
    world = _world(env)
    for a in world.agents:
        if a.id in broadcasters:
            a.state.has_message = True
            a.action = 1
        else:
            a.state.has_message = False
            a.action = 0

# ---------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------
def test_theta_in_unit_interval(env):
    theta = _world(env).theta
    assert theta.shape == (20,)
    assert np.all(theta > 0.0)
    assert np.all(theta < 1.0)


def test_weight_matrix_properties(env):
    W = _world(env).influence_weights
    n = W.shape[0]

    # zero diagonal
    assert np.allclose(np.diag(W), 0.0)

    # column sums: either 0 (isolated) or 1.0 (normalised)
    col_sums = W.sum(axis=0)
    assert np.allclose(col_sums[(col_sums > 0)], 1.0, atol=1e-6)


def test_every_node_has_strong_out_edge(env):
    """Property added by new sampler."""
    world = _world(env)
    for u in range(world.num_agents):
        # at least one neighbour satisfies weight >= theta_target
        assert any(
            world.influence_weights[u, v] >= world.theta[v]
            for v in range(world.num_agents)
            if u != v
        )


def test_directional_asymmetry(env):
    """If u→v can single-shot activate, v→u cannot."""
    world = _world(env)
    W, th = world.influence_weights, world.theta
    n = world.num_agents
    for u in range(n):
        for v in range(u + 1, n):
            cond_uv = W[u, v] >= th[v]
            cond_vu = W[v, u] >= th[u]
            assert not (cond_uv and cond_vu)


def test_reset_draws_fresh_parameters(env):
    env.reset()
    theta1, W1 = _world(env).theta.copy(), _world(env).influence_weights.copy()
    env.reset()
    theta2, W2 = _world(env).theta, _world(env).influence_weights
    assert not np.allclose(theta1, theta2)
    assert not np.allclose(W1, W2)


def test_same_seed_reproducible(env):
    env.seed(123)
    env.reset()
    theta1, W1 = _world(env).theta.copy(), _world(env).influence_weights.copy()
    env.seed(123)
    env.reset()
    theta2, W2 = _world(env).theta, _world(env).influence_weights
    assert np.allclose(theta1, theta2)
    assert np.allclose(W1, W2)


def test_single_broadcaster_triggers_activation(env):
    world = _world(env)
    v = np.where(world.agents[0].one_hop_neighbours_ids)[0][0]  # get a neighbour
    world.theta[v] = 0.3
    world.influence_weights[0, v] = 0.5
    _prep_broadcast(env, broadcasters=[0], uninformed=[v])
    world.step()
    assert world.agents[v].state.has_message is True


def test_below_threshold_no_activation(env):
    world = _world(env)
    v = np.where(world.agents[0].one_hop_neighbours_ids)[0][0]  # get a neighbour
    world.theta[v] = 0.8
    world.influence_weights[0, v] = 0.2
    _prep_broadcast(env, broadcasters=[0], uninformed=[v])
    world.step()
    assert world.agents[v].state.has_message is False


def test_combined_influence_multiple_broadcasters(env):
    world = _world(env)
    v = np.where(world.agents[0].one_hop_neighbours_ids)[0][0]  # get a neighbour
    # ge neighbor of v which is not 0
    v_neighbors = np.where(world.agents[v].one_hop_neighbours_ids)[0]
    w = v_neighbors[v_neighbors != 0][0]  # get a neighbour of v which is not 0
    world.theta[v] = 0.55
    world.influence_weights[0, v] = 0.30
    world.influence_weights[w, v] = 0.35
    _prep_broadcast(env, broadcasters=[0, w], uninformed=[v])
    world.step()
    assert world.agents[v].state.has_message is True


def test_scripted_agents_counters_cleared(env):
    world = _world(env)
    if not world.scripted_agents:
        pytest.skip("No scripted agents configured")

    for ag in world.scripted_agents:
        ag.state.has_message = True
        ag.action = 1
        ag.state.relays_for[:] = 1

    world.step()
    for ag in world.scripted_agents:
        assert ag.action == 0
        assert np.all(ag.state.relays_for == 0)


def test_influence_graph_internal_world(env):
    assert hasattr(env, "world")
    assert env.world.model == "LT"


def test_static_graph_not_moved(env):
    world = _world(env)
    before_edges = copy.deepcopy(world.graph.edges())
    _prep_broadcast(env, broadcasters=[0], uninformed=[1, 2])
    world.step()
    assert set(before_edges) == set(world.graph.edges())
