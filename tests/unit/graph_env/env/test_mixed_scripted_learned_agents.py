import networkx as nx
import pytest
from graph_env.env.graph import GraphEnv

@pytest.fixture
def simple_cycle_graph():
    """
    A small 4-node cycle so every node has exactly two neighbors.
    Positions chosen so radius=1.0 connects only true edges.
    """
    g = nx.cycle_graph(4)
    for i in g.nodes:
        g.nodes[i]['pos'] = (i, 0)
    return g

@pytest.fixture(scope="module")
def base_env_kwargs():
    """
    Minimal kwargs accepted by GraphEnv.__init__.
    """
    return dict(
        number_of_agents=20,
        radius=0.2,
        dynamic_graph=False,
        render_mode=None,
    )


def test_invalid_scripted_ratio(base_env_kwargs):
    """
    scripted_agents_ratio must lie in [0.0, 1.0].
    """
    with pytest.raises(ValueError):
        GraphEnv(**base_env_kwargs, scripted_agents_ratio=-0.1)
    with pytest.raises(ValueError):
        GraphEnv(**base_env_kwargs, scripted_agents_ratio=1.1)


def test_zero_ratio_all_policy(base_env_kwargs):
    """
    scripted_agents_ratio=0.0 and no heuristic ⇒ all agents are policy-driven,
    with no action_callback set on any Agent.
    """
    env = GraphEnv(**base_env_kwargs, scripted_agents_ratio=0.0, heuristic=None)
    env.reset(seed=42)

    assert len(env.world.scripted_agents) == 0
    # No Agent should have a callback
    for agent in env.world.agents:
        assert agent.action_callback is None


def test_sampling_reproducible_and_variable(base_env_kwargs):
    """
    With ratio>0, the set of scripted vs. policy actors should be:
     - reproducible for the same seed
     - different for different seeds
    """
    ratio = 0.3
    env = GraphEnv(**base_env_kwargs, scripted_agents_ratio=ratio, heuristic=None)

    env.reset(seed=123)
    scripted1 = set(env.world.scripted_indices)

    assert len(scripted1) == int(round(ratio * env.number_of_agents)), "Scripted count mismatch"

    env.reset(seed=123)
    scripted2 = set(env.world.scripted_indices)
    assert scripted1 == scripted2, "Same seed must yield same scripted set"

    env.reset(seed=999)
    scripted3 = set(env.world.scripted_indices)
    assert scripted1 != scripted3, "Different seed should change scripted set"

def test_partition_scripted_and_policy(simple_cycle_graph):
    """
    Verify that scripted_indices ∪ decision_makers = all nodes,
    scripted ∩ policy = ∅, and the origin is never scripted.
    """
    N = simple_cycle_graph.number_of_nodes()
    env = GraphEnv(
        graph=simple_cycle_graph,
        number_of_agents=N,
        radius=1.0,
        dynamic_graph=False,
        render_mode=None,
        scripted_agents_ratio=0.5,
        heuristic="simple_broadcast"
    )
    env.reset(seed=123)

    scripted = set(env.world.scripted_indices)
    policy = set([agent.id for agent in env.world.policy_agents])

    # complete cover
    assert scripted.union(policy) == set(range(N))
    # disjoint
    assert scripted.isdisjoint(policy)
    # source never scripted
    assert env.world.origin_agent not in scripted
