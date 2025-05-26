from typing import NamedTuple, Optional
import numpy as np

from graph_env.env.utils.core import Agent


class HeuristicResult(NamedTuple):
    relay_mask: Optional[np.ndarray]
    action:    Optional[int]

def simple_broadcast(agent: Agent)-> HeuristicResult:
    """
    Always broadcast to all one-hop neighbors.
    """
    action = 0 if agent.state.has_taken_action else 1
    return HeuristicResult(relay_mask=None, action=action)

def probabilistic_gossip(
        agent: Agent,
        prob: float
) -> HeuristicResult:
    """
    Send only if the probability is met.
    """
    action = 0 if agent.state.has_taken_action else np.random.binomial(1, prob)
    return HeuristicResult(relay_mask=None, action=action)

def probabilistic_relay(
        agent: Agent,
        prob: float
) -> HeuristicResult:
    """
    Set your MPR set probabilistically.
    """
    one_hop_neighbours_ids = agent.one_hop_neighbours_ids

    relay_mask = np.random.binomial(1, prob, size=one_hop_neighbours_ids.shape)
    relay_mask[~one_hop_neighbours_ids.astype(bool)] = 0
    return HeuristicResult(relay_mask=relay_mask, action=None)

def broadcast_if_any_interested(
        agent: Agent,
) -> HeuristicResult:
    """
    Relay the message if there is at least one interested neighbor.
    """
    action = 1 if agent.number_interested_neighbors > 0 else 0

    return HeuristicResult(relay_mask=None, action=action)
