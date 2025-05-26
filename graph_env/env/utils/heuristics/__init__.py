from .mpr import mpr_heuristic
from .core import simple_broadcast, probabilistic_gossip, probabilistic_relay, broadcast_if_any_interested, HeuristicResult

HEURISTIC_REGISTRY = {
    "mpr": mpr_heuristic,
    "probabilistic_gossip": probabilistic_gossip,
    "probabilistic_relay": probabilistic_relay,
    "simple_broadcast": simple_broadcast,
    "broadcast_if_any_interested": broadcast_if_any_interested
}
