from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from graph_env.graph_env_v0 import GraphEnv


def snapshot_edges(env: GraphEnv) -> List[Tuple[int, int]]:
    G = env.world.graph
    bidir = []
    for u, v in G.edges():
        bidir.extend([(int(u), int(v)), (int(v), int(u))])
    return bidir


def run_and_record(
    n_agents: int,
    radius: float,
    num_test_episodes: int = 1,
    outfile: Path = Path("spacetime.json"),
) -> Dict:
    env = GraphEnv(
        number_of_agents=n_agents,
        radius=radius,
        dynamic_graph=True,
        is_testing=True,
        num_test_episodes=num_test_episodes,
        all_agents_source=True,
    )
    env.seed(43)
    world = env.world

    for _ in range(num_test_episodes):
        source_id: int = int(world.origin_agent)
        interested = {int(a.id): int(a.is_interested) for a in world.agents}
        vertices = sorted(interested.keys())
        spacetime_edges: List[List[Tuple[int, int]]] = []
        prev_round = -1

        while env.agents:
            env.step(action=1)

            if env.num_moves != prev_round:
                spacetime_edges.append(snapshot_edges(env))
                prev_round = env.num_moves


        payload = {
            "num_agents": n_agents,
            "horizon": len(spacetime_edges),
            "vertices": vertices,
            "source_id": source_id,
            "interested": interested,        # {node_id: 0/1}
            "edges_per_t": spacetime_edges,  # list of edges for each time slot
        }

        output = outfile.with_suffix(f".{n_agents}agents.n{world.test_episode_index}.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as fp:
            json.dump(payload, fp, indent=2)
        print(
            f"✓ Saved space–time graph for {payload['horizon']} steps "
            f"({payload['num_agents']} agents) → {output}"
        )

        env.reset()


def main() -> None:
    parser = argparse.ArgumentParser("Collect space–time graph data")
    parser.add_argument("--agents", type=int, default=50)
    parser.add_argument("--radius", type=float, default=0.2)
    parser.add_argument("--outfile", type=Path, default=Path("space_time_graphs/spacetime"))
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.25, help="seconds between renders")
    parser.add_argument("--num-test-episodes", type=int, default=1, help="Number of test episodes to run (default: 1)")
    args = parser.parse_args()

    run_and_record(
        n_agents=args.agents,
        radius=args.radius,
        outfile=args.outfile,
        num_test_episodes=args.num_test_episodes
    )


if __name__ == "__main__":
    main()
