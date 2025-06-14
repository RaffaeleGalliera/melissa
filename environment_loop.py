import time

import numpy as np
from graph_env.env.influence_graph import InfluenceGraph


"""-------------------------------------------------------------------------
Small standalone script to *visually* debug InfluenceGraph.
Every agent always chooses action==1 (broadcast).  We terminate early when
all nodes in the world hold the message or when max_cycles is reached.
-------------------------------------------------------------------------"""


def run_debug_loop(
    n_agents: int = 20,
    radius: float = 0.2,
    render: bool = True,
    step_sleep: float = 0.25,  # seconds between renders
):
    env = InfluenceGraph(
        number_of_agents=n_agents,
        radius=radius,
        render_mode="human" if render else None,
    )
    env.seed(42)
    env.reset()
    done = False

    while not done:
        if not env.agents:  # no active agents left
            break
        current_agent = env.agent_selection
        obs = env.observe(current_agent)
        action_mask = obs["action_mask"]

        # always broadcast if allowed, else 0
        action = 1 if action_mask[1] == 1 else 0
        env.step(action)

        if render:
            env.render()
            time.sleep(step_sleep)

        # termination condition: everyone has the message
        world = env.world
        if all(a.state.has_message for a in world.agents):
            print(f"All {n_agents} agents activated in {env.num_moves} world steps!")
            done = True

    env.close()


if __name__ == "__main__":
    run_debug_loop()
