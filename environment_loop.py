import time
from graph_env.graph_env_v0 import GraphEnv


def run_debug_loop(
    n_agents: int = 50,
    radius: float = 0.2,
    render: bool = True,
    step_sleep: float = 0.25,  # seconds between renders
):
    env = GraphEnv(
        number_of_agents=n_agents,
        radius=radius,
        render_mode="human" if render else None,
        is_testing=True
    )
    env.reset()
    done = False

    while not done:
        if not env.agents:  # no active agents left
            break
        current_agent = env.agent_selection
        obs = env.observe(current_agent)
        action = 1
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
