import argparse
import datetime
import os
import warnings
from typing import Callable
import torch
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from tianshou.env.pettingzoo_env import PettingZooEnv
from graph_env import graph_env_v0
from graph_env.env.utils.constants import RADIUS_OF_INFLUENCE

os.environ["SDL_VIDEODRIVER"] = "x11"
warnings.filterwarnings("ignore")

def get_parser() -> argparse.ArgumentParser:
    """
    Build and return the argument parser with all hyperparameters.
    """
    parser = argparse.ArgumentParser(description="Refactored HL-DGN Training Script")
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--eps-test", type=float, default=0.001)
    parser.add_argument("--eps-train", type=float, default=1.0)
    parser.add_argument("--exploration-fraction", type=float, default=0.6)
    parser.add_argument("--eps-train-final", type=float, default=0.05)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-step", type=int, default=4)
    parser.add_argument("--hidden-emb", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--target-update-freq", type=int, default=500)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--step-per-epoch", type=int, default=100000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--training-num", type=int, default=40)
    parser.add_argument("--test-num", type=int, default=100)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument('--dueling-q-hidden-sizes', type=int, nargs='*', default=[128, 128])
    parser.add_argument('--dueling-v-hidden-sizes', type=int, nargs='*', default=[128, 128])
    parser.add_argument("--aggregator-function", type=str, default="max")
    parser.add_argument("--edge-attributes", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument("--mpr-policy", action="store_true", default=False, help="Use MPR policy")
    parser.add_argument("--n-agents", type=int, choices=[20, 50, 100], default=20)
    parser.add_argument("--watch", action="store_true", default=False, help="Watch the pre-trained policy only")
    parser.add_argument("--dynamic-graph", action="store_true", default=True, help="Enable dynamic graphs")
    parser.add_argument("--prio-buffer", action="store_true", default=False, help="Use prioritized replay buffer")
    parser.add_argument("--save-buffer-name", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=datetime.datetime.now().strftime("%y%m%d-%H%M%S"))
    parser.add_argument("--optimize", "--optimize-hyperparameters", action="store_true", default=False, help="Run hyperparameters search")
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--sampler-method", type=str, default="tpe")
    parser.add_argument("--pruner-method", type=str, default="median")
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--n-startup-trials", type=int, default=2)
    parser.add_argument("--n-warmup-steps", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--save-study", action="store_true", default=False, help="Save study")
    parser.add_argument("--heuristic", type=str, default=None, help="Heuristic function to use")
    parser.add_argument("--heuristic-params", nargs = "*", default = [], help = "List of key=value pairs for heuristic, e.g. prob=0.3 alpha=2")
    parser.add_argument("--scripted-agents-ratio", type=float, default=0.0, help="Ratio of scripted agents in the environment")
    return parser


def get_args() -> argparse.Namespace:
    """
    Parse the known arguments and set the learning algorithm name.
    """
    args = get_parser().parse_known_args()[0]


    params = {}
    for kv in args.heuristic_params:
        if "=" not in kv:
            raise ValueError(f"Bad heuristic-param '{kv}', use key=value")
        k, v = kv.split("=", 1)

        if v.isdigit():
            v2 = int(v)
        else:
            try:
                v2 = float(v)
            except ValueError:
                if v.lower() in ("true", "false"):
                    v2 = v.lower() == "true"
                else:
                    v2 = v
        params[k] = v2
    args.heuristic_params = params
    return args

def get_env(
    number_of_agents: int = 20,
    radius: float = RADIUS_OF_INFLUENCE,
    graph=None,
    render_mode=None,
    heuristic=None,
    heuristic_params=None,
    is_testing=False,
    dynamic_graph=False,
    all_agents_source=False,
    num_test_episodes=None,
    scripted_agents_ratio=None
) -> PettingZooEnv:
    """
    Create and wrap the GraphEnv in a PettingZooEnv interface.
    """
    env = graph_env_v0.env(
        graph=graph,
        render_mode=render_mode,
        number_of_agents=number_of_agents,
        radius=radius,
        heuristic=heuristic,
        heuristic_params=heuristic_params,
        is_testing=is_testing,
        dynamic_graph=dynamic_graph,
        all_agents_source=all_agents_source,
        num_test_episodes=num_test_episodes,
        scripted_agents_ratio=scripted_agents_ratio
    )
    return PettingZooEnv(env)
