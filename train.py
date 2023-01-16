import argparse
import os
import pprint
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import gymnasium
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import ShmemVectorEnv, DummyVectorEnv, SubprocVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, MultiAgentPolicyManager, DQNPolicy
from tianshou.trainer import onpolicy_trainer, offpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils.net.common import Net

from graph_env import graph_env_v0
from graph_env.env.utils.constants import NUMBER_OF_AGENTS, RADIUS_OF_INFLUENCE, NUMBER_OF_FEATURES
from graph_env.env.utils.core import load_testing_graph

from graph_env.network_policies.networks import GATNetwork

os.environ["SDL_VIDEODRIVER"]="x11"

DEVICE = 'cpu'

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=9)
    parser.add_argument('--eps-test', type=float, default=0.01)
    parser.add_argument('--eps-train', type=float, default=0.73)
    parser.add_argument('--buffer-size', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=0.013)
    parser.add_argument(
        '--gamma', type=float, default=0.99, help='a smaller gamma favors earlier win'
    )
    parser.add_argument('--n-step', type=int, default=10)
    parser.add_argument('--target-update-freq', type=int, default=4)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--step-per-collect', type=int, default=30)
    # parser.add_argument('--episode-per-collect', type=int, default=16)
    parser.add_argument('--repeat-per-collect', type=int, default=10)
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[128, 128, 128, 128])
    parser.add_argument('--training-num', type=int, default=8)
    parser.add_argument('--test-num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')

    parser.add_argument(
        '--watch',
        default=False,
        action='store_true',
        help='no training, '
        'watch the play of pre-trained models'
    )
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )

    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]


def get_env(number_of_agents=NUMBER_OF_AGENTS, radius=RADIUS_OF_INFLUENCE, graph=None, render_mode=None):
    env = graph_env_v0.env(graph=graph,
                           render_mode=render_mode,
                           number_of_agents=number_of_agents,
                           radius=radius)
    # env = ss.pad_observations_v0(env)
    # env = ss.pad_action_space_v0(env)
    return PettingZooEnv(env)


def get_agents(
    args: argparse.Namespace = get_args(),
    agents: Optional[List[BasePolicy]] = None,
    optims: Optional[List[torch.optim.Optimizer]] = None,
) -> Tuple[BasePolicy, List[torch.optim.Optimizer], List]:
    env = get_env()
    observation_space = env.observation_space['observation'] if isinstance(
        env.observation_space, (gym.spaces.Dict, gymnasium.spaces.Dict)
    ) else env.observation_space
    args.state_shape = observation_space.shape or observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = 1

    if agents is None:
        agents = []
        optims = []

        # model
        net = GATNetwork(
            NUMBER_OF_FEATURES,
            128,
            args.action_shape,
            5,
            device=DEVICE
        ).to(DEVICE)

        optim = torch.optim.Adam(
            net.parameters(), lr=args.lr
        )

        dist = torch.distributions.Categorical

        agent = DQNPolicy(
            net,
            optim,
            args.gamma,
            args.n_step,
            target_update_freq=args.target_update_freq
        )

        for _ in range(NUMBER_OF_AGENTS):
            agents.append(agent)

    policy = MultiAgentPolicyManager(agents, env)

    return policy, optim, env.agents


def train_agent(
    args: argparse.Namespace = get_args(),
    agents: Optional[List[BasePolicy]] = None,
    optims: Optional[List[torch.optim.Optimizer]] = None,
) -> Tuple[dict, BasePolicy]:
    train_envs = SubprocVectorEnv([lambda: get_env(graph=load_testing_graph(f"testing_graph_{NUMBER_OF_AGENTS}.gpickle")) for _ in range(args.training_num)])
    test_envs = SubprocVectorEnv([lambda: get_env(graph=load_testing_graph(f"testing_graph_{NUMBER_OF_AGENTS}.gpickle"), render_mode='human') for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    policy, optim, agents = get_agents(args, agents=agents, optims=optims)

    # collector
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs))
        # exploration_noise=True
    )
    test_collector = Collector(policy, test_envs)
    # train_collector.collect(n_step=args.batch_size * args.training_num)
    # log
    log_path = os.path.join(args.logdir, 'mpr', 'dqn')
    logger = WandbLogger(project='dancing_bees')
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger.load(writer)

    def save_best_fn(policy):
        model_save_path = os.path.join(
            args.logdir, "mpr", "dqn", "weights", f"policy.pth"
        )
        torch.save(
            policy.policies[agents[0]].state_dict(), model_save_path
        )

    def stop_fn(mean_rewards):
        # best -19.86
        return mean_rewards >= -19

    def train_fn(epoch, env_step):
        eps = max(args.eps_train * (1 - 5e-6) ** env_step, args.eps_test)
        [agent.set_eps(eps) for agent in policy.policies.values()]

    def test_fn(epoch, env_step):
        [agent.set_eps(args.eps_test) for agent in policy.policies.values()]

    def reward_metric(rews):
        return rews[:, 0]

    # trainer
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        max_epoch=10,
        step_per_epoch=80000,
        step_per_collect=16,
        episode_per_test=1,
        batch_size=128,
        train_fn=train_fn,
        test_fn=test_fn,
        # stop_fn=stop_fn,
        update_per_step=0.0625,
        test_in_train=False,
        save_best_fn=save_best_fn,
        logger=logger
        # resume_from_log=args.resume
    )

    return result, policy


def watch(
    args: argparse.Namespace = get_args(),
    policy: Optional[BasePolicy] = None
) -> None:
    env = DummyVectorEnv([lambda: get_env(graph=load_testing_graph(f"testing_graph_{NUMBER_OF_AGENTS}.gpickle"), render_mode='human')])

    policy = load_policy("log/mpr/dqn/weights/policy.pth", args)
    policy.eval()
    policy.set_eps(0.05)

    collector = Collector(policy, env)
    result = collector.collect(n_episode=1)
    pprint.pprint(result)
    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews[:, 0].mean()}, length: {lens.mean()}")


def load_policy(path, args):
    # load from existing checkpoint
    print(f"Loading agent under {path}")
    if os.path.exists(path):
        # model
        net = GATNetwork(
            NUMBER_OF_FEATURES,
            128,
            2,
            5,
            device=DEVICE
        ).to(DEVICE)

        optim = torch.optim.Adam(
            net.parameters(), lr=args.lr
        )

        policy = DQNPolicy(
            net,
            optim,
            args.gamma,
            args.n_step,
            target_update_freq=args.target_update_freq
        )

        policy.load_state_dict(torch.load(path))
        print("Successfully restore policy and optim.")
        return policy
    else:
        print("Fail to restore policy and optim.")
        exit(0)
