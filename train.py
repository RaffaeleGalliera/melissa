import argparse
import datetime
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

from graph_env.env.utils.networks import GATNetwork
from graph_env.env.utils.policies import MultiAgentSharedPolicy

os.environ["SDL_VIDEODRIVER"]="x11"


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eps-test", type=float, default=0.005)
    parser.add_argument("--eps-train", type=float, default=1.)
    parser.add_argument("--eps-train-final", type=float, default=0.05)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=500)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=100000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.)

    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--hidden-emb", type=int, default=128)

    parser.add_argument("--model-name", type=str, default="best.pth")
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
    return PettingZooEnv(env)


def get_agents(
    args: argparse.Namespace = get_args(),
    policy: BasePolicy = None,
    optim: torch.optim.Optimizer = None,
) -> Tuple[BasePolicy, torch.optim.Optimizer, List]:
    env = get_env()
    observation_space = env.observation_space['observation'] if isinstance(
        env.observation_space, (gym.spaces.Dict, gymnasium.spaces.Dict)
    ) else env.observation_space
    args.state_shape = observation_space.shape or observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = 1

    if policy is None:
        # model
        net = GATNetwork(
            NUMBER_OF_FEATURES,
            args.hidden_emb,
            args.action_shape,
            args.num_heads,
            device=args.device
        ).to(args.device)

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

    masp_policy = MultiAgentSharedPolicy(policy, env)

    return masp_policy, optim, env.agents


def train_agent(
    args: argparse.Namespace = get_args(),
    masp_policy: BasePolicy = None,
    optim: torch.optim.Optimizer = None,
) -> Tuple[dict, BasePolicy]:
    train_envs = SubprocVectorEnv([lambda: get_env(graph=load_testing_graph(f"testing_graph_{NUMBER_OF_AGENTS}.gpickle")) for _ in range(args.training_num)])
    test_envs = SubprocVectorEnv([lambda: get_env(graph=load_testing_graph(f"testing_graph_{NUMBER_OF_AGENTS}.gpickle"), render_mode='human') for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    masp_policy, optim, agents = get_agents(args, policy=masp_policy, optim=optim)

    # collector
    train_collector = Collector(
        masp_policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True
    )
    test_collector = Collector(masp_policy, test_envs)
    train_collector.collect(n_step=args.batch_size * args.training_num)
    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    log_path = os.path.join(args.logdir,'mpr', 'dqn')
    logger = WandbLogger(project='dancing_bees')
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger.load(writer)
    weights_path = os.path.join(args.logdir, "mpr", "dqn", "weights")

    def save_best_fn(pol):
        weights_name = os.path.join(
            f"{weights_path}", f"{now}_best.pth"
        )

        torch.save(
            pol.policy.state_dict(), weights_name
        )

    def stop_fn(mean_rewards):
        # test_reward:  37.02
        return mean_rewards > 37

    def train_fn(epoch, env_step):
        # nature DQN setting, linear decay in the first 1M steps
        if env_step <= 1e6:
            eps = args.eps_train - env_step / 1e6 * \
                (args.eps_train - args.eps_train_final)
        else:
            eps = args.eps_train_final
        masp_policy.policy.set_eps(eps)
        if env_step % 1000 == 0:
            logger.write("train/env_step", env_step, {"train/eps": eps})

    def test_fn(epoch, env_step):
        masp_policy.policy.set_eps(args.eps_test)

    def reward_metric(rews):
        return rews[:, 0]

    # trainer
    result = offpolicy_trainer(
        masp_policy,
        train_collector,
        test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=10,
        batch_size=args.batch_size,
        train_fn=train_fn,
        test_fn=test_fn,
        # stop_fn=stop_fn,
        update_per_step=args.update_per_step,
        test_in_train=True,
        save_best_fn=save_best_fn,
        logger=logger
        # resume_from_log=args.resume
    )

    torch.save(
        masp_policy.policy.state_dict(), os.path.join(f"{weights_path}", f"{now}_last.pth")
    )

    return result, masp_policy


def watch(
    args: argparse.Namespace = get_args(),
    masp_policy: BasePolicy = None,
) -> None:
    weights_path = os.path.join(args.logdir, "mpr", "dqn", "weights", f"{args.model_name}")

    env = DummyVectorEnv([lambda: get_env(graph=load_testing_graph(f"testing_graph_{NUMBER_OF_AGENTS}.gpickle"), render_mode='human')])

    if masp_policy is None:
        masp_policy = load_policy(weights_path, args, env)

    masp_policy.policy.eval()
    masp_policy.policy.set_eps(args.eps_test)

    collector = Collector(masp_policy, env, exploration_noise=True)
    result = collector.collect(n_episode=1)

    pprint.pprint(result)
    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews.mean()}, length: {lens.mean()}")


def load_policy(path, args, env):
    # load from existing checkpoint
    print(f"Loading agent under {path}")
    if os.path.exists(path):
        # model
        net = GATNetwork(
            NUMBER_OF_FEATURES,
            args.hidden_emb,
            2,
            args.num_heads,
            device=args.device
        ).to(args.device)

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

        masp_policy, _, _, = get_agents(args, policy, optim)
        masp_policy.policy.load_state_dict(torch.load(path))

        print("Successfully restore policy and optim.")
        return masp_policy
    else:
        print("Fail to restore policy and optim.")
        exit(0)
