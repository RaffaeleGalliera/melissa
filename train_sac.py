import argparse
import datetime
import os
import pprint
from typing import List, Tuple

import gym
import gymnasium
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import VectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy
from tianshou.utils import WandbLogger
from tianshou.utils.net.continuous import Critic

from tianshou.policy import DiscreteSACPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.discrete import Actor, Critic


from graph_env import graph_env_v0
from graph_env.env.utils.constants import NUMBER_OF_AGENTS, RADIUS_OF_INFLUENCE, NUMBER_OF_FEATURES
from graph_env.env.utils.core import load_graph
from graph_env.env.utils.logger import CustomLogger

from graph_env.env.networks import GATNetwork
from graph_env.env.policies import MultiAgentSharedPolicy

from graph_env.env.collector import MultiAgentCollector
os.environ["SDL_VIDEODRIVER"]="x11"

import time

import warnings
warnings.filterwarnings("ignore")

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=4213)
    parser.add_argument("--scale-obs", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--actor-lr", type=float, default=1e-5)
    parser.add_argument("--critic-lr", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--auto-alpha", action="store_true", default=True)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--step-per-epoch", type=int, default=100000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=1)
    parser.add_argument("--rew-norm", type=int, default=False)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.)
    parser.add_argument("--hidden-emb", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only"
    )
    parser.add_argument(
        "--wandb",
        default=False,
        action="store_true",
        help="Set WANDB logger"
    )
    parser.add_argument("--save-buffer-name", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=datetime.datetime.now().strftime("%y%m%d-%H%M%S"))

    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]


def get_env(number_of_agents=NUMBER_OF_AGENTS,
            radius=RADIUS_OF_INFLUENCE,
            graph=None,
            render_mode=None,
            is_testing=False):
    env = graph_env_v0.env(graph=graph,
                           render_mode=render_mode,
                           number_of_agents=number_of_agents,
                           radius=radius,
                           is_testing=is_testing)
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
        # features
        net = GATNetwork(
            NUMBER_OF_FEATURES,
            args.hidden_emb,
            args.action_shape,
            args.num_heads,
            features_only=True,
            device=args.device
        )

        actor = Actor(net, args.action_shape, device=args.device,
                      softmax_output=False)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
        critic1 = Critic(net, last_size=args.action_shape, device=args.device)
        critic1_optim = torch.optim.Adam(critic1.parameters(),
                                         lr=args.critic_lr)
        critic2 = Critic(net, last_size=args.action_shape, device=args.device)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

        if args.auto_alpha:
            target_entropy = 0.98 * np.log(np.prod(args.action_shape))
            log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
            alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
            args.alpha = (target_entropy, log_alpha, alpha_optim)

        policy = DiscreteSACPolicy(
            actor,
            actor_optim,
            critic1,
            critic1_optim,
            critic2,
            critic2_optim,
            args.tau,
            args.gamma,
            args.alpha,
            estimation_step=args.n_step,
            reward_normalization=args.rew_norm,
        ).to(args.device)

    masp_policy = MultiAgentSharedPolicy(policy, env)

    return masp_policy, optim, env.agents


def watch(
    args: argparse.Namespace = get_args(),
    masp_policy: BasePolicy = None,
) -> None:
    weights_path = os.path.join(args.logdir, "mpr", "dqn", "weights", f"{args.model_name}")

    env = DummyVectorEnv([lambda: get_env(render_mode='human', is_testing=True)])
    if masp_policy is None:
        masp_policy = load_policy(weights_path, args, env)

    masp_policy.policy.eval()

    collector = MultiAgentCollector(masp_policy, env, exploration_noise=True, number_of_agents=20)
    result = collector.collect(n_episode=1)

    pprint.pprint(result)
    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews.mean()}, length: {lens.mean()}")

    time.sleep(100)


def train_agent(
    args: argparse.Namespace = get_args(),
    masp_policy: BasePolicy = None,
    optim: torch.optim.Optimizer = None,
) -> Tuple[dict, BasePolicy]:
    train_envs = SubprocVectorEnv([lambda: get_env(render_mode=None) for _ in range(args.training_num)])
    test_envs = SubprocVectorEnv([lambda:
                                  get_env(is_testing=True) for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    masp_policy, optim, agents = get_agents(args, policy=masp_policy, optim=optim)

    # collector
    train_collector = MultiAgentCollector(
        masp_policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size,
                           len(train_envs)*len(agents),
                           ignore_obs_next=True),
        exploration_noise=True,
        number_of_agents=len(agents)
    )
    test_collector = MultiAgentCollector(
        masp_policy,
        test_envs,
        VectorReplayBuffer(args.buffer_size,
                           len(test_envs)*len(agents),
                           ignore_obs_next=True),
        exploration_noise=True,
        number_of_agents=len(agents)
    )

    # train_collector.collect(n_step=args.batch_size * args.training_num)
    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    log_path = os.path.join(args.logdir, 'mpr', 'dqn')
    logger = CustomLogger(project='dancing_bees', name=args.model_name)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger.load(writer)
    weights_path = os.path.join(args.logdir, "mpr", "dqn", "weights")

    def save_best_fn(pol):
        weights_name = os.path.join(
            f"{weights_path}", f"{args.model_name}_best.pth"
        )

        print(f"Saving {args.model_name} best")
        torch.save(
            pol.policy.state_dict(), weights_name
        )

    # trainer
    result = offpolicy_trainer(
        masp_policy,
        train_collector,
        test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=args.test_num,
        batch_size=args.batch_size,
        update_per_step=args.update_per_step,
        test_in_train=False,
        save_best_fn=save_best_fn,
        logger=logger
        # resume_from_log=args.resume
    )
    print(f"Saving {args.model_name} last")
    torch.save(
        masp_policy.policy.state_dict(), os.path.join(f"{weights_path}", f"{args.model_name}_last.pth")
    )

    return result, masp_policy


def load_policy(path, args, env):
    # load from existing checkpoint
    args.action_shape = 2

    print(f"Loading agent under {path}")
    if os.path.exists(path):
        # features
        # features
        net = GATNetwork(
            NUMBER_OF_FEATURES,
            args.hidden_emb,
            args.action_shape,
            args.num_heads,
            features_only=True,
            device=args.device
        )

        actor = Actor(net, args.action_shape, device=args.device,
                      softmax_output=False)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
        critic1 = Critic(net, last_size=args.action_shape, device=args.device)
        critic1_optim = torch.optim.Adam(critic1.parameters(),
                                         lr=args.critic_lr)
        critic2 = Critic(net, last_size=args.action_shape, device=args.device)
        critic2_optim = torch.optim.Adam(critic2.parameters(),
                                         lr=args.critic_lr)

        # better not to use auto alpha in CartPole
        if args.auto_alpha:
            target_entropy = 0.98 * np.log(np.prod(args.action_shape))
            log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
            alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
            args.alpha = (target_entropy, log_alpha, alpha_optim)

        policy = DiscreteSACPolicy(
            actor,
            actor_optim,
            critic1,
            critic1_optim,
            critic2,
            critic2_optim,
            args.tau,
            args.gamma,
            args.alpha,
            estimation_step=args.n_step,
            reward_normalization=args.rew_norm,
        ).to(args.device)

        masp_policy, _, _, = get_agents(args, policy)
        masp_policy.policy.load_state_dict(torch.load(path))

        print("Successfully restore policy and optim.")
        return masp_policy
    else:
        print("Fail to restore policy and optim.")
        exit(0)
