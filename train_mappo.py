import argparse
import datetime
import os
import pprint
from typing import List, Tuple

import gym
import gymnasium
import numpy as np
import torch
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.discrete import Actor, Critic
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import VectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, PPOPolicy
from tianshou.trainer import onpolicy_trainer

from graph_env import graph_env_v0
from graph_env.env.utils.constants import NUMBER_OF_AGENTS, RADIUS_OF_INFLUENCE, NUMBER_OF_FEATURES
from graph_env.env.utils.logger import CustomLogger

from graph_env.env.utils.networks.actor_critic_gat import GATNetwork
from graph_env.env.utils.policies.multi_agent_managers.shared_policy import MultiAgentSharedPolicy

from graph_env.env.utils.collectors.collector import MultiAgentCollector

import time
import warnings

os.environ["SDL_VIDEODRIVER"]="x11"
warnings.filterwarnings("ignore")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--eps-test", type=float, default=0.001)
    parser.add_argument("--eps-train", type=float, default=1.)
    parser.add_argument("--eps-train-final", type=float, default=0.05)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-step", type=int, default=4)
    parser.add_argument("--hidden-emb", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--target-update-freq", type=int, default=500)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=100000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--training-num", type=int, default=20)
    parser.add_argument("--test-num", type=int, default=100)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.)
    parser.add_argument('--dueling-q-hidden-sizes', type=int, nargs='*', default=[128, 128])
    parser.add_argument('--dueling-v-hidden-sizes', type=int, nargs='*', default=[128, 128])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument("--rew-norm", type=int, default=True)
    # In theory, `vf-coef` will not make any difference if using Adam optimizer.
    parser.add_argument("--repeat-per-collect", type=int, default=4)
    parser.add_argument("--vf-coef", type=float, default=0.25)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--lr-decay", type=int, default=True)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--eps-clip", type=float, default=0.1)
    parser.add_argument("--dual-clip", type=float, default=None)
    parser.add_argument("--value-clip", type=int, default=1)
    parser.add_argument("--norm-adv", type=int, default=1)
    parser.add_argument("--recompute-adv", type=int, default=0)
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

    parser.add_argument(
        "--dynamic-graph",
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
            is_testing=False,
            dynamic_graph=False):
    env = graph_env_v0.env(graph=graph,
                           render_mode=render_mode,
                           number_of_agents=number_of_agents,
                           radius=radius,
                           is_testing=is_testing,
                           dynamic_graph=dynamic_graph)
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
        net_a = GATNetwork(
            NUMBER_OF_FEATURES,
            args.hidden_emb,
            args.action_shape,
            args.num_heads,
            device=args.device
        )

        net_c = GATNetwork(
            NUMBER_OF_FEATURES,
            args.hidden_emb,
            args.action_shape,
            args.num_heads,
            is_critic=True,
            device=args.device
        )

        actor = Actor(net_a, args.action_shape, device=args.device,
                      softmax_output=False)
        critic = Critic(net_c, device=args.device)

        optim = torch.optim.Adam(
            ActorCritic(actor, critic).parameters(), lr=args.lr, eps=1e-5
        )

        lr_scheduler = None
        if args.lr_decay:
            # decay learning rate to 0 linearly
            max_update_num = np.ceil(
                args.step_per_epoch / args.step_per_collect
            ) * args.epoch

            lr_scheduler = LambdaLR(
                optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num
            )

        # define policy
        def dist(p):
            return torch.distributions.Categorical(logits=p)

        policy = PPOPolicy(
            actor,
            critic,
            optim,
            dist,
            discount_factor=args.gamma,
            gae_lambda=args.gae_lambda,
            max_grad_norm=args.max_grad_norm,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            reward_normalization=args.rew_norm,
            action_scaling=False,
            lr_scheduler=lr_scheduler,
            action_space=env.action_space,
            eps_clip=args.eps_clip,
            value_clip=args.value_clip,
            dual_clip=args.dual_clip,
            advantage_normalization=args.norm_adv,
            recompute_advantage=args.recompute_adv,
        ).to(args.device)

    masp_policy = MultiAgentSharedPolicy(policy, env)

    return masp_policy, optim, env.agents


def watch(
    args: argparse.Namespace = get_args(),
    masp_policy: BasePolicy = None,
) -> None:
    weights_path = os.path.join(args.logdir, "mpr", 'ppo', "weights", f"{args.model_name}")

    env = DummyVectorEnv([lambda: get_env(is_testing=True)])

    if masp_policy is None:
        masp_policy = load_policy(weights_path, args, env)

    masp_policy.policy.eval()

    collector = MultiAgentCollector(masp_policy,
                                    env,
                                    exploration_noise=False,
                                    number_of_agents=NUMBER_OF_AGENTS)
    result = collector.collect(n_episode=args.test_num)

    pprint.pprint(result)
    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews.mean()}, length: {lens.mean()}")

    time.sleep(100)


def train_agent(
    args: argparse.Namespace = get_args(),
    masp_policy: BasePolicy = None,
    optim: torch.optim.Optimizer = None,
) -> Tuple[dict, BasePolicy]:

    train_envs = SubprocVectorEnv([lambda: get_env() for i in range(args.training_num)])
    test_envs = SubprocVectorEnv([lambda: get_env(is_testing=True)])

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
        exploration_noise=False,
        number_of_agents=len(agents)
    )

    # train_collector.collect(n_step=args.batch_size * args.training_num)
    # log
    log_path = os.path.join(args.logdir, 'mpr', 'ppo')
    logger = CustomLogger(project='dancing_bees', name=args.model_name)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger.load(writer)
    weights_path = os.path.join(args.logdir, "mpr", 'ppo', "weights")

    result = onpolicy_trainer(
        masp_policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.repeat_per_collect,
        args.test_num,
        args.batch_size,
        step_per_collect=args.step_per_collect,
        logger=logger,
        test_in_train=False,
    )
    # trainer

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
        net_a = GATNetwork(
            NUMBER_OF_FEATURES,
            args.hidden_emb,
            args.action_shape,
            args.num_heads,
            device=args.device
        )
        net_c = GATNetwork(
            NUMBER_OF_FEATURES,
            args.hidden_emb,
            args.action_shape,
            args.num_heads,
            is_critic=True,
            device=args.device
        )
        actor = Actor(net_a, args.action_shape, device=args.device,
                      softmax_output=False)
        critic = Critic(net_c, device=args.device)

        optim = torch.optim.Adam(
            ActorCritic(actor, critic).parameters(), lr=args.lr, eps=1e-5
        )

        lr_scheduler = None
        if args.lr_decay:
            # decay learning rate to 0 linearly
            max_update_num = np.ceil(
                args.step_per_epoch / args.step_per_collect
            ) * args.epoch

            lr_scheduler = LambdaLR(
                optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num
            )

        # define policy
        def dist(p):
            return torch.distributions.Categorical(logits=p)

        policy = PPOPolicy(
            actor,
            critic,
            optim,
            dist,
            discount_factor=args.gamma,
            gae_lambda=args.gae_lambda,
            max_grad_norm=args.max_grad_norm,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            reward_normalization=args.rew_norm,
            action_scaling=False,
            lr_scheduler=lr_scheduler,
            action_space=env.action_space,
            eps_clip=args.eps_clip,
            value_clip=args.value_clip,
            dual_clip=args.dual_clip,
            advantage_normalization=args.norm_adv,
            recompute_advantage=args.recompute_adv,
        ).to(args.device)

        masp_policy, _, _, = get_agents(args, policy, optim)
        masp_policy.policy.load_state_dict(torch.load(path))

        print("Successfully restore policy and optim.")
        return masp_policy
    else:
        print("Fail to restore policy and optim.")
        exit(0)
