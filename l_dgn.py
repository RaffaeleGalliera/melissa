import argparse
import datetime
import os
import pprint
from pathlib import Path
from typing import List, Tuple
from math import pow, e, log

import gym
import gymnasium
import numpy as np
import optuna
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import VectorReplayBuffer, PrioritizedVectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.policy.modelfree.dqn import DQNPolicy
from torch_geometric.nn import global_max_pool, global_mean_pool, \
    global_add_pool

from graph_env import graph_env_v0
from graph_env.env.utils.constants import RADIUS_OF_INFLUENCE, \
    NUMBER_OF_FEATURES
from graph_env.env.utils.logger import CustomLogger

from graph_env.env.utils.networks.l_dgn import LDGNNetwork
from graph_env.env.utils.policies.multi_agent_managers.shared_policy import \
    MultiAgentSharedPolicy

from graph_env.env.utils.collectors.collector import MultiAgentCollector
from graph_env.env.utils.hyp_optimizer.offpolicy_opt import offpolicy_optimizer

import time
import warnings

os.environ["SDL_VIDEODRIVER"] = "x11"
warnings.filterwarnings("ignore")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--eps-test", type=float, default=0.001)
    parser.add_argument("--eps-train", type=float, default=1.)
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
    parser.add_argument("--training-num", type=int, default=20)
    parser.add_argument("--test-num", type=int, default=100)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.)
    parser.add_argument('--dueling-q-hidden-sizes', type=int, nargs='*',
                        default=[128, 128])
    parser.add_argument('--dueling-v-hidden-sizes', type=int, nargs='*',
                        default=[128, 128])
    parser.add_argument("--aggregator-function", type=str,
                        default="global_max_pool")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument("--mpr-policy",
                        action="store_true",
                        default=False,
                        help="Use MPR policy")
    parser.add_argument('--n-agents', type=int, choices=[20, 50, 100], default=20)
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
        help="Enable dynamic graphs"
    )

    parser.add_argument(
        "--prio-buffer",
        default=False,
        action="store_true",
        help="Enable prioritized experience replay"
    )

    parser.add_argument("--save-buffer-name", type=str, default=None)
    parser.add_argument("--model-name", type=str,
                        default=datetime.datetime.now().strftime(
                            "%y%m%d-%H%M%S"))

    parser.add_argument(
        "--optimize", "--optimize-hyperparameters", action="store_true",
        default=False,
        help="Run hyperparameters search"
    )

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

    parser.add_argument(
        "--save-study",
        default=False,
        action="store_true",
        help="Save study"
    )

    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser().parse_known_args()[0]
    parser.learning_algorithm = "l_dgn"
    return parser


def get_env(
        number_of_agents=20,
        radius=RADIUS_OF_INFLUENCE,
        graph=None,
        render_mode=None,
        is_scripted=False,
        is_testing=False,
        dynamic_graph=False,
        all_agents_source=False
):
    env = graph_env_v0.env(
        graph=graph,
        render_mode=render_mode,
        number_of_agents=number_of_agents,
        radius=radius,
        is_scripted=is_scripted,
        is_testing=is_testing,
        dynamic_graph=dynamic_graph,
        all_agents_source=all_agents_source
    )
    return PettingZooEnv(env)

def get_agents(
        args: argparse.Namespace = get_args(),
        policy: BasePolicy = None,
        optim: torch.optim.Optimizer = None,
) -> Tuple[BasePolicy, torch.optim.Optimizer, List]:
    env = get_env(number_of_agents=args.n_agents)
    observation_space = env.observation_space['observation'] if isinstance(
        env.observation_space, (gym.spaces.Dict, gymnasium.spaces.Dict)
    ) else env.observation_space
    args.state_shape = observation_space.shape or observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = 1

    if policy is None:
        # features
        q_param = {"hidden_sizes": args.dueling_q_hidden_sizes}
        v_param = {"hidden_sizes": args.dueling_v_hidden_sizes}

        aggregator = None
        if args.aggregator_function == "global_max_pool":
            aggregator = global_max_pool
        elif args.aggregator_function == "global_mean_pool":
            aggregator = global_mean_pool
        elif args.aggregator_function == "global_add_pool":
            aggregator = global_add_pool

        net = LDGNNetwork(
            NUMBER_OF_FEATURES,
            args.hidden_emb,
            args.action_shape,
            args.num_heads,
            device=args.device,
            dueling_param=(q_param, v_param),
            aggregator_function=aggregator
        )

        optim = torch.optim.Adam(
            net.parameters(), lr=args.lr
        )

        policy = DQNPolicy(
            net,
            optim,
            args.gamma,
            args.n_step,
            target_update_freq=args.target_update_freq
        ).to(args.device)

    masp_policy = MultiAgentSharedPolicy(policy, env)

    return masp_policy, optim, env.agents


def watch(
        args: argparse.Namespace = get_args(),
        masp_policy: BasePolicy = None,
) -> None:
    weights_path = os.path.join(args.logdir, "mpr", "l_dgn", "weights", f"{args.model_name}")

    env = DummyVectorEnv(
        [
            lambda: get_env(
                number_of_agents=args.n_agents,
                is_scripted=args.mpr_policy,
                is_testing=True,
                dynamic_graph=args.dynamic_graph,
                render_mode="human",
                all_agents_source=True
            )
        ]
    )

    env.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if masp_policy is None:
        masp_policy = load_policy(weights_path, args, env)

    masp_policy.policy.eval()
    masp_policy.policy.set_eps(args.eps_test)

    collector = MultiAgentCollector(
        masp_policy,
        env,
        exploration_noise=False,
        number_of_agents=args.n_agents
    )

    result = collector.collect(n_episode=args.test_num * args.n_agents)

    pprint.pprint(result)
    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews.mean()}, length: {lens.mean()}")

    time.sleep(100)


def train_agent(
        args: argparse.Namespace = get_args(),
        masp_policy: BasePolicy = None,
        optim: torch.optim.Optimizer = None,
        opt_trial: optuna.Trial = None
) -> Tuple[dict, BasePolicy]:
    train_envs = SubprocVectorEnv(
        [
            lambda: get_env(
                number_of_agents=args.n_agents,
                dynamic_graph=args.dynamic_graph
            ) for i in range(args.training_num)
        ]
    )

    test_envs = SubprocVectorEnv(
        [
            lambda: get_env(
                number_of_agents=args.n_agents,
                dynamic_graph=args.dynamic_graph,
                is_testing=True
            )
        ]
    )

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    masp_policy, optim, agents = get_agents(args, policy=masp_policy, optim=optim)

    train_replay_buffer = PrioritizedVectorReplayBuffer(
        args.buffer_size,
        len(train_envs) * len(agents),
        ignore_obs_next=True,
        alpha=args.alpha,
        beta=args.beta
    ) if args.prio_buffer else VectorReplayBuffer(
        args.buffer_size,
        len(train_envs) * len(agents),
        ignore_obs_next=True
    )

    # collector
    train_collector = MultiAgentCollector(
        masp_policy,
        train_envs,
        train_replay_buffer,
        exploration_noise=True,
        number_of_agents=len(agents)
    )
    test_collector = MultiAgentCollector(
        masp_policy,
        test_envs,
        VectorReplayBuffer(
            args.buffer_size,
            len(test_envs) * len(agents),
            ignore_obs_next=True
        ),
        exploration_noise=False,
        number_of_agents=len(agents)
    )
    # train_collector.collect(n_step=args.batch_size * args.training_num)

    if not args.optimize:
        # log
        log_path = os.path.join(args.logdir, 'mpr', 'l_dgn')
        logger = CustomLogger(project='dancing_bees', name=args.model_name)
        writer = SummaryWriter(log_path)
        writer.add_text("args", str(args))
        logger.load(writer)
    weights_path = os.path.join(args.logdir, "mpr", "l_dgn", "weights")
    Path(weights_path).mkdir(parents=True, exist_ok=True)

    def save_best_fn(pol):
        weights_name = os.path.join(
            f"{weights_path}", f"{args.model_name}_best.pth"
        )

        print(f"Saving {args.model_name} Best")
        torch.save(
            pol.policy.state_dict(), weights_name
        )

    def stop_fn(mean_rewards):
        # test_reward:  -4.84
        return mean_rewards > -4.84

    def train_fn(epoch, env_step):
        decay_factor = (1 - pow(e, (
                log(args.eps_train_final) / (
                args.exploration_fraction * args.epoch * args.step_per_epoch))))
        eps = max(args.eps_train * (1 - decay_factor) ** env_step,
                  args.eps_train_final)
        masp_policy.policy.set_eps(eps)
        if not args.optimize:
            if env_step % 1000 == 0:
                logger.write("train/env_step", env_step, {"train/eps": eps})

    def test_fn(epoch, env_step):
        masp_policy.policy.set_eps(args.eps_test)

    def reward_metric(rews):
        return rews.mean()

    if not args.optimize:
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
            train_fn=train_fn,
            test_fn=test_fn,
            # reward_metric
            # stop_fn=stop_fn,
            update_per_step=args.update_per_step,
            test_in_train=False,
            save_best_fn=save_best_fn,
            logger=logger
            # resume_from_log=args.resume
        )
    else:
        # optimizer
        result = offpolicy_optimizer(
            masp_policy,
            train_collector,
            test_collector,
            max_epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            step_per_collect=args.step_per_collect,
            episode_per_test=args.test_num,
            batch_size=args.batch_size,
            train_fn=train_fn,
            test_fn=test_fn,
            # reward_metric
            # stop_fn=stop_fn,
            update_per_step=args.update_per_step,
            test_in_train=False,
            save_best_fn=save_best_fn,
            trial=opt_trial
            # resume_from_log=args.resume
        )

    print(f"Saving {args.model_name} last")
    torch.save(
        masp_policy.policy.state_dict(),
        os.path.join(f"{weights_path}", f"{args.model_name}_last.pth")
    )

    return result, masp_policy


def load_policy(path, args, env):
    # load from existing checkpoint
    args.action_shape = 2

    print(f"Loading agent under {path}")
    if os.path.exists(path):
        # model
        # features
        q_param = {"hidden_sizes": args.dueling_q_hidden_sizes}
        v_param = {"hidden_sizes": args.dueling_v_hidden_sizes}

        net = LDGNNetwork(
            NUMBER_OF_FEATURES,
            args.hidden_emb,
            args.action_shape,
            args.num_heads,
            device=args.device,
            dueling_param=(q_param, v_param)
        )

        optim = torch.optim.Adam(
            net.parameters(), lr=args.lr
        )

        policy = DQNPolicy(
            net,
            optim,
            args.gamma,
            args.n_step,
            target_update_freq=args.target_update_freq
        ).to(args.device)

        masp_policy, _, _, = get_agents(args, policy, optim)
        masp_policy.policy.load_state_dict(torch.load(path))

        print("Successfully restore policy and optim.")
        return masp_policy
    else:
        print("Fail to restore policy and optim.")
        exit(0)
