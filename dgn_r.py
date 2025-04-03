import argparse
import os
import pprint
import time
from math import e, log, pow
from pathlib import Path
from typing import List, Tuple

import gymnasium
import numpy as np
import optuna
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import (
    VectorReplayBuffer,
    PrioritizedVectorReplayBuffer
)
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.policy import BasePolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import WandbLogger

from graph_env.env.utils.constants import NUMBER_OF_FEATURES
from graph_env.env.utils.networks.dgn_r import DGNRNetwork

from graph_env.env.utils.policies.multi_agent_managers.collaborative_shared_policy import MultiAgentCollaborativeSharedPolicy
from graph_env.env.utils.policies.dgn import DGNPolicy

from graph_env.env.utils.collectors.collective_experience_collector import CollectiveExperienceCollector
from graph_env.env.utils.hyp_optimizer.offpolicy_opt import offpolicy_optimizer

from common import get_args, get_env, select_aggregator

def get_agents(
    args: argparse.Namespace,
    policy: BasePolicy = None,
    optim: torch.optim.Optimizer = None
) -> Tuple[BasePolicy, torch.optim.Optimizer, List[str]]:
    """
    Build or return the MultiAgentCollaborativeSharedPolicy, the optimizer, and list of agents.
    """
    env = get_env(number_of_agents=args.n_agents)
    observation_space = env.observation_space['observation'] if isinstance(
        env.observation_space, (gymnasium.spaces.Dict, gymnasium.spaces.Dict)
    ) else env.observation_space
    args.state_shape = observation_space.shape or observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = 1  # Not strictly used if env is discrete, but keep for consistency

    if policy is None:
        # Construct aggregator
        aggregator = select_aggregator(args.aggregator_function)

        # Q and V param
        q_param = {"hidden_sizes": args.dueling_q_hidden_sizes}
        v_param = {"hidden_sizes": args.dueling_v_hidden_sizes}

        # Create network
        net = DGNRNetwork(
            NUMBER_OF_FEATURES,
            args.hidden_emb,
            args.action_shape,
            args.num_heads,
            device=args.device,
            dueling_param=(q_param, v_param),
            aggregator_function=aggregator
        )

        # Optimizer
        optim = torch.optim.Adam(net.parameters(), lr=args.lr)

        # Policy
        policy = DGNPolicy(
            model=net,
            optim=optim,
            discount_factor=args.gamma,
            estimation_step=args.n_step,
            target_update_freq=args.target_update_freq,
            action_space=env.action_space  # If discrete, Tianshou automatically handles that
        ).to(args.device)

    masp_policy = MultiAgentCollaborativeSharedPolicy(policy, env)
    return masp_policy, optim, env.agents


def watch(args: argparse.Namespace, masp_policy: BasePolicy = None) -> None:
    """
    Load a pre-trained policy (if masp_policy not given) and run it in watch mode.
    """
    weights_path = os.path.join(
        args.logdir, "mpr", args.algorithm, "weights", f"{args.model_name}"
    )

    # Create a single test environment
    env = DummyVectorEnv([
        lambda: get_env(
            number_of_agents=args.n_agents,
            is_scripted=args.mpr_policy,
            is_testing=True,
            dynamic_graph=args.dynamic_graph,
            render_mode="human",
            all_agents_source=True,
            num_test_episodes=args.test_num
        )
    ])

    # Set seeds
    env.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load the policy if none is provided
    if masp_policy is None:
        masp_policy = load_policy(weights_path, args, env)

    masp_policy.policy.eval()
    masp_policy.policy.set_eps(args.eps_test)

    # Create collector in watch mode
    collector = CollectiveExperienceCollector(
        agents_num=args.n_agents,
        policy=masp_policy,
        env=env,
        exploration_noise=False
    )

    # Collect some episodes
    result = collector.collect(n_episode=args.test_num * args.n_agents)
    pprint.pprint(result)
    time.sleep(5)

def train_agent(
    args: argparse.Namespace,
    masp_policy: BasePolicy = None,
    optim: torch.optim.Optimizer = None,
    opt_trial: optuna.Trial = None
) -> Tuple[dict, BasePolicy]:
    """
    Main training loop, with optional hyperparameter optimization
    (when args.optimize is True).
    """

    train_envs = SubprocVectorEnv([
        lambda: get_env(
            number_of_agents=args.n_agents,
            dynamic_graph=args.dynamic_graph
        )
        for _ in range(args.training_num)
    ])
    test_envs = SubprocVectorEnv([
        lambda: get_env(
            number_of_agents=args.n_agents,
            dynamic_graph=args.dynamic_graph,
            is_testing=True,
            num_test_episodes=args.test_num
        )
    ])

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # Build/Load policy
    masp_policy, optim, agents = get_agents(args, policy=masp_policy, optim=optim)

    # Replay buffers
    if args.prio_buffer:
        replay_buffer = PrioritizedVectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs) * len(agents),
            ignore_obs_next=True,
            alpha=args.alpha,
            beta=args.beta
        )
    else:
        replay_buffer = VectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs) * len(agents),
            ignore_obs_next=True
        )

    # Collectors
    train_collector = CollectiveExperienceCollector(
        agents_num=len(agents),
        policy=masp_policy,
        env=train_envs,
        buffer=replay_buffer,
        exploration_noise=True
    )
    test_collector = CollectiveExperienceCollector(
        agents_num=len(agents),
        policy=masp_policy,
        env=test_envs,
        exploration_noise=False
    )

    # Pre-collect if desired
    train_collector.reset()
    train_collector.collect(n_step=args.batch_size * args.training_num)

    # Setup logger
    logger, writer = None, None
    log_path = os.path.join(args.logdir, "mpr", args.algorithm)
    weights_path = os.path.join(log_path, "weights")
    Path(weights_path).mkdir(parents=True, exist_ok=True)

    if not args.optimize:
        logger = WandbLogger(project="group_interest_dissemination", name=args.model_name)
        writer = SummaryWriter(log_path)
        writer.add_text("args", str(args))
        if logger is not None:
            logger.load(writer)

    def save_best_fn(pol: BasePolicy):
        """
        Save the best model checkpoint.
        """
        best_path = os.path.join(weights_path, f"{args.model_name}_best.pth")
        print(f"Saving best model to {best_path}")
        torch.save(pol.policy.state_dict(), best_path)
        logger.wandb_run.save(best_path)

    def train_fn(epoch: int, env_step: int):
        """
        Adjust epsilon during training based on env steps.
        """
        decay_factor = 1.0 - pow(
            e,
            (log(args.eps_train_final) / (args.exploration_fraction * args.epoch * args.step_per_epoch))
        )
        eps = max(args.eps_train * (1.0 - decay_factor) ** env_step, args.eps_train_final)
        masp_policy.policy.set_eps(eps)

        # Logging
        if logger is not None and env_step % 1000 == 0:
            logger.write("train/env_step", env_step, {"train/eps": eps})

    def test_fn(epoch: int, env_step: int):
        """
        Set epsilon to test level for evaluation.
        """
        masp_policy.policy.set_eps(args.eps_test)

    # Decide trainer vs. optimizer
    if not args.optimize:
        result = OffpolicyTrainer(
            policy=masp_policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            step_per_collect=args.step_per_collect,
            episode_per_test=args.test_num,
            batch_size=args.batch_size,
            train_fn=train_fn,
            test_fn=test_fn,
            update_per_step=args.update_per_step,
            test_in_train=False,
            save_best_fn=save_best_fn,
            logger=logger
        ).run()
    else:
        # hyperparameter optimization
        result = offpolicy_optimizer(
            policy=masp_policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            step_per_collect=args.step_per_collect,
            episode_per_test=args.test_num,
            batch_size=args.batch_size,
            train_fn=train_fn,
            test_fn=test_fn,
            update_per_step=args.update_per_step,
            test_in_train=False,
            save_best_fn=save_best_fn,
            trial=opt_trial
        )

    # Always save the final model
    last_path = os.path.join(weights_path, f"{args.model_name}_last.pth")
    print(f"Saving last model to {last_path}")
    torch.save(masp_policy.policy.state_dict(), last_path)
    logger.wandb_run.save(last_path)

    return result, masp_policy


def load_policy(path: str, args: argparse.Namespace, env: DummyVectorEnv) -> BasePolicy:
    """
    Load a saved policy from a given path. If it doesn't exist, exit.
    """
    args.action_shape = 2  # if your action space is discrete of size 2, etc.
    print(f"Loading agent checkpoint from {path}")
    if not os.path.exists(path):
        print("Fail to restore policy and optim. Exiting.")
        exit(0)

    # Construct aggregator
    aggregator = select_aggregator(args.aggregator_function)
    q_param = {"hidden_sizes": args.dueling_q_hidden_sizes}
    v_param = {"hidden_sizes": args.dueling_v_hidden_sizes}

    # Build the same network used in training
    net = DGNRNetwork(
        NUMBER_OF_FEATURES,
        args.hidden_emb,
        args.action_shape,
        args.num_heads,
        device=args.device,
        dueling_param=(q_param, v_param),
        aggregator_function=aggregator
    )

    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    policy = DGNPolicy(
        model=net,
        optim=optim,
        discount_factor=args.gamma,
        estimation_step=args.n_step,
        target_update_freq=args.target_update_freq,
        action_space=env.action_space
    ).to(args.device)

    # Wrap in MultiAgentCollaborativeSharedPolicy
    masp_policy, _, _ = get_agents(args, policy, optim)

    # Load weights
    masp_policy.policy.load_state_dict(torch.load(path))
    print("Successfully restored policy and optimizer.")
    return masp_policy


if __name__ == '__main__':
    args = get_args()
    args.algorithm = "dgn_r"
    if args.watch:
        watch(args)

    elif args.optimize:
        pass

    else:
        result, masp_policy = train_agent(args)
        pprint.pprint(result)