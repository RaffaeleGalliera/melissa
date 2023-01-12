import argparse
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv, DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, MultiAgentPolicyManager, DQNPolicy
from tianshou.trainer import onpolicy_trainer, offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils.net.common import Net

from graph_env import graph_env_v0
from graph_env.env.utils.constants import NUMBER_OF_AGENTS, RADIUS_OF_INFLUENCE, NUMBER_OF_FEATURES
from graph_env.env.utils.core import load_testing_graph

from torch_geometric.nn import GCNConv

os.environ["SDL_VIDEODRIVER"]="x11"

DEVICE = 'cuda'
class GCN(nn.Module):
    def __init__(self, state_shape, action_shape):
        super(GCN, self).__init__()
        self.action_shape = action_shape
        self.conv1 = GCNConv(NUMBER_OF_FEATURES, 128).to(DEVICE)
        # self.prebuilt_net = Net(128, 2, [128, 128], device=DEVICE).to(DEVICE)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, 128)
        self.logits = nn.Linear(128, 2)

    def forward(self, obs, state=None, info={}):
        logits = []
        for observation in obs.observation:
            # TODO: Need here we move tensors to CUDA, cannot just put it in Batch because of data time -> slows down
            x = torch.Tensor(observation[2]).to(device=DEVICE, dtype=torch.float32)
            edge_index = torch.Tensor(observation[0]).to(device=DEVICE, dtype=torch.long)
            x = self.conv1(x, edge_index)
            x = x.relu()
            x = self.lin2(x)
            x = x.relu()
            x = self.lin3(x)
            x = x.relu()
            x = torch.nn.functional.dropout(x, training=self.training)
            x = self.logits(x)
            logits.append(x[observation[3]].flatten())
        logits = torch.stack(logits)
        return logits, state


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--eps-test', type=float, default=0.05)
    parser.add_argument('--eps-train', type=float, default=0.003)
    parser.add_argument('--buffer-size', type=int, default=500000)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument(
        '--gamma', type=float, default=0.9, help='a smaller gamma favors earlier win'
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
    parser.add_argument('--training-num', type=int, default=5)
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
    # ppo special
    parser.add_argument('--vf-coef', type=float, default=0.25)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--rew-norm', type=int, default=1)
    parser.add_argument('--dual-clip', type=float, default=None)
    parser.add_argument('--value-clip', type=int, default=1)
    parser.add_argument('--norm-adv', type=int, default=1)
    parser.add_argument('--recompute-adv', type=int, default=0)
    parser.add_argument('--resume', action="store_true")
    parser.add_argument("--save-interval", type=int, default=4)
    parser.add_argument('--render', type=float, default=0.1)

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
        env.observation_space, gym.spaces.Dict
    ) else env.observation_space
    args.state_shape = observation_space['observation'].shape or observation_space['observation'].n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = 1

    if agents is None:
        agents = []
        optims = []

        # model
        net = GCN(
            args.state_shape,
            args.action_shape,
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
            optims.append(optim)

    policy = MultiAgentPolicyManager(
        agents, env, action_scaling=True, action_bound_method='clip'
    )

    return policy, optims, env.agents


def train_agent(
    args: argparse.Namespace = get_args(),
    agents: Optional[List[BasePolicy]] = None,
    optims: Optional[List[torch.optim.Optimizer]] = None,
) -> Tuple[dict, BasePolicy]:
    train_envs = SubprocVectorEnv([get_env for _ in range(args.training_num)])
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
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        model_save_path = os.path.join(
            args.logdir, "mpr", "dqn", "weights", f"policy.pth"
        )
        torch.save(
            policy.policies[agents[0]].state_dict(), model_save_path
        )

    def stop_fn(mean_rewards):
        return False

    def train_fn(epoch, env_step):
        [agent.set_eps(args.eps_train) for agent in policy.policies.values()]

    def test_fn(epoch, env_step):
        [agent.set_eps(args.eps_test) for agent in policy.policies.values()]

    def reward_metric(rews):
        return rews[:, 0]

    # trainer
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.repeat_per_collect,
        args.test_num,
        args.batch_size,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        update_per_step=args.update_per_step,
        test_in_train=True,
        save_best_fn=save_best_fn,
        logger=logger,
        resume_from_log=args.resume
    )

    return result, policy


def watch(
    args: argparse.Namespace = get_args(), policy: Optional[BasePolicy] = None
) -> None:
    env = SubprocVectorEnv([lambda: get_env(graph=load_testing_graph(f"testing_graph_{NUMBER_OF_AGENTS}.gpickle"), render_mode='human')])
    if not policy:
        warnings.warn(
            "watching random agents, as loading pre-trained policies is "
            "currently not supported"
        )
        policy, _, _ = get_agents(args)
    policy.eval()
    collector = Collector(policy, env)
    result = collector.collect(n_episode=1, render=args.render)
    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews[:, 0].mean()}, length: {lens.mean()}")
