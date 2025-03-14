import functools
import logging
from typing import TypeVar, Optional, Dict, Any

import gymnasium
import networkx as nx
from gymnasium.utils import seeding
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
import numpy as np
from tianshou.data.batch import Batch

import matplotlib.pyplot as plt

from .utils.constants import NUMBER_OF_FEATURES, RENDER_PAUSE
from .utils.core import World
from .utils.selector import CustomSelector


ActionType = TypeVar("ActionType")


class GraphEnv(AECEnv):
    metadata = {
        'render_modes': ["human"],
        'name': "graph_environment",
        'is_parallelizable': False
    }

    def __init__(
            self,
            graph=None,
            render_mode=None,
            number_of_agents=10,
            radius=10,
            max_cycles=100,
            device='cuda',
            local_ratio=None,
            is_scripted=False,
            is_testing=False,
            random_graph=False,
            dynamic_graph=False,
            all_agents_source=False,
            num_test_episodes=None
    ):
        super().__init__()
        self.seed()
        self.device = device
        self.number_of_agents = number_of_agents
        self.render_mode = render_mode
        self.renderOn = False
        self.local_ratio = local_ratio
        self.radius = radius
        self.is_new_round = None

        self.world = World(
            graph=graph,
            number_of_agents=self.number_of_agents,
            radius=radius,
            np_random=self.np_random,
            is_scripted=is_scripted,
            is_testing=is_testing,
            random_graph=random_graph,
            dynamic_graph=dynamic_graph,
            all_agents_source=all_agents_source,
            num_test_episodes=num_test_episodes
        )

        self.agents = [agent.name for agent in self.world.agents]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(
            zip(self.agents, list(range(len(self.possible_agents))))
        )
        self._agent_selector = CustomSelector(self.agents)

        # Store a shared obs matrix that we'll update incrementally.
        # Shape: (number_of_agents, 2 + NUMBER_OF_FEATURES)
        self.obs_matrix = np.zeros(
            (self.number_of_agents, 2 + NUMBER_OF_FEATURES), dtype=np.float32
        )

        # set observation/action spaces
        # Flattened dimension: N * (2 + NUMBER_OF_FEATURES) + 1 controlling agent index
        obs_dim = self.number_of_agents * (2 + NUMBER_OF_FEATURES) + 1

        self.action_spaces = {}
        self.observation_spaces = {}
        for agent in self.world.agents:
            self.observation_spaces[agent.name] = gymnasium.spaces.Dict({
                'observation': gymnasium.spaces.Box(
                    low=-1e6,
                    high=1e6,
                    shape=(obs_dim,),
                    dtype=np.float32,
                ),
                'action_mask': gymnasium.spaces.Box(
                    low=0,
                    high=1,
                    shape=(2,),
                    dtype=np.int8,
                ),
            })
            self.action_spaces[agent.name] = gymnasium.spaces.Discrete(2)

        self.state_space = gymnasium.spaces.Box(
            low=-1e6,
            high=1e6,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.max_cycles = max_cycles
        self.num_moves = 0
        self.current_actions = [None] * self.number_of_agents
        self.episode_rewards_sum = 0.0

        self.reset()

    def enable_render(self, mode="human"):
        if not self.renderOn and mode == "human":
            self.renderOn = True

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if self.render_mode == "human" and self.world.agents:
            if self.world.dynamic_graph:
                draw_graph(self.world.pre_move_graph, self.world.pre_move_agents)
            draw_graph(self.world.graph, self.world.agents)

        return

    def close(self):
        if self.renderOn:
            self.renderOn = False

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent) -> gymnasium.spaces.Space:
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent) -> gymnasium.spaces.Space:
        return self.action_spaces[agent]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def get_info(self, agent: str):
        current_topic = self.world.current_topic

        interested_agents = [
            ag for ag in self.world.agents
            if ag.topic_of_interest == current_topic
        ]
        num_interested = len(interested_agents)

        interested_with_message = sum([
            1 for ag in interested_agents
            if sum(ag.state.received_from) > 0 or ag.state.message_origin == 1
        ])

        uninterested_agents = [
            ag for ag in self.world.agents
            if ag.topic_of_interest != current_topic
        ]
        uninterested_with_message = sum([
            1 for ag in uninterested_agents
            if sum(ag.state.received_from) > 0 or ag.state.message_origin == 1
        ])

        coverage_all = sum([
            1 for ag in self.world.agents
            if sum(ag.state.received_from) > 0 or ag.state.message_origin == 1
        ]) / self.world.num_agents

        return {
            "logger_stats": {
                'total_messages_transmitted': self.world.messages_transmitted,
                'coverage': coverage_all,
                'messages_sent': sum([ag.messages_transmitted for ag in self.world.agents]),
                'messages_received': sum([sum(ag.state.received_from) for ag in self.world.agents]),
                'n_neighbours': sum([sum(ag.one_hop_neighbours_ids) for ag in self.world.agents]),
                'topic_of_the_message': current_topic,
                'interested_agents': num_interested,
                'coverage_interested_fraction': (
                    interested_with_message / num_interested if num_interested > 0 else 0.0
                ),
                'coverage_interested_count': interested_with_message,
                'uninterested_with_message': uninterested_with_message,
                'episode_rewards_sum': self.episode_rewards_sum
            },
        }

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observe(self, agent: str):
        """
        Return a single flattened array representing the entire 'obs_matrix'
        plus the controlling-agent index appended at the end.
        This means each agent sees the full global state.
        """
        # controlling index
        controlling_idx = self.agent_name_mapping[agent]
        # flatten shape (number_of_agents*(2+NUMBER_OF_FEATURES),)
        obs_flat = self.obs_matrix.reshape(-1)
        # append controlling-agent index => shape (... + 1,)
        obs_final = np.concatenate([obs_flat, [controlling_idx]]).astype(np.float32)

        action_mask = np.array([1, 1], dtype=np.int8)
        if self.terminations[agent] or self.truncations[agent]:
            action_mask = np.array([0, 0], dtype=np.int8)

        self.infos[agent]['env_step'] = self.num_moves
        self.infos[agent]['environment_step'] = False
        self.infos[agent]['explicit_reset'] = False

        if all(value for key, value in self.terminations.items() if key in self.agents) and len(self.agents) == 1:
            self.is_new_round = False
            self.infos[agent]["explicit_reset"] = True

        if self.is_new_round:
            self.infos[agent]['environment_step'] = True
            self.is_new_round = False

        return {
            "observation": obs_final,
            "action_mask": action_mask
        }

    def state(self):
        # If you need a global state, just flatten self.obs_matrix
        return self.obs_matrix.reshape(-1)

    def reset(self, seed=None, return_info=False, options=None):
        if seed is not None:
            self.seed(seed=seed)
        self.world.np_random = self.np_random

        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.rewards = {name: 0.0 for name in self.agents}
        self._cumulative_rewards = {name: 0.0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}
        self.num_moves = 0

        self.world.reset()
        self.episode_rewards_sum = 0.0

        # Initialize and fill in self.obs_matrix for all agents
        self._init_obs_matrix()

        # Filter out agents that have the message, etc.
        self.agents = [
            agent.name for agent in self.world.agents
            if (sum(agent.state.received_from) and not agent.state.message_origin)
        ]
        self._agent_selector.enable(self.agents)

        self.agent_selection = self._agent_selector.next()
        self.current_actions = [None] * self.number_of_agents

    def _init_obs_matrix(self):
        """
        Initialize self.obs_matrix from scratch, used at the start of an episode.
        """
        for i, ag in enumerate(self.world.agents):
            self._update_obs_matrix_for_agent(i, ag)

    def _update_obs_matrix_for_agent(self, idx: int, agent_obj):
        """
        Update the row self.obs_matrix[idx] to reflect the agent’s
        latest pos, message status, steps_taken, etc.
        Customize the layout to match your desired columns.
        Example layout:
            0: pos_x
            1: pos_y
            2: has_message
            3: topic_of_interest
            4: carried_topic
            5: steps_taken
            ...
        """
        pos_x, pos_y = agent_obj.pos

        self.obs_matrix[idx, 0] = pos_x
        self.obs_matrix[idx, 1] = pos_y
        self.obs_matrix[idx, 2] = sum(agent_obj.one_hop_neighbours_ids)
        self.obs_matrix[idx, 3] = agent_obj.messages_transmitted
        self.obs_matrix[idx, 4] = agent_obj.action if agent_obj.action is not None else 0
        self.obs_matrix[idx, 5] = agent_obj.topic_of_interest
        self.obs_matrix[idx, 6] = agent_obj.state.carried_topic if agent_obj.state.carried_topic is not None else -1

    def _was_dead_step(self, action: ActionType) -> None:
        """
        Same default was_dead_step method used by PettingZoo, but avoids clearing
        rewards for all agents at the end.
        """
        if action is not None:
            raise ValueError("when an agent is dead, the only valid action is None")

        agent = self.agent_selection
        assert (
            self.terminations[agent] or self.truncations[agent]
        ), "an agent that was not dead attempted to be removed"
        del self.terminations[agent]
        del self.truncations[agent]
        del self.rewards[agent]
        del self._cumulative_rewards[agent]
        del self.infos[agent]
        self.agents.remove(agent)

        _deads_order = [
            agent
            for agent in self.agents
            if (self.terminations[agent] or self.truncations[agent])
        ]
        if _deads_order:
            if getattr(self, "_skip_agent_selection", None) is None:
                self._skip_agent_selection = self.agent_selection
            self.agent_selection = _deads_order[0]
        else:
            if getattr(self, "_skip_agent_selection", None) is not None:
                self.agent_selection = self._skip_agent_selection
            self._skip_agent_selection = None

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._agent_selector.disable(self.agent_selection)
            self._was_dead_step(None)
            return

        current_agent = self.agent_selection
        current_idx = self.agent_name_mapping[self.agent_selection]
        self.current_actions[current_idx] = action
        agent_tmp = self.world.agents[int(current_agent)]
        agent_tmp.steps_taken += 1

        self._cumulative_rewards[current_agent] = 0
        self.agent_selection = self._agent_selector.next()

        # If we've got an action from every agent in the round:
        if not self.agent_selection:
            self._accumulate_rewards()
            self._clear_rewards()
            self._execute_world_step()
            self.num_moves += 1

            for agent in self.agents:
                agent_obj = self.world.agents[int(agent)]
                if agent_obj.steps_taken >= 4 and not agent_obj.truncated:
                    agent_obj.truncated = True
                    self.terminations[agent_obj.name] = True

            self.agents = [
                agent.name for agent in self.world.agents
                if (sum(agent.state.received_from)
                    and not agent.state.message_origin
                    and agent.name in self.terminations)
            ]
            self._agent_selector.enable(self.agents)
            self._agent_selector.start_new_round()
            self.is_new_round = True
            self.agent_selection = self._agent_selector.next()

            self.current_actions = [None] * self.number_of_agents

            n_received = sum(
                [1 for agent in self.world.agents if
                 sum(agent.state.received_from) or agent.state.message_origin]
            )

            if n_received == self.number_of_agents and self.render_mode == 'human':
                cds = [agent.id for agent in self.world.agents if agent.messages_transmitted > 0]
                print(
                    f"Every agent has received the message, terminating in {self.num_moves}, "
                    f"messages transmitted: {self.world.messages_transmitted}"
                )
                print(cds)
            if self.render_mode == 'human':
                self.render()

        self.infos[self.agent_selection] = self.get_info(self.agent_selection)
        self._deads_step_first()

    def _execute_world_step(self):
        # Convert each agent's discrete action into scenario action
        for i, agent in enumerate(self.world.agents):
            action = self.current_actions[i]
            scenario_action = [action]
            self._set_action(scenario_action, agent, self.action_spaces[agent.name])

        self.world.step()

        # Now that the environment might have changed (message states, etc.),
        # update obs rows for all agents
        for i, ag in enumerate(self.world.agents):
            self._update_obs_matrix_for_agent(i, ag)

        global_reward = 0.0
        if self.local_ratio is not None:
            global_reward = float(self.global_reward())

        # Compute each agent's reward and accumulate to episode sum
        for agent in [a for a in self.world.agents if a.name in self.agents]:
            agent_reward = float(self.reward(agent))
            if self.local_ratio is not None:
                reward = (
                        global_reward * (1 - self.local_ratio)
                        + agent_reward * self.local_ratio
                )
            else:
                reward = agent_reward

            self.rewards[agent.name] = reward

            self.episode_rewards_sum += reward

    def _set_action(self, action, agent, param):
        agent.action = action[0]
        action = action[1:]
        assert len(action) == 0

    def global_reward(self):
        pass

    def reward(self, agent):
        """
            Computes a reward that prioritizes forwarding the message
            only to agents interested in the topic the agent carries.
            """
        carried_topic = agent.state.carried_topic

        one_hop_neighbor_indices = np.where(agent.one_hop_neighbours_ids)[0]
        two_hop_neighbor_indices = np.where(agent.two_hop_neighbours_ids)[0]

        one_hop_interested = [
            idx for idx in one_hop_neighbor_indices
            if self.world.agents[idx].topic_of_interest == carried_topic
        ]
        two_hop_interested = [
            idx for idx in two_hop_neighbor_indices
            if self.world.agents[idx].topic_of_interest == carried_topic
        ]

        num_covered_interested_2hop = 0
        for idx in two_hop_interested:
            if sum(self.world.agents[idx].state.received_from) > 0 or \
                    self.world.agents[idx].state.message_origin == 1:
                num_covered_interested_2hop += 1

        total_interested_2hop = len(two_hop_interested)
        coverage_ratio = (num_covered_interested_2hop / total_interested_2hop) if total_interested_2hop > 0 else 0.0

        # Start with the coverage ratio as the base reward
        reward = coverage_ratio

        if agent.action:  # agent transmits
            uninterested_neighbors = [
                idx for idx in one_hop_neighbor_indices
                if self.world.agents[idx].topic_of_interest != carried_topic
            ]
            if len(one_hop_neighbor_indices) > 0:
                penalty_uninterested = len(uninterested_neighbors) / len(one_hop_neighbor_indices)
            else:
                penalty_uninterested = 0

            repeated_sends = [
                idx for idx in one_hop_neighbor_indices
                if sum(self.world.agents[idx].state.received_from) > 0  # or messages_transmitted > 0
            ]
            penalty_already_covered = len(repeated_sends) / len(one_hop_neighbor_indices) if len(
                one_hop_neighbor_indices) else 0

            penalty = penalty_uninterested + penalty_already_covered

            # Subtract penalty from coverage-based reward
            reward -= penalty

        else:  # agent does NOT transmit
            # If it has interested neighbors that haven't received the message, penalize
            uncovered_interested = [
                idx for idx in one_hop_interested
                if (sum(self.world.agents[idx].state.received_from) == 0
                    and self.world.agents[idx].state.message_origin == 0)
            ]
            if len(uncovered_interested) > 0:
                penalty_uncovered = len(uncovered_interested) / len(one_hop_interested)
                reward -= penalty_uncovered

        return reward


def draw_graph(graph, agent_list):
    plt.clf()
    pos = nx.get_node_attributes(graph, "pos")
    color_map = []
    for node in graph:
        # Color each node depending on message state
        if sum(agent_list[node].state.received_from) and not sum(agent_list[node].state.transmitted_to):
            color_map.append('green')
        elif agent_list[node].state.message_origin:
            color_map.append('blue')
        elif agent_list[node].messages_transmitted > 1:
            color_map.append('purple')
        elif sum(agent_list[node].state.transmitted_to):
            color_map.append('red')
        else:
            color_map.append("yellow")

    nx.draw(graph, node_color=color_map, pos=pos, with_labels=True)
    plt.pause(RENDER_PAUSE)


def make_env(raw_env):
    def env(**kwargs):
        env_ = raw_env(**kwargs)
        env_ = wrappers.AssertOutOfBoundsWrapper(env_)
        env_ = wrappers.OrderEnforcingWrapper(env_)
        return env_
    return env


env = make_env(GraphEnv)
