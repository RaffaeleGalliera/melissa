import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import gym
import numpy as np
import torch

from tianshou.data import (
    Batch,
    CachedReplayBuffer,
    ReplayBuffer,
    ReplayBufferManager,
    VectorReplayBuffer,
    to_numpy,
)
from tianshou.env import BaseVectorEnv, DummyVectorEnv
from tianshou.policy import BasePolicy
import copy

from tianshou.data.collector import Collector


class MultiAgentCollector(Collector):
    def \
            __init__(
            self,
            policy: BasePolicy,
            env: Union[gym.Env, BaseVectorEnv],
            buffer: Optional[ReplayBuffer] = None,
            preprocess_fn: Optional[Callable[..., Batch]] = None,
            exploration_noise: bool = False,
            number_of_agents: int = 0
    ) -> None:
        self.number_of_agents = number_of_agents
        self.done = np.full((len(env),), False)
        self.terminated = np.full((len(env),), False)
        self.truncated = np.full((len(env),), False)
        self.to_be_reset = np.full((len(env),), False)
        self.environment_step = np.full((len(env),), False)
        self.rews = np.zeros((len(env), number_of_agents))
        self.master_episode_rews = np.zeros((len(env),))
        self.envs_to_step_exps = Batch()
        self.exps = None
        super().__init__(policy,
                         env,
                         buffer,
                         preprocess_fn,
                         exploration_noise)

    def _assign_buffer(self, buffer: Optional[ReplayBuffer]) -> None:
        """Check if the buffer matches the constraint."""
        if buffer is None:
            buffer = VectorReplayBuffer(
                self.env_num * self.number_of_agents,
                self.env_num * self.number_of_agents
            )
        elif isinstance(buffer, ReplayBufferManager):
            assert buffer.buffer_num >= self.env_num
            if isinstance(buffer, CachedReplayBuffer):
                assert buffer.cached_buffer_num >= self.env_num
        else:  # ReplayBuffer or PrioritizedReplayBuffer
            assert buffer.maxsize > 0
            if self.env_num > 1:
                if type(buffer) == ReplayBuffer:
                    buffer_type = "ReplayBuffer"
                    vector_type = "VectorReplayBuffer"
                else:
                    buffer_type = "PrioritizedReplayBuffer"
                    vector_type = "PrioritizedVectorReplayBuffer"
                raise TypeError(
                    f"Cannot use {buffer_type}(size={buffer.maxsize}, ...) to collect "
                    f"{self.env_num} envs,\n\tplease use {vector_type}(total_size="
                    f"{buffer.maxsize}, buffer_num={self.env_num}, ...) instead."
                )
        self.buffer = buffer

    def collect(
            self,
            n_step: Optional[int] = None,
            n_episode: Optional[int] = None,
            random: bool = False,
            render: Optional[float] = None,
            no_grad: bool = True,
            gym_reset_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """ Collector resetting environments only when all agents are done.
        """
        assert not self.env.is_async, "Please use AsyncCollector if using async venv."
        if n_step is not None:
            assert n_episode is None, (
                f"Only one of n_step or n_episode is allowed in Collector."
                f"collect, got n_step={n_step}, n_episode={n_episode}."
            )
            assert n_step > 0
            if not n_step % self.env_num == 0:
                warnings.warn(
                    f"n_step={n_step} is not a multiple of #env ({self.env_num}), "
                    "which may cause extra transitions collected into the buffer."
                )
            ready_env_ids = np.arange(self.env_num)
        elif n_episode is not None:
            assert n_episode > 0
            ready_env_ids = np.arange(min(self.env_num, n_episode))
            self.data = self.data[:min(self.env_num, n_episode)]
        else:
            raise TypeError(
                "Please specify at least one (either n_step or n_episode) "
                "in AsyncCollector.collect()."
            )

        start_time = time.time()

        step_count = 0
        episode_count = 0
        master_episode_count = 0
        episode_rews = []
        master_episode_rew = []
        episode_lens = []
        episode_start_indices = []
        coverage = []
        messages_transmitted = []
        while True:
            assert len(self.data) == len(ready_env_ids)
            # restore the state: if the last state is None, it won't store
            last_state = self.data.policy.pop("hidden_state", None)

            # get the next action
            if random:
                try:
                    act_sample = [
                        self._action_space[i].sample() for i in ready_env_ids
                    ]
                except TypeError:  # envpool's action space is not for per-env
                    act_sample = [self._action_space.sample() for _ in ready_env_ids]
                act_sample = self.policy.map_action_inverse(act_sample)  # type: ignore
                self.data.update(act=act_sample)
            else:
                if no_grad:
                    with torch.no_grad():  # faster than retain_grad version
                        # self.data.obs will be used by agent to get result
                        result = self.policy(self.data, last_state)
                else:
                    result = self.policy(self.data, last_state)
                # update state / act / policy into self.data
                policy = result.get("policy", Batch())
                assert isinstance(policy, Batch)
                state = result.get("state", None)
                if state is not None:
                    policy.hidden_state = state  # save state into buffer
                act = to_numpy(result.act)
                if self.exploration_noise:
                    act = self.policy.exploration_noise(act, self.data)
                self.data.update(policy=policy, act=act)

            # get bounded and remapped actions first (not saved into buffer)
            action_remap = self.policy.map_action(self.data.act)
            # step in available envs (we don't take a step if we have the last agent)
            available = np.where(self.to_be_reset==False)[0]
            result = None
            if len(available) > 0:
                result = self.env.step(action_remap[available],
                                       ready_env_ids[available])  # type: ignore
            if result is not None:
                if len(result) == 5:
                    obs_next, rew, terminated, truncated, info = result
                    # This done signal is referred to the next observation
                    # (next agent), not the one we are gathering now
                    next_done = np.logical_or(terminated, truncated)
                    self.rews[ready_env_ids[available]] = rew
                elif len(result) == 4:
                    obs_next, rew, done, info = result
                    if isinstance(info, dict):
                        truncated = info["TimeLimit.truncated"]
                    else:
                        truncated = np.array(
                            [
                                info_item.get("TimeLimit.truncated", False)
                                for info_item in info
                            ]
                        )
                    terminated = np.logical_and(done, ~truncated)
                else:
                    raise ValueError()

            can_reset = False
            ready_to_reset = np.where(self.to_be_reset == True)[0]

            # Result is not none because some environment has not finished yet
            if len(ready_to_reset) and result is not None:
                can_reset = True

                tmp_obs = copy.deepcopy(self.data.obs)
                tmp_obs[available] = obs_next
                obs_next = tmp_obs

                tmp_rew = copy.deepcopy(self.data.rew)
                tmp_rew[available] = rew
                rew = tmp_rew

                tmp_done = copy.deepcopy(self.done)
                tmp_done[available] = next_done
                next_done = tmp_done

                tmp_terminated = copy.deepcopy(self.terminated)
                tmp_terminated[available] = terminated
                terminated = tmp_terminated

                tmp_truncated = copy.deepcopy(self.truncated)
                tmp_truncated[available] = truncated
                truncated = tmp_truncated

                # Envs ready_to_reset should not be reset again
                self.data.info.reset_all[ready_to_reset] = False

                tmp_info = copy.deepcopy(self.data.info)

                tmp_info['environment_step'] = np.array(
                    [
                        info_item.get("environment_step", False)
                        for info_item in tmp_info
                    ]
                )

                tmp_info[available] = info
                info = tmp_info
                info['environment_step'][ready_to_reset] = True

            # Result returns none when every agent in every environment is done
            elif len(ready_to_reset) and result is None:
                can_reset = True

                obs_next = self.data.obs
                rew = self.data.rew
                next_done = self.done
                terminated = self.terminated
                truncated = self.truncated
                # These environments will be reset now
                # Envs to reset should not be reset again
                self.data.info.reset_all[ready_to_reset] = False
                info = self.data.info

                info['environment_step'] = np.array(
                    [
                        info_item.get("environment_step", False)
                        for info_item in info
                    ]
                )
                info['environment_step'][ready_to_reset] = True

            self.to_be_reset = np.array(
                [
                    info_item.get("reset_all", False)
                    for info_item in info
                ]
            )
            self.environment_step = np.array(
                [
                    info_item.get("environment_step", False)
                    for info_item in info
                ]
            )

            # This information will NOT update obs while updating its rewards
            # so it gives r and done referred to step t-1 and step t
            # FOR THE SAME AGENT
            self.data.update(
                obs_next=obs_next,
                rew=rew,
                terminated=self.terminated,
                truncated=self.truncated,
                done=self.done,
                info=info
            )
            if self.preprocess_fn:
                self.data.update(
                    self.preprocess_fn(
                        obs_next=self.data.obs_next,
                        rew=self.data.rew,
                        done=self.data.done,
                        info=self.data.info,
                        policy=self.data.policy,
                        env_id=ready_env_ids,
                    )
                )

            if render:
                self.env.render()
                if render > 0 and not np.isclose(render, 0):
                    time.sleep(render)

            self.envs_to_step_exps = Batch.cat(
                (
                    self.envs_to_step_exps,
                    self.data
                )
            )

            # Add experiences to buffer once every agent in the env has stepped
            if self.environment_step.any():
                stepping_envs = np.where(self.environment_step)[0]
                # Get envs that stepped
                envs_to_step_exps_indices = self.envs_to_step_exps.info.env_id

                # Filter experiences for the stepping environments
                self.exps = self.envs_to_step_exps[
                    np.isin(envs_to_step_exps_indices, stepping_envs)]

                # Remove experiences from envs_to_step_exps
                self.envs_to_step_exps = self.envs_to_step_exps[
                    ~np.isin(envs_to_step_exps_indices, stepping_envs)]

                step_count += len(self.exps)

            # Add rew to exps after they all stepped
            if self.exps is not None:
                self.exps.rew = self.rews[self.exps.info.env_id]
                # Calculate buffer_ids and exps_indices in a vectorized manner
                buffer_ids = (self.exps.info.env_id * self.number_of_agents) + self.exps.obs.agent_id.astype(int)
                exps_indices = np.array(
                    [x.last_index + self.buffer._offset[i] for
                     x, i in zip(self.buffer.buffers[buffer_ids],
                                 buffer_ids)]).flatten()
                # Update indices for Collaborative Learning Algos
                indices = np.full(
                    (self.env_num, self.number_of_agents),
                    -1,
                    dtype=np.int64
                )
                row_indices = self.exps.info.env_id
                col_indices = self.exps.obs.agent_id.astype(int)
                indices[row_indices, col_indices] = exps_indices

                self.exps.info.update(indices=indices[self.exps.info.env_id])

                # Add experiences to buffer
                ptr, ep_rew, ep_len, ep_idx = self.buffer.add(
                    self.exps,
                    buffer_ids=buffer_ids
                )

                # Add stats for done environments
                if np.any(self.exps.done):
                    done_eps = np.where(self.exps.done)[0]
                    episode_count += len(done_eps)
                    episode_lens.append(ep_len[done_eps])
                    episode_rews.append(ep_rew[done_eps,
                    self.exps.obs.agent_id[done_eps].astype(int)])
                    for env, rew in zip(self.exps.info.env_id[done_eps],
                                        ep_rew[done_eps,
                                        self.exps.obs.agent_id[
                                            done_eps].astype(int)]):
                        self.master_episode_rews[env] += rew
                    episode_start_indices.append(ep_idx[done_eps])
                # assert not np.any(ep_len > 5)
                self.exps = None

            # episode rews should be gathered when an agent's episode is done not
            # when everyone resets
            if can_reset:
                env_ind_local = ready_to_reset
                env_ind_global = ready_env_ids[env_ind_local]
                master_episode_count += len(env_ind_local)
                master_episode_rew.append(self.master_episode_rews[env_ind_global])
                self.master_episode_rews[env_ind_global] = 0
                coverage.append(self.data.info.coverage[env_ind_local])
                messages_transmitted.append(self.data.info.messages_transmitted[env_ind_local])
                # now we copy obs_next to obs, but since there might be
                # finished episodes, we have to reset finished envs first.
                self._reset_env_with_ids(
                    env_ind_local, env_ind_global, gym_reset_kwargs
                )
                for i in env_ind_local:
                    self._reset_state(i)

                # remove surplus env id from ready_env_ids
                # to avoid bias in selecting environments
                if n_episode:
                    surplus_env_num = len(ready_env_ids) - (n_episode - master_episode_count)
                    if surplus_env_num > 0:
                        mask = np.ones_like(ready_env_ids, dtype=bool)
                        mask[env_ind_local[:surplus_env_num]] = False
                        ready_env_ids = ready_env_ids[mask]
                        self.data = self.data[mask]

                next_done[ready_to_reset] = False
                terminated[ready_to_reset] = False
                truncated[ready_to_reset] = False

            self.data.obs = self.data.obs_next
            self.done = next_done
            self.terminated = terminated
            self.truncated = truncated

            if (n_step and step_count >= n_step) or \
                    (n_episode and master_episode_count >= n_episode):
                break

        # generate statistics
        self.collect_step += step_count
        self.collect_episode += episode_count
        self.collect_time += max(time.time() - start_time, 1e-9)

        if n_episode:
            self.data = Batch(
                obs={},
                act={},
                rew={},
                terminated={},
                truncated={},
                done={},
                obs_next={},
                info={},
                policy={}
            )
            self.reset_env()

        if episode_count > 0:
            rews, lens, idxs = list(
                map(
                    np.concatenate,
                    [episode_rews, episode_lens, episode_start_indices]
                )
            )
            rew_mean, rew_std = rews.mean(), rews.std()
            len_mean, len_std = lens.mean(), lens.std()
        else:
            rews, lens, idxs = np.array([]), np.array([], int), np.array([], int)
            rew_mean = rew_std = len_mean = len_std = 0

        if master_episode_count > 0:
            msgs, coverages, master_episodes_rew = list(
                map(
                    np.concatenate,
                    [messages_transmitted, coverage, master_episode_rew]
                )
            )
            msg_mean, msg_std = msgs.mean(), msgs.std()
            coverage_mean, coverage_std = coverages.mean(), coverages.std()
            # Spread factor: how "fast" a message is spread through the network
            spread_factor = (coverages / np.sqrt(msgs))
            spread_factor_mean = spread_factor.mean()
            master_episodes_rew_mean, master_episodes_rew_std = master_episodes_rew.mean(), master_episodes_rew.std()
        else:
            msgs, coverages, spread_factor = np.array([]), np.array([], int), np.array([])
            msg_mean = msg_std = coverage_mean = coverage_std = spread_factor_mean = master_episodes_rew_mean = master_episodes_rew_std = 0

        return {
            "n/ep": episode_count,
            "n/graphs": master_episode_count,
            "n/st": step_count,
            "rews": rews,
            "lens": lens,
            "msgs": msgs,
            "coverages": coverages,
            "idxs": idxs,
            "rew": rew_mean,
            "len": len_mean,
            "rew_std": rew_std,
            "len_std": len_std,
            "coverage": coverage_mean,
            "coverage_std": coverage_std,
            "msg": msg_mean,
            "msg_std": msg_std,
            "spread_factor": spread_factor,
            "spread_factor_mean": spread_factor_mean,
            "master_episode_rew": master_episodes_rew_mean,
            "master_episode_rew_std": master_episodes_rew_std
        }
