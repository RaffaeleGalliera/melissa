from collections import defaultdict
from tianshou.data.collector import Collector
from tianshou.data import CollectStats
from dataclasses import dataclass
from typing import Any, cast
import warnings
import numpy as np
import time
from tianshou.data.types import RolloutBatchProtocol
import torch
from tianshou.utils.print import DataclassPPrintMixin
from tianshou.data import Batch, to_numpy, SequenceSummaryStats

@dataclass(kw_only=True)
class DictOfSequenceSummaryStats(DataclassPPrintMixin):
    """A data structure for storing a dictionary of sequence summary statistics."""
    stats: dict[str, SequenceSummaryStats]

    @classmethod
    def from_dict(cls, stats: dict[str, list[float]]) -> "DictOfSequenceSummaryStats":
        """
        Transform a dictionary of lists into a DictOfSequenceSummaryStats object.

        Args:
            stats (dict[str, list[float]]): A dictionary where keys are strings and values are lists of floats.

        Returns:
            DictOfSequenceSummaryStats: An instance of DictOfSequenceSummaryStats.
        """
        return cls(stats={key: SequenceSummaryStats.from_sequence(value) for key, value in stats.items()})


@dataclass(kw_only=True)
class CollectStatsWithInfo(CollectStats):
    """A data structure for storing the statistics of a collection of episodes."""
    info: DictOfSequenceSummaryStats = None


class SingleAgentCollector(Collector):
    """A custom collector that extends the default collector with additional functionality."""

    def collect(
            self,
            n_step: int | None = None,
            n_episode: int | None = None,
            random: bool = False,
            render: float | None = None,
            no_grad: bool = True,
            gym_reset_kwargs: dict[str, Any] | None = None,
    ) -> CollectStatsWithInfo:
        """
        Collect a specified number of steps or episodes.

        To ensure unbiased sampling result with n_episode option, this function will
        first collect ``n_episode - env_num`` episodes, then for the last ``env_num``
        episodes, they will be collected evenly from each env.

        Args:
            n_step (int | None): How many steps you want to collect.
            n_episode (int | None): How many episodes you want to collect.
            random (bool): Whether to use random policy for collecting data. Default to False.
            render (float | None): The sleep time between rendering consecutive frames. Default to None (no rendering).
            no_grad (bool): Whether to retain gradient in policy.forward(). Default to True (no gradient retaining).
            gym_reset_kwargs (dict[str, Any] | None): Extra keyword arguments to pass into the environment's reset function. Defaults to None (extra keyword arguments).

        Returns:
            CollectStatsWithInfo: A dataclass object containing the collection statistics.

        The method works as follows:
        1. **Assertions and Initializations**:
            - Ensures only one of `n_step` or `n_episode` is specified.
            - Sets up `ready_env_ids` based on `n_step` or `n_episode`.

        2. **Main Collection Loop**:
            - Collects data until the specified number of steps or episodes is reached.
            - Samples actions either randomly or using the policy.
            - Steps the environment(s) and updates the data.
            - Collects and updates statistics for each episode.

        3. **Statistics and Return**:
            - Updates collection statistics and returns them in a `CollectStatsWithInfo` object.
        """
        assert not self.env.is_async, "Please use AsyncCollector if using async venv."
        if n_step is not None:
            assert n_episode is None, (
                f"Only one of n_step or n_episode is allowed in Collector."
                f"collect, got n_step={n_step}, n_episode={n_episode}."
            )
            assert n_step > 0
            if n_step % self.env_num != 0:
                warnings.warn(
                    f"n_step={n_step} is not a multiple of #env ({self.env_num}), "
                    "which may cause extra transitions collected into the buffer.",
                )
            ready_env_ids = np.arange(self.env_num)
        elif n_episode is not None:
            assert n_episode > 0
            ready_env_ids = np.arange(min(self.env_num, n_episode))
            self.data = self.data[: min(self.env_num, n_episode)]
        else:
            raise TypeError(
                "Please specify at least one (either n_step or n_episode) "
                "in AsyncCollector.collect().",
            )

        start_time = time.time()

        step_count = 0
        episode_count = 0
        episode_returns: list[float] = []
        episode_lens: list[int] = []
        episode_start_indices: list[int] = []
        episode_info: list[dict[str, Any]] = []
        while True:
            assert len(self.data) == len(ready_env_ids)
            # restore the state: if the last state is None, it won't store
            last_state = self.data.policy.pop("hidden_state", None)

            # get the next action
            if random:
                try:
                    act_sample = [self._action_space[i].sample() for i in ready_env_ids]
                except TypeError:  # envpool's action space is not for per-env
                    act_sample = [self._action_space.sample() for _ in ready_env_ids]
                act_sample = self.policy.map_action_inverse(act_sample)  # type: ignore
                self.data.update(act=act_sample)
            else:
                if no_grad:
                    with torch.no_grad():  # faster than retain_grad version
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
            # step in env

            obs_next, rew, terminated, truncated, info = self.env.step(
                action_remap,
                ready_env_ids,
            )
            done = np.logical_or(terminated, truncated)

            self.data.update(
                obs_next=obs_next,
                rew=rew,
                terminated=terminated,
                truncated=truncated,
                done=done,
                info=info,
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
                        act=self.data.act,
                    ),
                )

            if render:
                self.env.render()
                if render > 0 and not np.isclose(render, 0):
                    time.sleep(render)

            # add data into the buffer
            ptr, ep_rew, ep_len, ep_idx = self.buffer.add(self.data, buffer_ids=ready_env_ids)

            # collect statistics
            step_count += len(ready_env_ids)
            # Custom info statistics
            episode_info.extend(info)

            if np.any(done):
                env_ind_local = np.where(done)[0]
                env_ind_global = ready_env_ids[env_ind_local]
                episode_count += len(env_ind_local)
                episode_lens.extend(ep_len[env_ind_local])
                episode_returns.extend(ep_rew[env_ind_local])
                episode_start_indices.extend(ep_idx[env_ind_local])
                # now we copy obs_next to obs, but since there might be
                # finished episodes, we have to reset finished envs first.
                self._reset_env_with_ids(env_ind_local, env_ind_global, gym_reset_kwargs)
                for i in env_ind_local:
                    self._reset_state(i)

                # remove surplus env id from ready_env_ids
                # to avoid bias in selecting environments
                if n_episode:
                    surplus_env_num = len(ready_env_ids) - (n_episode - episode_count)
                    if surplus_env_num > 0:
                        mask = np.ones_like(ready_env_ids, dtype=bool)
                        mask[env_ind_local[:surplus_env_num]] = False
                        ready_env_ids = ready_env_ids[mask]
                        self.data = self.data[mask]

            self.data.obs = self.data.obs_next

            if (n_step and step_count >= n_step) or (n_episode and episode_count >= n_episode):
                break

        # generate statistics
        self.collect_step += step_count
        self.collect_episode += episode_count
        collect_time = max(time.time() - start_time, 1e-9)
        self.collect_time += collect_time
        # Add the custom statistics to the episode_info
        fused_logger_stats = defaultdict(list)
        for d in episode_info:
            for key, value in d['logger_stats'].items():
                fused_logger_stats[key].append(value)
        episode_info = fused_logger_stats

        if n_episode:
            data = Batch(
                obs={},
                act={},
                rew={},
                terminated={},
                truncated={},
                done={},
                obs_next={},
                info={},
                policy={},
            )
            self.data = cast(RolloutBatchProtocol, data)
            self.reset_env()

        return CollectStatsWithInfo(
            n_collected_episodes=episode_count,
            n_collected_steps=step_count,
            collect_time=collect_time,
            collect_speed=step_count / collect_time,
            returns=np.array(episode_returns),
            returns_stat=SequenceSummaryStats.from_sequence(episode_returns)
            if len(episode_returns) > 0
            else None,
            lens=np.array(episode_lens, int),
            lens_stat=SequenceSummaryStats.from_sequence(episode_lens)
            if len(episode_lens) > 0
            else None,
            info=DictOfSequenceSummaryStats.from_dict(episode_info)
        )
