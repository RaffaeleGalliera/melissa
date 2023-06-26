from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from tianshou.data import Batch, ReplayBuffer, to_torch_as
from tianshou.policy import BasePolicy, DQNPolicy
try:
    from tianshou.env.pettingzoo_env import PettingZooEnv
except ImportError:
    PettingZooEnv = None  # type: ignore


class MultiAgentSharedPolicy(BasePolicy):
    """Multi-agent shared policy manager for MARL.

    This multi-agent policy manager accepts a
    :class:`~tianshou.policy.BasePolicy`. It dispatches the batch data to the
    policy when the "forward" is called. The same as "process_fn"
    and "learn": it splits the data and feeds them to the policy.
    """

    def __init__(
        self, policy: BasePolicy, env: PettingZooEnv, **kwargs: Any
    ) -> None:
        super().__init__(action_space=env.action_space, **kwargs)

        self.agent_idx = env.agent_idx
        self.agents = env.agents
        self.policy = policy

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray
    ) -> Batch:
        """Dispatch batch data from obs.agent_id to the policy's process_fn.

        Save original multi-dimensional rew in "save_rew", set rew to the
        reward of each agent during their "process_fn", and restore the
        original reward afterwards.
        """
        results = {}
        # reward can be empty Batch (after initial reset) or nparray.
        has_rew = isinstance(buffer.rew, np.ndarray)
        if has_rew:  # save the original reward in save_rew
            # Since we do not override buffer.__setattr__, here we use _meta to
            # change buffer.rew, otherwise buffer.rew = Batch() has no effect.
            save_rew, buffer._meta.rew = buffer.rew, Batch()
        for agent in self.agents:
            agent_index = np.nonzero(batch.obs.agent_id == agent)[0]
            if len(agent_index) == 0:
                results[agent] = Batch()
                continue
            tmp_batch, tmp_indice = batch[agent_index], indice[agent_index]
            if has_rew:
                tmp_batch['rew_list'] = tmp_batch.rew
                valid_indices = np.where(tmp_batch.info.indices >= 0)

                tmp_batch['active_obs'] = buffer._meta[tmp_batch.info.indices][valid_indices]
                tmp_batch['active_obs'].rew = save_rew[tmp_batch.info.indices[valid_indices], tmp_batch['active_obs'].obs.agent_id.astype(int)]
                tmp_batch['active_obs'].index = tmp_batch.info.indices[valid_indices]

                tmp_batch.rew = tmp_batch.rew[:, self.agent_idx[agent]]
                buffer._meta.rew = save_rew[:, self.agent_idx[agent]]
                tmp_batch['active_obs'] = self.policy.process_fn(tmp_batch['active_obs'], buffer, tmp_batch.info.indices[valid_indices])
            if not hasattr(tmp_batch.obs, "mask"):
                if hasattr(tmp_batch.obs, 'obs'):
                    tmp_batch.obs = tmp_batch.obs.obs
                if hasattr(tmp_batch.obs_next, 'obs'):
                    tmp_batch.obs_next = tmp_batch.obs_next.obs
            results[agent] = self.policy.process_fn(tmp_batch, buffer, tmp_indice)
        if has_rew:  # restore from save_rew
            buffer._meta.rew = save_rew
        return Batch(results)

    def exploration_noise(self, act: Union[np.ndarray, Batch],
                          batch: Batch) -> Union[np.ndarray, Batch]:
        """Add exploration noise from sub-policy onto act."""
        for agent_id in self.agents:
            agent_index = np.nonzero(batch.obs.agent_id == agent_id)[0]
            if len(agent_index) == 0:
                continue
            act[agent_index] = self.policy.exploration_noise(
                act[agent_index], batch[agent_index]
            )
        return act

    def forward(  # type: ignore
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch]] = None,
        **kwargs: Any,
    ) -> Batch:
        """Dispatch batch data from obs.agent_id to the policy's forward.

        :param state: if None, it means all agents have no state. If not
            None, it should contain keys of "agent_1", "agent_2", ...

        :return: a Batch with the following contents:

        ::

            {
                "act": actions corresponding to the input
                "state": {
                    "agent_1": output state of agent_1's policy for the state
                    "agent_2": xxx
                    ...
                    "agent_n": xxx}
                "out": {
                    "agent_1": output of agent_1's policy for the input
                    "agent_2": xxx
                    ...
                    "agent_n": xxx}
            }
        """
        results: List[Tuple[bool, np.ndarray, Batch, Union[np.ndarray, Batch],
                            Batch]] = []

        for agent_id in self.agents:
            # This part of code is difficult to understand.
            # Let's follow an example with two agents
            # batch.obs.agent_id is [1, 2, 1, 2, 1, 2] (with batch_size == 6)
            # each agent plays for three transitions
            # agent_index for agent 1 is [0, 2, 4]
            # agent_index for agent 2 is [1, 3, 5]
            # we separate the transition of each agent according to agent_id
            agent_index = np.nonzero(batch.obs.agent_id == agent_id)[0]
            if len(agent_index) == 0:
                # (has_data, agent_index, out, act, state)
                results.append((False, np.array([-1]), Batch(), Batch(), Batch()))
                continue
            tmp_batch = batch[agent_index]
            if isinstance(tmp_batch.rew, np.ndarray):
                # reward can be empty Batch (after initial reset) or nparray.
                tmp_batch.rew = tmp_batch.rew[:, self.agent_idx[agent_id]]
            if not hasattr(tmp_batch.obs, "mask"):
                if hasattr(tmp_batch.obs, 'obs'):
                    tmp_batch.obs = tmp_batch.obs.obs
                if hasattr(tmp_batch.obs_next, 'obs'):
                    tmp_batch.obs_next = tmp_batch.obs_next.obs
            out = self.policy(
                batch=tmp_batch,
                state=None if state is None else state[agent_id],
                **kwargs
            )
            act = out.act
            each_state = out.state \
                if (hasattr(out, "state") and out.state is not None) \
                else Batch()
            results.append((True, agent_index, out, act, each_state))
        holder = Batch.cat(
            [
                {
                    "act": act
                } for (has_data, agent_index, out, act, each_state) in results
                if has_data
            ]
        )
        state_dict, out_dict = {}, {}
        for agent_id, (has_data, agent_index, out, act, state) in zip(self.agents, results):
            if has_data:
                holder.act[agent_index] = act
            state_dict[agent_id] = state
            out_dict[agent_id] = out
        holder["out"] = out_dict
        holder["state"] = state_dict
        return holder

    def learn(self, batch: Batch,
              **kwargs: Any) -> Dict[str, Union[float, List[float]]]:
        """Dispatch the data to the policy for learning.

        :return: policy loss
        """

        return self.policy.learn(Batch.cat([batch[agent_id] for agent_id in self.agents if not batch[agent_id].is_empty()]))


class NVDNPolicy(DQNPolicy):
    """VDN policy.  https://arxiv.org/abs/1706.02275 restristed to one-hop neighbours only"""

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self._target and self._iter % self._freq == 0:
            self.sync_weight()
        self.optim.zero_grad()
        weight = batch.pop("weight", 1.0)
        batch_q = torch.zeros((len(batch), ), device='cuda')
        batch_returns = torch.zeros((len(batch), ), device='cuda')

        for exp, i in zip(batch, range(len(batch))):
            # Get indices from valid local graph ids only (1-hop neighbours and self)
            valid_indices = np.intersect1d(np.where(exp.info.indices >= 0), exp.obs.obs.observation[1]).astype(int)
            neighbour_obs = batch.active_obs[np.where(np.isin(exp.info.indices[valid_indices], exp.info.indices))[0]]
            q = self(neighbour_obs).logits
            q = q[np.arange(len(q)), neighbour_obs.act]

            returns = to_torch_as(neighbour_obs.rew.flatten(), q)
            sum_q = q.sum()
            sum_returns = returns.sum()
            batch_q[i] = sum_q
            batch_returns[i] = sum_returns

        td_error = batch_returns - batch_q

        if self._clip_loss_grad:
            y = batch_q.reshape(-1, 1)
            t = batch_returns.reshape(-1, 1)
            loss = torch.nn.functional.huber_loss(y, t, reduction="mean")
        else:
            loss = (td_error.pow(2) * weight).mean()

        batch.weight = td_error  # prio-buffer
        loss.backward()
        self.optim.step()
        self._iter += 1
        return {"loss": loss.item()}
