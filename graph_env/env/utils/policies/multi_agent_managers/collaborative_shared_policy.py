from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from tianshou.data import Batch, ReplayBuffer, to_torch_as
from tianshou.policy import BasePolicy
from .shared_policy import MultiAgentSharedPolicy
try:
    from tianshou.env.pettingzoo_env import PettingZooEnv
except ImportError:
    PettingZooEnv = None  # type: ignore


class MultiAgentCollaborativeSharedPolicy(MultiAgentSharedPolicy):
    """Multi-agent shared policy manager for MARL.

    This multi-agent policy manager accepts a
    :class:`~tianshou.policy.BasePolicy`. It dispatches the batch data to the
    policy when the "forward" is called. The same as "process_fn"
    and "learn": it splits the data and feeds them to the policy.
    """

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
        has_rnn_mask = isinstance(buffer.obs.mask, np.ndarray) and len(batch.obs.mask.shape) == 3
        if has_rew:  # save the original reward in save_rew
            # Since we do not override buffer.__setattr__, here we use _meta to
            # change buffer.rew, otherwise buffer.rew = Batch() has no effect.
            save_rew, buffer._meta.rew = buffer.rew, Batch()
        for agent in self.agents:
            # Need to be unique in case of stacked obs
            agent_index = np.unique(np.nonzero(batch.obs.agent_id == agent)[0])
            if len(agent_index) == 0:
                results[agent] = Batch()
                continue
            tmp_batch, tmp_indice = batch[agent_index], indice[agent_index]
            if has_rew:
                tmp_batch.rew_list = tmp_batch.rew
                if has_rnn_mask:
                    # Indices can be -1 when agents appear dynamically
                    tmp_batch.info.indices = tmp_batch.info.indices[:,0,:]

                valid_indices = np.where(tmp_batch.info.indices >= 0)
                tmp_batch.active_obs = buffer.get(index=tmp_batch.info.indices[valid_indices], key='obs')
                tmp_batch.active_obs.act = buffer.get(index=tmp_batch.info.indices[valid_indices], key='act', stack_num=1)
                tmp_batch.active_obs.rew = save_rew[tmp_batch.info.indices[valid_indices], tmp_batch.active_obs.agent_id.astype(int)[:,0]]
                tmp_batch.active_obs.index = tmp_batch.info.indices[valid_indices]
                tmp_batch.active_obs.info = tmp_batch.info[valid_indices]

                tmp_batch.rew = tmp_batch.rew[:, self.agent_idx[agent]]
                buffer._meta.rew = save_rew[:, self.agent_idx[agent]]

                tmp_batch.active_obs = self.policy.process_fn(tmp_batch.active_obs, buffer, tmp_batch.info.indices[valid_indices])
            if not hasattr(tmp_batch.obs, "mask"):
                if hasattr(tmp_batch.obs, 'obs'):
                    tmp_batch.obs = tmp_batch.obs.obs
                if hasattr(tmp_batch.obs_next, 'obs'):
                    tmp_batch.obs_next = tmp_batch.obs_next.obs
            if has_rnn_mask:  # We need to handle masks in form [buffer_size, stack]
                # at the moment masking is disabled
                save_mask = buffer.obs.pop('mask')
                results[agent] = self.policy.process_fn(tmp_batch, buffer,
                                                        tmp_indice)
                buffer.obs.mask = save_mask
            else:
                results[agent] = self.policy.process_fn(tmp_batch, buffer,
                                                        tmp_indice)
        if has_rew:  # restore from save_rew
            buffer._meta.rew = save_rew
        return Batch(results)
