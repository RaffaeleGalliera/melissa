from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from tianshou.data import Batch, ReplayBuffer, to_torch_as
from tianshou.policy import BasePolicy, DQNPolicy
try:
    from tianshou.env.pettingzoo_env import PettingZooEnv
except ImportError:
    PettingZooEnv = None  # type: ignore


class VDNPolicy(DQNPolicy):
    """VDN policy.
     https://arxiv.org/abs/1706.02275 restricted to one-hop neighbours only"""

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self._target and self._iter % self._freq == 0:
            self.sync_weight()
        self.optim.zero_grad()
        weight = batch.pop("weight", 1.0)
        batch_q = torch.zeros((len(batch), ), device='cuda')
        batch_returns = torch.zeros((len(batch), ), device='cuda')

        for i, exp in enumerate(batch):
            # Get ID indices of active obs in the whole experiment
            valid_indices = np.where(exp.info.indices >= 0)

            # Get active neighbour obs from batch filtering by obs index
            neighbour_obs = batch.active_obs[np.where(np.isin(batch.active_obs.index, exp.info.indices[valid_indices]))[0]]
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
