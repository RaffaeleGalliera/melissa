from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from tianshou.data import Batch, ReplayBuffer, to_torch_as
from tianshou.policy import BasePolicy, DQNPolicy
try:
    from tianshou.env.pettingzoo_env import PettingZooEnv
except ImportError:
    PettingZooEnv = None  # type: ignore


class DGNPolicy(DQNPolicy):

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self._target and self._iter % self._freq == 0:
            self.sync_weight()
        self.optim.zero_grad()
        weight = batch.pop("weight", 1.0)
        device = 'cuda' if torch.cuda.is_available() else None
        partial_loss = torch.zeros((len(batch),), device=device)
        td_error = torch.zeros((len(batch),), device=device)

        for i, exp in enumerate(batch):
            # Get ID indices of batch active obs and intersect with valid neighbours
            active_neighbors = np.intersect1d(
                np.where(exp.info.indices >= 0),
                np.append(exp.obs.obs.observation[6], int(exp.obs.agent_id))
            ).astype(int)
            valid_indices = exp.info.indices[active_neighbors]

            # Get active neighbour obs from batch filtering by obs index
            neighbour_obs = batch.active_obs[
                [
                    np.where(batch.active_obs.index == index)[0][0]
                    for index in valid_indices
                ]
            ]

            q = self(neighbour_obs).logits
            q = q[np.arange(len(q)), neighbour_obs.act]

            returns = to_torch_as(neighbour_obs.returns.flatten(), q)
            partial_loss[i] = ((returns - q).pow(2)).mean()
            td_error[i] = (returns - q).mean()

        loss = (partial_loss * weight).mean()

        batch.weight = td_error # prio-buffer
        loss.backward()
        self.optim.step()
        self._iter += 1
        return {"loss": loss.item()}
