from typing import Any, Dict
from dataclasses import dataclass
import numpy as np
import torch

from tianshou.data import Batch, to_torch_as
from tianshou.policy import DQNPolicy
try:
    from tianshou.env.pettingzoo_env import PettingZooEnv
except ImportError:
    PettingZooEnv = None  # type: ignore
from tianshou.policy.base import TrainingStats


@dataclass(kw_only=True)
class DGNTrainingStats(TrainingStats):
    loss: float


class DGNPolicy(DQNPolicy):

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self._target and self._iter % self.freq == 0:
            self.sync_weight()
        self.optim.zero_grad()
        weight = batch.pop("weight", 1.0)
        device = 'cuda' if torch.cuda.is_available() else None
        batch_q = torch.zeros((len(batch),), device=device)
        batch_returns = torch.zeros((len(batch),), device=device)

        for i, exp in enumerate(batch):
            # Get ID indices of batch active obs and intersect with valid neighbours
            if len(exp.obs.obs.shape) == 2:
                obs_stacked_neighbors = np.unique(np.concatenate(exp.obs.obs.observation[:, 6]))
                stacked_neighbors_indices = np.append(obs_stacked_neighbors, int(exp.obs.agent_id[0]))
                active_neighbors = np.intersect1d(
                    np.where(exp.info.indices >= 0),
                    stacked_neighbors_indices
                ).astype(int)
            else:
                indices = np.append(exp.obs.obs.observation[6], int(exp.obs.agent_id))
                active_neighbors = np.intersect1d(
                    np.where(exp.info.indices >= 0),
                    indices
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
            sum_q = q.sum()
            sum_returns = returns.sum()
            batch_q[i] = sum_q
            batch_returns[i] = sum_returns

        td_error = batch_returns - batch_q
        loss = (td_error.pow(2) * weight).mean()

        batch.weight = td_error # prio-buffer
        loss.backward()
        self.optim.step()
        self._iter += 1
        return DGNTrainingStats(loss=loss.item())  # type: ignore[return-value]
