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
            # Get ID indice np.where(batch.active_obs.index == batch.active_obs.index[1])s of active obs in the whole experiment
            # TODO fix active obs should contain already only active obs thanks to collab shared policy
            valid_indices = exp.info.indices[np.where(exp.info.indices >= 0)].astype(int)

            # Get active active obs from batch filtering by obs index
            active_obs = batch.active_obs[
                [
                    np.where(batch.active_obs.index == index)[0][0]
                    for index in valid_indices
                ]
            ]

            q = self(active_obs).logits
            q = q[np.arange(len(q)), active_obs.act]

            returns = to_torch_as(active_obs.returns.flatten(), q)
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
