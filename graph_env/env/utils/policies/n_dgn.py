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
        batch_q = torch.zeros((len(batch), ), device=device)
        batch_returns = torch.zeros((len(batch), ), device=device)

        for i, exp in enumerate(batch):
            # Get ID indices of batch active obs and intersect with valid neighbours
            if len(exp.obs.obs.shape) == 2:
                # TODO implement in case shape is larger
                raise NotImplementedError("Need to implement when shape is > 2")
            else:
                indices = exp.info['active_one_hop_neighbors']
                indices[int(exp.obs.agent_id)] = True
                active_neighbors = np.bitwise_and(
                    exp.info.indices >= 0,
                    indices
                )

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

            sum_q = q.sum()
            returns = exp.returns
            batch_q[i] = sum_q
            batch_returns[i] = returns

        td_error = batch_returns - batch_q

        if self.clip_loss_grad:
            y = batch_q.reshape(-1, 1)
            t = batch_returns.reshape(-1, 1)
            loss = torch.nn.functional.huber_loss(y, t, reduction="mean")
        else:
            loss = (td_error.pow(2) * weight).mean()

        batch.weight = td_error  # prio-buffer
        loss.backward()
        self.optim.step()
        self._iter += 1
        return DGNTrainingStats(loss=loss.item())  # type: ignore[return-value]
