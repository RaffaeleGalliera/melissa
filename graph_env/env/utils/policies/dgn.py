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
            # Need to handle graph stacking case
            if len(exp.info.indices.shape) == 2:
                indices = exp.info.indices[0]
            else:
                indices = exp.info.indices

            # Get ID indices of active obs in the whole experiment
            valid_indices = np.where(indices >= 0)

            # Get active obs from batch filtering by obs index
            active_obs = Batch(
                [
                    batch.active_obs[
                        np.where(batch.active_obs.index == index)[0][0]
                    ] for index in indices[valid_indices]
                ]
            )
            q = self(active_obs).logits
            q = q[np.arange(len(q)), active_obs.act]

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
