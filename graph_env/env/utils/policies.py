from typing import Optional, Union, Any, Dict

import numpy as np
from tianshou.policy import DQNPolicy
from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as
import torch

class CTDEDQN(DQNPolicy):
    def __init__(
            self,
            model: torch.nn.Module,
            optim: torch.optim.Optimizer,
            discount_factor: float = 0.99,
            estimation_step: int = 1,
            target_update_freq: int = 0,
            reward_normalization: bool = False,
            is_double: bool = True,
            clip_loss_grad: bool = False,
            env=None,
            **kwargs: Any,
    ) -> None:
        self.agent_idx = env.agent_idx
        self.agents = env.agents

        super().__init__(model,
                         optim,
                         discount_factor,
                         estimation_step,
                         target_update_freq,
                         reward_normalization,
                         is_double,
                         clip_loss_grad,
                         **kwargs)

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "model",
        input: str = "obs",
        **kwargs: Any,
    ):
        new_batch = Batch()
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
            new_batch = Batch.stack((new_batch, tmp_batch))

        result = super().forward(
            batch=new_batch[0],
            state=None,
            **kwargs
        )
        return result

    def learn(self, batch: Batch, **kwargs: Any):
        new_batch = Batch()
        for agent_id in self.agents:
            data = batch[agent_id]
            if not data.is_empty():
                new_batch = Batch.stack((new_batch, data))

        result = super().learn(new_batch[0], **kwargs)
        return result
