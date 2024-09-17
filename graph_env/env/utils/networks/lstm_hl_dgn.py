from typing import Optional, Tuple, Dict, Any
from tianshou.data import Batch

import torch
import torch.nn as nn
import torch.nn.functional as F
from tianshou.utils.net.common import MLP
from torch_geometric.nn import GATv2Conv, GAT
from torch_geometric.nn import global_mean_pool, global_add_pool, \
    global_max_pool

from torch_geometric.data.data import Data
from torch_geometric.data.batch import Batch as PyGeomBatch


def to_pytorch_geometric_batch(obs, device):
    """
    In the evaluation mode, `obs` should be with shape ``[bsz, dim]``; in the
    training mode, `obs` should be with shape ``[bsz, len, dim]``. See the code
    and comment for more detail.
    """

    # obs [bsz, len, dim] (training) or [bsz, dim] (evaluation)
    # In short, the tensor's shape in training phase is longer than which
    # in evaluation phase.
    if len(obs.shape) == 2:
        obs = Batch.stack((obs, Batch()), axis=-2)

    observations = [
        Data(
            x=torch.as_tensor(
                observation[2],
                device=device,
                dtype=torch.float32
            ),
            edge_index=torch.as_tensor(
                observation[0],
                device=device,
                dtype=torch.int
            ),
            index=observation[3][0],
            batch_index=[batch_index]
        ) for sub_observation, batch_index
        in zip(obs.observation, range(len(obs.observation)))
        for observation in sub_observation
    ]

    return PyGeomBatch.from_data_list(observations)


class RecurrentHLDGNNetwork(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            num_heads,
            features_only=False,
            dueling_param: Optional[
                Tuple[Dict[str, Any], Dict[str, Any]]] = None,
            device='cpu',
            aggregator_function=global_max_pool
    ):
        super(RecurrentHLDGNNetwork, self).__init__()
        self.aggregator_function = aggregator_function
        self.device = device
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.final_latent_dim = hidden_dim * num_heads
        self.use_dueling = dueling_param is not None
        self.conv1 = GATv2Conv(
            input_dim,
            hidden_dim,
            num_heads
        )
        self.lstm = nn.LSTM(
            input_size=self.final_latent_dim,
            hidden_size=self.final_latent_dim,
            num_layers=1,
            batch_first=True,
        )

        if self.use_dueling:
            q_kwargs, v_kwargs = dueling_param
            q_output_dim, v_output_dim = self.output_dim, 1

            q_kwargs: Dict[str, Any] = {
                **q_kwargs,
                "input_dim": self.final_latent_dim,
                "output_dim": q_output_dim,
                "device": self.device
            }
            v_kwargs: Dict[str, Any] = {
                **v_kwargs,
                "input_dim": self.final_latent_dim,
                "output_dim": v_output_dim,
                "device": self.device
            }
            self.Q, self.V = MLP(**q_kwargs), MLP(**v_kwargs)

    def forward(self, obs, state=None, info={}):
        bsz = obs.shape[0]
        horizon = obs.shape[1] if len(obs.shape) == 3 else 1

        obs = to_pytorch_geometric_batch(obs, self.device)
        x, edge_index = obs.x, obs.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.aggregator_function(x, batch=obs.batch)
        x = x.reshape(bsz, horizon, self.final_latent_dim)
        self.lstm.flatten_parameters()
        if state is None or state.is_empty():
            x, (hidden, cell) = self.lstm(x)
        else:
            x, (hidden, cell) = self.lstm(
                x, (
                    state["hidden"].transpose(0, 1),
                    state["cell"].transpose(0, 1)
                )
            )
        q, v = self.Q(x[:, -1]), self.V(x[:, -1])
        x = q - q.mean(dim=1, keepdim=True) + v
        return x, {
            "hidden": hidden.transpose(0, 1).detach(),
            "cell": cell.transpose(0, 1).detach()
        }
