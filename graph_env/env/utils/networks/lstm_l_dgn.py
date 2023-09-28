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

    observations = [Data(
            x=torch.as_tensor(observation[5], device=device, dtype=torch.float32),
            edge_index=torch.as_tensor(observation[4], device=device,
                                       dtype=torch.int),
            index=observation[3][0],
            batch_index=[batch_index]
        )
        for sub_observation, batch_index in zip(obs.observation, range(len(obs.observation))) for observation in
        sub_observation]
    return PyGeomBatch.from_data_list(observations)


class RecurrentLDGNNetwork(nn.Module):
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
        super(RecurrentLDGNNetwork, self).__init__()
        self.aggregator_function = aggregator_function
        self.device = device
        self.output_dim = hidden_dim * num_heads
        self.hidden_dim = hidden_dim
        self.use_dueling = dueling_param is not None
        output_dim = output_dim if not self.use_dueling else 0
        self.encoder = MLP(input_dim=input_dim, hidden_sizes=[512],
                           output_dim=hidden_dim, device=self.device)
        self.conv1 = GATv2Conv(hidden_dim, hidden_dim, num_heads,
                               device=self.device)
        self.conv2 = GATv2Conv(hidden_dim * num_heads, hidden_dim, num_heads,
                               device=self.device)
        self.lstm = nn.LSTM(
            input_size=hidden_dim + hidden_dim * num_heads * 2,
            hidden_size=hidden_dim + hidden_dim * num_heads * 2,
            num_layers=1,
            batch_first=True,
        )

        if self.use_dueling:
            q_kwargs, v_kwargs = dueling_param
            q_output_dim, v_output_dim = 2, 1

            q_kwargs: Dict[str, Any] = {
                **q_kwargs,
                "input_dim": hidden_dim + hidden_dim * num_heads * 2,
                "output_dim": q_output_dim,
                "device": self.device
            }
            v_kwargs: Dict[str, Any] = {
                **v_kwargs,
                "input_dim": hidden_dim + hidden_dim * num_heads * 2,
                "output_dim": v_output_dim,
                "device": self.device
            }
            self.Q, self.V = MLP(**q_kwargs), MLP(**v_kwargs)
            self.output_dim = self.Q.output_dim

    def forward(self, obs, state=None, info={}):
        bsz = obs.shape[0]
        horizon = obs.shape[1] if len(obs.shape) == 3 else 1

        obs = to_pytorch_geometric_batch(obs, self.device)
        indices = [range[0][index[0]] for range, index in
                   zip([torch.where(obs.batch == value) for value in
                        torch.unique(obs.batch)], obs.index)]
        x, edge_index = obs.x, obs.edge_index
        x = self.encoder(x)
        x = F.relu(x)
        x_1 = x[indices, :]
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x_2 = x[indices, :]
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x_3 = x[indices, :]
        # I need (bsz, 4, 7)
        # Regroup the data into (bsz, 4, 7)
        x = torch.cat([x_1, x_2, x_3], dim=1)
        x = x.reshape(bsz, horizon, 1152)
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
