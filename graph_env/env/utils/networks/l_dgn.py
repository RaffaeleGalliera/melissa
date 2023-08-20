from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tianshou.utils.net.common import MLP
from torch_geometric.nn import GATv2Conv, GAT
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool

from torch_geometric.data.data import Data
from torch_geometric.data.batch import Batch as PyGeomBatch


def to_pytorch_geometric_batch(obs, device):
    observations = [Data(x=torch.as_tensor(observation[5],
                                           device=device,
                                           dtype=torch.float32),
                         edge_index=torch.as_tensor(observation[4],
                                                    device=device,
                                                    dtype=torch.int),
                         index=observation[3][0]) for observation in obs.observation]
    return PyGeomBatch.from_data_list(observations)


class LDGNNetwork(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 num_heads,
                 features_only=False,
                 dueling_param: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
                 device='cpu',
                 aggregator_function=global_max_pool):
        super(LDGNNetwork, self).__init__()
        self.aggregator_function = aggregator_function
        self.device = device
        self.output_dim = hidden_dim * num_heads
        self.hidden_dim = hidden_dim
        self.use_dueling = dueling_param is not None
        output_dim = output_dim if not self.use_dueling else 0
        self.encoder = MLP(input_dim=input_dim, hidden_sizes=[512], output_dim=hidden_dim, device=self.device)
        self.conv1 = GATv2Conv(hidden_dim, hidden_dim, num_heads, device=self.device)
        self.conv2 = GATv2Conv(hidden_dim * num_heads, hidden_dim, num_heads, device=self.device)

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
        obs = to_pytorch_geometric_batch(obs, self.device)
        indices = [range[0][index[0]] for range, index in zip([torch.where(obs.batch==value) for value in torch.unique(obs.batch)], obs.index)]
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
        x = torch.cat([x_1, x_2, x_3], dim=1)
        q, v = self.Q(x), self.V(x)
        x = q - q.mean(dim=1, keepdim=True) + v
        return x, state
