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
    observations = [Data(x=torch.as_tensor(observation[2],
                                           device=device,
                                           dtype=torch.float32),
                         edge_index=torch.as_tensor(observation[0],
                                                    device=device,
                                                    dtype=torch.int)) for observation in obs.observation]
    return PyGeomBatch.from_data_list(observations)


class GATNetwork(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 num_heads,
                 features_only=False,
                 dueling_param: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
                 device='cpu',
                 aggregator_function="global_max_pool"):
        super(GATNetwork, self).__init__()
        self.aggregator_function = aggregator_function
        self.device = device
        self.output_dim = hidden_dim * num_heads
        self.hidden_dim = hidden_dim
        self.use_dueling = dueling_param is not None
        output_dim = output_dim if not self.use_dueling else 0
        self.conv1 = GATv2Conv(input_dim, hidden_dim, num_heads)
        if self.use_dueling:
            q_kwargs, v_kwargs = dueling_param
            q_output_dim, v_output_dim = 2, 1

            q_kwargs: Dict[str, Any] = {
                **q_kwargs,
                "input_dim": self.output_dim,
                "output_dim": q_output_dim,
                "device": self.device
            }
            v_kwargs: Dict[str, Any] = {
                **v_kwargs,
                "input_dim": self.output_dim,
                "output_dim": v_output_dim,
                "device": self.device
            }
            self.Q, self.V = MLP(**q_kwargs), MLP(**v_kwargs)
            self.output_dim = self.Q.output_dim

    def forward(self, obs, state=None, info={}):
        obs = to_pytorch_geometric_batch(obs, self.device)
        x, edge_index = obs.x, obs.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.aggregate(x, batch=obs.batch)
        q, v = self.Q(x), self.V(x)
        x = q - q.mean(dim=1, keepdim=True) + v
        return x, state

    def aggregate(self, x, batch):
        new_x = x
        if self.aggregator_function == "global_max_pool":
            new_x = global_max_pool(x, batch=batch)
        elif self.aggregator_function == "global_mean_pool":
            new_x = global_mean_pool(x, batch=batch)
        elif self.aggregator_function == "global_add_pool":
            new_x = global_add_pool(x, batch=batch)
        return new_x
