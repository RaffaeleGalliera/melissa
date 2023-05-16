from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tianshou.utils.net.common import MLP
from torch_geometric.nn import GATv2Conv, GAT
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool

from torch_geometric.data.data import Data
from torch_geometric.data.batch import Batch as PyGeomBatch


def to_pytorch_geometric_batch(obs, device, is_critic):
    if is_critic:
        observations = [Data(x=torch.as_tensor(observation[5],
                                               device=device,
                                               dtype=torch.float32),
                             edge_index=torch.as_tensor(observation[4],
                                                        device=device,
                                                        dtype=torch.int)) for observation in obs.obs.observation]
    else:
        observations = [Data(x=torch.as_tensor(observation[2],
                                               device=device,
                                               dtype=torch.float32),
                             edge_index=torch.as_tensor(observation[0],
                                                        device=device,
                                                        dtype=torch.int)) for observation in obs.obs.observation]
    return PyGeomBatch.from_data_list(observations)


class GATNetwork(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 num_heads,
                 is_critic=False,
                 features_only=False,
                 dueling_param: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
                 device='cpu'):
        super(GATNetwork, self).__init__()
        self.device = device
        self.output_dim = hidden_dim * num_heads
        self.hidden_dim = hidden_dim
        self.use_dueling = dueling_param is not None
        self.is_critic = is_critic
        self.conv1 = GATv2Conv(input_dim, hidden_dim, num_heads) if self.is_critic else GATv2Conv(input_dim - 1, hidden_dim, num_heads)
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
            self.A, self.V = MLP(**q_kwargs), MLP(**v_kwargs)
            self.output_dim = self.A.output_dim

    def forward(self, obs, state=None, info={}):
        obs = to_pytorch_geometric_batch(obs, self.device, self.is_critic)

        x, edge_index = obs.x, obs.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = global_max_pool(x, batch=obs.batch)
        if self.use_dueling:
            a, v = self.A(x), self.V(x)
            x = a + v - a.mean(dim=1, keepdim=True)
        return x, state
