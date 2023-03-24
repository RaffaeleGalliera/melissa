from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tianshou.utils.net.common import MLP
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import global_mean_pool


class GATNetwork(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 num_heads,
                 features_only=False,
                 dueling_param: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
                 device='cpu'):
        super(GATNetwork, self).__init__()
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
        logits = []
        for observation in obs.observation:
            # TODO: Need here we move tensors to CUDA, cannot just put it in Batch because of data time -> slows down
            x = torch.as_tensor(observation[2], device=self.device, dtype=torch.float32)
            edge_index = torch.as_tensor(observation[0], device=self.device, dtype=torch.int)
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = x.view(x.size(0), -1)
            x = global_mean_pool(x, None)
            q, v = self.Q(x), self.V(x)
            x = q - q.mean(dim=1, keepdim=True) + v
            logits.append(x.flatten())
        logits = torch.stack(logits)
        return logits, state

