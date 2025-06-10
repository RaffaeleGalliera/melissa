import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from tianshou.utils.net.common import MLP
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_add_pool, global_max_pool

from graph_env.env.utils.constants import RADIUS_OF_INFLUENCE
from graph_env.env.utils.networks.common import build_pyg_batch_time


class HLDGNNetwork(nn.Module):
    """
    Hierarchical LDGN: single GAT layer + global pooling, with masking.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_heads: int,
        agents_num: int,
        aggregator: str = 'mean',
        dueling_param: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
        device: str = 'cpu',
        edge_attributes: bool = False,
    ):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.agents_num = agents_num
        self.edge_attributes = edge_attributes

        # Encoder maps raw node features (minus mask) to hidden_dim
        self.encoder = MLP(
            input_dim=input_dim - 1,
            hidden_sizes=[hidden_dim],
            output_dim=hidden_dim,
            device=self.device,
        )

        # Single GAT layer
        self.conv1 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=num_heads,
        )

        # Choose pooling function
        self.aggregator = {
            'mean': global_mean_pool,
            'add': global_add_pool,
            'max': global_max_pool,
        }[aggregator]

        # Dueling / linear head
        self.use_dueling = dueling_param is not None
        in_head_dim = hidden_dim * num_heads
        if self.use_dueling:
            q_kwargs, v_kwargs = dueling_param
            q_kwargs.update({
                'input_dim': in_head_dim,
                'output_dim': output_dim,
                'device': self.device,
            })
            v_kwargs.update({
                'input_dim': in_head_dim,
                'output_dim': 1,
                'device': self.device,
            })
            self.Q = MLP(**q_kwargs)
            self.V = MLP(**v_kwargs)
        else:
            self.out_linear = nn.Linear(in_head_dim, output_dim)

    def forward(self, obs, state=None, info={}):
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, device=self.device)

        # Build PyG batch and extract mask
        data, agent_indices, bs_size = build_pyg_batch_time(
            obs,
            radius=RADIUS_OF_INFLUENCE,
            device=self.device,
            input_dim=self.input_dim,
            edge_attributes=self.edge_attributes,
            agents_num=self.agents_num,
        )

        # 1) Encode
        x = self.encoder(data.x)   # (bs*N, hidden_dim)
        x = F.relu(x)

        # 2) GAT layer
        x = self.conv1(x, data.edge_index)
        x = F.relu(x)

        # 3) Mask out non-decision-makers
        x = x * data.dm_mask

        # 4) Global pooling across the graph
        x_pooled = self.aggregator(x, data.batch)  # (bs, hidden_dim*num_heads)

        # 5) Head
        if self.use_dueling:
            q = self.Q(x_pooled)
            v = self.V(x_pooled)
            q_mean = q.mean(dim=1, keepdim=True)
            out = q - q_mean + v
        else:
            out = self.out_linear(x_pooled)

        return out, state
