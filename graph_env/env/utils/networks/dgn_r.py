import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from tianshou.utils.net.common import MLP
from torch_geometric.nn import TransformerConv

from graph_env.env.utils.constants import RADIUS_OF_INFLUENCE
from graph_env.env.utils.networks.common import build_pyg_batch_time

class DGNRNetwork(nn.Module):
    """
    Recurrent DGN (TransformerConv-based) with decision-maker masking.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_heads: int,
        agents_num: int,
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
            input_dim=input_dim,
            hidden_sizes=[hidden_dim],
            output_dim=hidden_dim,
            device=self.device,
        )

        # Two TransformerConv layers
        self.conv1 = TransformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=num_heads,
            root_weight=False,
        )
        self.conv2 = TransformerConv(
            in_channels=hidden_dim * num_heads,
            out_channels=hidden_dim,
            heads=num_heads,
            root_weight=False,
        )

        # Dueling head configuration
        self.use_dueling = dueling_param is not None
        # We will snapshot three embeddings: x1 (post-encoder), x2 (post-conv1), x3 (post-conv2)
        self.final_latent_dim = hidden_dim + hidden_dim * num_heads * 2

        if self.use_dueling:
            q_kwargs, v_kwargs = dueling_param
            q_kwargs.update({
                'input_dim': self.final_latent_dim,
                'output_dim': output_dim,
                'device': self.device,
            })
            v_kwargs.update({
                'input_dim': self.final_latent_dim,
                'output_dim': 1,
                'device': self.device,
            })
            self.Q = MLP(**q_kwargs)
            self.V = MLP(**v_kwargs)
        else:
            self.out_linear = nn.Linear(self.final_latent_dim, output_dim)

    def forward(self, obs, state=None, info={}):
        # obs: (bs, N*node_dim+1)
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, device=self.device)

        # Build graph batch, extract node features and decision mask
        data, agent_indices, bs_size = build_pyg_batch_time(
            obs,
            radius=RADIUS_OF_INFLUENCE,
            device=self.device,
            input_dim=self.input_dim,
            edge_attributes=self.edge_attributes,
            agents_num=self.agents_num,
        )

        # 1) Encode raw features
        x = self.encoder(data.x)      # (bs*N, hidden_dim)
        x = F.relu(x)
        # global_indices: location of each controlling agent's embedding in flat tensor
        global_indices = data.ptr[:-1] + agent_indices
        x1 = x[global_indices, :]

        # 2) First GNN layer
        x = self.conv1(x, data.edge_index)
        x = F.relu(x)
        x2 = x[global_indices, :]

        # 3) Mask out non-decision-makers before second layer
        x = x * data.dm_mask

        # 4) Second GNN layer
        x = self.conv2(x, data.edge_index)
        x = F.relu(x)
        x3 = x[global_indices, :]

        # 5) Concatenate snapshots
        x_cat = torch.cat([x1, x2, x3], dim=1)  # (bs, final_latent_dim)

        # 6) Dueling or linear head
        if self.use_dueling:
            q = self.Q(x_cat)
            v = self.V(x_cat)
            q_mean = q.mean(dim=1, keepdim=True)
            out = q - q_mean + v
        else:
            out = self.out_linear(x_cat)

        return out, state
