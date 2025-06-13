from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tianshou.utils.net.common import MLP
from torch_geometric.nn import GATv2Conv
from graph_env.env.utils.networks.common import build_pyg_batch_time
from graph_env.env.utils.constants import RADIUS_OF_INFLUENCE


class LDGNNetwork(nn.Module):
    """
    Updated LDGNNetwork using the new full-graph approach:
      1) Each obs row is shape (N*node_dim + 1).
      2) We parse it into a PyG batch, gather controlling-agent embeddings
         at each layer, cat them, and do a dueling head (or single) as before.
    """
    def __init__(
            self,
            input_dim: int,         # total columns per node
            hidden_dim: int,
            output_dim: int,
            num_heads: int,
            agents_num: int,
            dueling_param: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
            device: str = 'cpu',
            edge_attributes = False,
    ):
        super(LDGNNetwork, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.agents_num = agents_num
        self.edge_attributes = edge_attributes
        # We'll produce embeddings from each layer, so the final dimension is hidden_dim*heads for each layer
        # but we cat them from 3 "snapshots" => x_1, x_2, x_3 => total = hidden_dim*heads * 3
        # plus the initial "encoder" snapshot if you want x_0 as well. The old code only does 3 snapshots:
        # We can do something like hidden_dim * num_heads * 3
        # But the old code used "hidden_dim + hidden_dim * num_heads * 2" => the logic was a bit different.
        # We'll replicate that logic exactly, or we can do simpler. Let's replicate your old final dimension:
        self.final_latent_dim = hidden_dim + hidden_dim * num_heads * 2
        # Whether we do dueling
        self.use_dueling = (dueling_param is not None)

        # Build MLP for input
        self.encoder = MLP(
            input_dim=input_dim,
            hidden_sizes=[hidden_dim],
            output_dim=hidden_dim,
            device=self.device
        )
        # Two GATv2Conv layers
        self.conv1 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=num_heads
        )
        self.conv2 = GATv2Conv(
            in_channels=hidden_dim * num_heads,
            out_channels=hidden_dim,
            heads=num_heads
        )

        # If dueling, build Q/V heads
        if self.use_dueling:
            q_kwargs, v_kwargs = dueling_param
            # Suppose we want 2 actions or something else. Typically user sets "output_dim" for Q.
            q_output_dim = q_kwargs.pop("output_dim", output_dim)
            v_output_dim = v_kwargs.pop("output_dim", 1)

            # We'll pass self.final_latent_dim as input to Q and V
            q_kwargs.update({
                "input_dim": self.final_latent_dim,
                "output_dim": q_output_dim,
                "device": self.device
            })
            v_kwargs.update({
                "input_dim": self.final_latent_dim,
                "output_dim": v_output_dim,
                "device": self.device
            })
            self.Q = MLP(**q_kwargs)
            self.V = MLP(**v_kwargs)
            # We'll set self.output_dim = q_output_dim so the policy knows final shape
            self.output_dim = q_output_dim
        else:
            self.out_linear = nn.Linear(self.final_latent_dim, output_dim)

    def forward(self, obs, state=None, info={}):
        """
        obs: shape (bs, N*node_dim + 1). The last col => controlling agent index
        We'll build a PyG Data, do 2 GATv2Conv steps, gather controlling-agent node
        embeddings after each step (like old x_1, x_2, x_3).
        Then cat them => final dimension self.final_latent_dim.
        Then do dueling or linear.

        Return: (bs, #actions), state=None
        """
        # 1) Convert obs to tensor
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, device=self.device)

        # 2) Build PyG Data
        data, agent_indices, bs_size = build_pyg_batch_time(
            obs,
            radius=RADIUS_OF_INFLUENCE,
            device=self.device,
            input_dim=self.input_dim,
            edge_attributes=self.edge_attributes,
            agents_num=self.agents_num
        )

        # 3) MLP encoder
        x = self.encoder(data.x)  # shape (bs*N, hidden_dim)
        x = F.relu(x)
        # gather controlling agent embedding => x_1
        # data.ptr: shape (bs_size+1,)
        global_indices = data.ptr[:-1] + agent_indices
        x_1 = x[global_indices, :]

        # 4) GATv2Conv #1
        x = self.conv1(x, data.edge_index)
        x = F.relu(x)
        x_2 = x[global_indices, :]
        x = x * data.dm_mask # apply the decision-maker mask

        # 5) GATv2Conv #2
        x = self.conv2(x, data.edge_index)
        x = F.relu(x)
        x_3 = x[global_indices, :]

        # 6) cat => shape (bs_size, hidden_dim + hidden_dim*num_heads*2),
        # matching old final_latent_dim from your snippet
        x_cat = torch.cat([x_1, x_2, x_3], dim=1)

        # 7) Dueling or single
        if self.use_dueling:
            q = self.Q(x_cat)
            v = self.V(x_cat)
            # standard dueling: Q_final = q - mean(q) + v
            q_mean = q.mean(dim=1, keepdim=True)
            x_out = q - q_mean + v
        else:
            x_out = self.out_linear(x_cat)

        return x_out, None
