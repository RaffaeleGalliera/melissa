import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn.pool import radius_graph
from torch_geometric.transforms import Cartesian, Distance

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from tianshou.utils.net.common import MLP
from torch_geometric.nn import TransformerConv, global_max_pool
from graph_env.env.utils.constants import RADIUS_OF_INFLUENCE


def build_pyg_batch_time(obs: torch.Tensor, radius: float, device: torch.device, input_dim: int) -> tuple:
    """
    Expects obs of shape (bs, N*input_dim + 1). For example, if:
      - input_dim=7 columns per node
      - N=20 nodes
      => N*input_dim=140, plus 1 controlling-agent index => 141 columns total per row
    We then parse:
      - The last column => controlling-agent index
      - The first N*input_dim => node_data => shape (bs, N, input_dim)
        (with columns [0..1] possibly for position, columns [2..6] for other features)

    Steps:
      1) ncols_without_agent = dim - 1
      2) N = ncols_without_agent // input_dim
      3) node_data => shape (bs, N, input_dim)
      4) Flatten to (bs*N, input_dim)
      5) columns [0..1] => pos, columns [2..(input_dim-1)] => node_feats
      6) build radius_graph(pos, batch, r=radius)
      7) controlling_agent = obs[:, -1]
      8) return (data, agent_indices, bs)
    """
    if obs.ndim != 2:
        raise ValueError(f"Expected obs to be 2D, but got shape {obs.shape}")

    bs, dim = obs.shape  # e.g. (7, 141)
    ncols_without_agent = dim - 1       # 140
    # e.g. if input_dim=7 => N = 140 // 7 = 20
    N = int(ncols_without_agent / (2 + input_dim))
    if N * (2 + input_dim) != ncols_without_agent:
        raise ValueError(
            f"Mismatch: we have {ncols_without_agent} columns for nodes, "
            f"but input_dim={input_dim} => N={N} leaves remainder. "
            f"obs.shape={obs.shape}"
        )

    # (bs, N, input_dim)
    node_data = obs[:, :ncols_without_agent].reshape(bs, N, (2 + input_dim)).float()

    # flatten => (bs*N, input_dim)
    x_all = node_data.reshape(bs * N, (2 + input_dim)).to(device)

    # Suppose columns [0..1] = (pos_x, pos_y), columns [2..(input_dim)] = the GNN features
    # If input_dim=7 => columns [2..7] => 5 features
    pos = x_all[:, :2]     # shape (bs*N, 2)
    node_feats = x_all[:, 2:]  # shape (bs*N, input_dim-2)

    # Build PyG Data
    batch_vec = torch.arange(bs, device=device).repeat_interleave(N)
    edge_index = radius_graph(pos, batch=batch_vec, r=radius, loop=False)


    data = Data(
        x=node_feats,   # the actual GNN features
        pos=pos,        # used for adjacency or transforms
        edge_index=edge_index,
        batch=batch_vec,
    )
    data.ptr = torch.arange(0, (bs + 1) * N, step=N, device=device)


    # Optional transforms
    data = Cartesian(norm=False, cat=True)(data)
    data = Distance(norm=False)(data)

    # controlling agent index is the last column
    agent_indices = obs[:, -1].clamp(0, N - 1).long().to(device)

    return data, agent_indices, bs

class DGNRNetwork(nn.Module):
    """
    A non-recurrent TransformerConv-based network for multi-node subgraphs.
    We assume each row in obs is (N*input_dim + 1). For example (20*7 + 1=141).
    The last column is controlling-agent index in [0..N-1].
    Columns [0..1] per node => (pos_x, pos_y)
    Columns [2..(input_dim-1)] => GNN features
    """

    def __init__(
        self,
        input_dim: int,    # e.g. 7
        hidden_dim: int,
        output_dim: int,
        num_heads: int,
        device: str = 'cpu',
        features_only: bool = False,
        dueling_param: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
        aggregator_function=global_max_pool,  # optional aggregator
    ):
        super().__init__()
        self.device = device
        self.radius = RADIUS_OF_INFLUENCE
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.final_latent_dim = hidden_dim * num_heads
        self.use_dueling = dueling_param is not None
        self.input_dim = input_dim  # total columns per node

        # The GNN features are (input_dim - 2) if we skip the first 2 columns (pos).
        # e.g. if input_dim=7 => the net sees 5 columns per node
        self.encoder = MLP(
            input_dim=input_dim,
            hidden_sizes=[hidden_dim],
            output_dim=hidden_dim,
            device=device
        )

        self.conv1 = TransformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=num_heads,
            root_weight=False,
            device=device
        )
        self.conv2 = TransformerConv(
            in_channels=hidden_dim * num_heads,
            out_channels=hidden_dim,
            heads=num_heads,
            root_weight=False,
            device=device
        )

        if self.use_dueling:
            q_kwargs, v_kwargs = dueling_param
            q_output_dim = q_kwargs.pop("output_dim", output_dim)
            v_output_dim = v_kwargs.pop("output_dim", 1)

            self.Q = MLP(
                input_dim=self.final_latent_dim,
                output_dim=q_output_dim,
                device=device,
                **q_kwargs
            )
            self.V = MLP(
                input_dim=self.final_latent_dim,
                output_dim=v_output_dim,
                device=device,
                **v_kwargs
            )
        else:
            self.out_linear = nn.Linear(self.final_latent_dim, output_dim)

    def forward(self, obs: torch.Tensor, state=None, info=None):
        """
        obs.shape = (bs, N*input_dim + 1).
        E.g. (7, 141) => bs=7, N=20, input_dim=7 => 20*7=140 + 1=141
        """
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, device=self.device)

        data, agent_indices, bs_size = build_pyg_batch_time(
            obs, self.radius, self.device, self.input_dim
        )
        # data.x => shape (bs*N, input_dim-2) if columns [2..(input_dim)]
        x_enc = self.encoder(data.x)
        x_enc = F.relu(x_enc)

        x_enc = self.conv1(x_enc, data.edge_index)
        x_enc = F.relu(x_enc)

        x_enc = self.conv2(x_enc, data.edge_index)
        x_enc = F.relu(x_enc)

        # data.ptr => shape (bs_size+1,), so subgraph i has nodes in [data.ptr[i], data.ptr[i+1])
        # The controlling agent is data.ptr[i] + agent_indices[i].
        global_indices = data.ptr[:-1] + agent_indices
        agent_emb = x_enc[global_indices]  # shape (bs_size, final_latent_dim)

        if self.use_dueling:
            q = self.Q(agent_emb)
            v = self.V(agent_emb)
            q_mean = q.mean(dim=1, keepdim=True)
            final_out = q - q_mean + v
        else:
            final_out = self.out_linear(agent_emb)

        return final_out, None
