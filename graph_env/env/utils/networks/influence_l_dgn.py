from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.pool import radius_graph

from tianshou.utils.net.common import MLP

from graph_env.env.utils.constants import RADIUS_OF_INFLUENCE


def build_pyg_batch_time(
    obs: torch.Tensor | np.ndarray,
    radius: float,
    device: torch.device | str,
    node_dim: int,
    agents_num: int,
) -> tuple[Data, torch.Tensor, int]:
    """Convert the flattened observation (nodes, edges, ctrl‑idx) into a
    PyG `Data` batch.  Edge attrs are one‑hot (E,3) with 0/1/2 class.

    Args
    ----
    obs : 2‑D tensor (bs, flatten_len)
    node_dim : columns per node *excluding* (x,y) and dm_flag.
               I.e. NUMBER_OF_FEATURES.
    Returns
    -------
    data : PyG Data with directed `edge_index`, `edge_attr` (one‑hot)
    ctrl_idx : (bs,) long tensor — index of controlling agent per
                batch entry.
    bs        : batch size (int)
    """
    if isinstance(obs, np.ndarray):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    else:
        obs = obs.to(device).float()

    bs, flat_dim = obs.shape

    # -------- dimensions --------
    per_node_dim = node_dim + 2 + 1          # + x,y + dm_flag
    nodes_block  = agents_num * per_node_dim
    edges_block  = agents_num * agents_num

    # unwrap blocks
    nodes_flat   = obs[:, :nodes_block]
    edges_flat   = obs[:, nodes_block: nodes_block + edges_block]
    ctrl_idx     = obs[:, -1].long()

    # reshape nodes -> (bs, N, per_node_dim)
    nodes = nodes_flat.view(bs, agents_num, per_node_dim)
    nodes_all = nodes.reshape(bs * agents_num, per_node_dim)

    pos   = nodes_all[:, :2]                     # (x,y)
    feats = nodes_all[:, 2:-1]                   # real features (NODE_DIM)
    dm    = nodes_all[:, -1:].clone()            # (bs*N,1)

    # batch vector
    batch_vec = torch.arange(bs, device=device).repeat_interleave(agents_num)

    # directed radius graph (source->target)
    edge_index = radius_graph(
        pos, r=radius, batch=batch_vec, loop=False, flow="source_to_target"
    )  # shape (2,E)

    # -------- edge class -> one‑hot --------
    edges_cls = edges_flat.view(bs, agents_num, agents_num).long()
    src_nodes = edge_index[0] % agents_num
    dst_nodes = edge_index[1] % agents_num
    batch_of_edge = batch_vec[edge_index[0]]

    cls = edges_cls[batch_of_edge, src_nodes, dst_nodes]  # (E,)
    cls = torch.clamp(cls, min=0)                         # -1 → 0 (ignored)
    edge_attr = F.one_hot(cls.to(torch.long), num_classes=3).float()

    data = Data(
        x=feats, edge_index=edge_index, edge_attr=edge_attr,
        pos=pos, batch=batch_vec
    )
    data.ptr = torch.arange(0, (bs + 1) * agents_num, step=agents_num, device=device)
    data.dm_mask = dm  # (bs*N,1)

    return data, ctrl_idx, bs


# ---------------------------------------------------------------------
# LDGNNetwork
# ---------------------------------------------------------------------

class LDGNNetwork(nn.Module):
    """Lightweight Directed Graph Network (LDGN) with two GATv2 layers that
    consume one‑hot edge attributes (dim=3).

    * Expects observation produced by InfluenceGraph.observe().
    * Uses `build_pyg_batch_time` to convert to PyG graph.
    * Returns Q‑values (dueling optional).
    """

    def __init__(
        self,
        node_input_dim: int,                # NUMBER_OF_FEATURES
        hidden_dim: int,
        output_dim: int,
        num_heads: int,
        agents_num: int,
        dueling_param: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.node_input_dim = node_input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.agents_num = agents_num
        self.use_dueling = dueling_param is not None

        # encoder maps raw node features -> hidden_dim
        self.encoder = MLP(
            input_dim=node_input_dim,
            hidden_sizes=[hidden_dim],
            output_dim=hidden_dim,
            device=self.device,
        )

        # two GATv2 layers with edge_dim=3 (one‑hot attr)
        self.conv1 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=num_heads,
            edge_dim=3,
        )
        self.conv2 = GATv2Conv(
            in_channels=hidden_dim * num_heads,
            out_channels=hidden_dim,
            heads=num_heads,
            edge_dim=3,
        )

        # latent dim after concat of three snapshots (x1,x2,x3)
        self.latent_dim = hidden_dim + hidden_dim * num_heads * 2

        if self.use_dueling:
            q_cfg, v_cfg = dueling_param
            self.Q = MLP(
                input_dim=self.latent_dim,
                output_dim=q_cfg.get("output_dim", output_dim),
                hidden_sizes=q_cfg.get("hidden_sizes", [hidden_dim]),
                device=self.device,
            )
            self.V = MLP(
                input_dim=self.latent_dim,
                output_dim=v_cfg.get("output_dim", 1),
                hidden_sizes=v_cfg.get("hidden_sizes", [hidden_dim]),
                device=self.device,
            )
            self.out_dim = self.Q.output_dim
        else:
            self.final_lin = nn.Linear(self.latent_dim, output_dim)
            self.out_dim = output_dim

    # --------------------------------------------------------------
    # forward
    # --------------------------------------------------------------
    @torch.no_grad()
    def _gather_ctrl_emb(self, emb: torch.Tensor, data: Data, ctrl_idx: torch.Tensor):
        """Return (bs, feature) tensor of embeddings for controlling nodes."""
        # data.ptr : offset per batch
        offsets = data.ptr[:-1]  # (bs,)
        global_idx = offsets + ctrl_idx  # broadcasting
        return emb[global_idx]

    def forward(self, obs: torch.Tensor, state=None, info: dict | None = None):
        # 1) build graph
        data, ctrl_idx, bs = build_pyg_batch_time(
            obs=obs,
            radius=RADIUS_OF_INFLUENCE,
            device=self.device,
            node_dim=self.node_input_dim,
            agents_num=self.agents_num,
        )

        # 2) encode nodes
        x = F.relu(self.encoder(data.x))
        x1 = self._gather_ctrl_emb(x, data, ctrl_idx)

        # 3) GAT layer 1
        x = F.relu(self.conv1(x, data.edge_index, data.edge_attr))
        x2 = self._gather_ctrl_emb(x, data, ctrl_idx)

        # 4) mask non‑DM nodes before second layer (optional but keeps parity)
        x = x * data.dm_mask

        # 5) GAT layer 2
        x = F.relu(self.conv2(x, data.edge_index, data.edge_attr))
        x3 = self._gather_ctrl_emb(x, data, ctrl_idx)

        # 6) concatenate snapshots → latent
        lat = torch.cat([x1, x2, x3], dim=-1)

        # 7) heads
        if self.use_dueling:
            q = self.Q(lat)
            v = self.V(lat)
            q_mean = q.mean(dim=1, keepdim=True)
            out = v + (q - q_mean)
        else:
            out = self.final_lin(lat)
        return out, None
