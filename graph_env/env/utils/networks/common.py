from torch_geometric.data import Data
from torch_geometric.nn.pool import radius_graph
from torch_geometric.transforms import Cartesian, Distance
import torch

def build_pyg_batch_time(
        obs: torch.Tensor,
        radius: float,
        device: torch.device,
        input_dim: int,          # now includes the extra mask feature
        edge_attributes: bool,
        agents_num: int
) -> tuple:
    """
    Now expects that each node has (dim_with_extra) columns in obs,
    where obs[:, :2] = (x,y), obs[:, 2:-1] = true GNN features,
    and obs[:, -1] = is_decision_maker flag (0 or 1).
    """

    if obs.ndim != 2:
        raise ValueError(f"Expected obs to be 2D, but got shape {obs.shape}")
    bs, dim = obs.shape
    ncols_without_agent = dim - 1
    dim_with_extra = input_dim + 2 + 1  # +1 for the decision-maker flag
    expected = agents_num * dim_with_extra
    if ncols_without_agent != expected:
        raise ValueError(
            f"Expected {expected} feature cols for nodes, got {ncols_without_agent}"
        )

    # reshape into (bs, N, 2+input_dim)
    node_data = obs[:, :ncols_without_agent].reshape(bs, agents_num, dim_with_extra).float()

    # flatten to (bs*N, 2+input_dim)
    x_all = node_data.reshape(-1, dim_with_extra).to(device)

    # split pos / all_feats
    pos = x_all[:, :2]                 # (bs*N, 2)
    all_feats = x_all[:, 2:]           # (bs*N, input_dim)

    # last column of all_feats is our decision‐maker flag
    dm_mask = all_feats[:, -1].unsqueeze(1)   # (bs*N, 1)
    # drop the mask from the feature tensor
    x_feats = all_feats[:, :-1]              # (bs*N, input_dim-1)

    # build graph connectivity
    batch_vec = torch.arange(bs, device=device).repeat_interleave(agents_num)
    edge_index = radius_graph(pos, batch=batch_vec, r=radius, loop=False)

    data = Data(
        x=x_feats,         # now only the real GNN features
        pos=pos,
        edge_index=edge_index,
        batch=batch_vec,
    )
    data.ptr = torch.arange(0, (bs + 1) * agents_num, step=agents_num, device=device)
    data.dm_mask = dm_mask                      # carry the mask through

    if edge_attributes:
        data = Cartesian(norm=False, cat=True)(data)
        data = Distance(norm=False)(data)

    agent_indices = obs[:, -1].clamp(0, agents_num - 1).long().to(device)
    return data, agent_indices, bs
