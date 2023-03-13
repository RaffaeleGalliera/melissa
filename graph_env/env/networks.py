import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import global_mean_pool


class GATNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, features_only = False, device='cpu'):
        super(GATNetwork, self).__init__()
        self.device = device
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.conv1 = GATv2Conv(input_dim, hidden_dim, num_heads)
        if not features_only:
            self.fc_model = nn.Sequential(
                nn.Linear(hidden_dim * num_heads, hidden_dim * num_heads),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim * num_heads, 2),
            )
        else:
            self.fc_model = nn.Sequential(
                nn.Linear(hidden_dim * num_heads, hidden_dim * num_heads),
                nn.ReLU(inplace=True)
            )
            self.output_dim = hidden_dim * num_heads

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
            x = self.fc_model(x)
            logits.append(x.flatten())
        logits = torch.stack(logits)
        return logits, state

