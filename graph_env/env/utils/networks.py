import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, device='cpu'):
        super(GATNetwork, self).__init__()
        self.device = device
        self.conv1 = GATConv(input_dim, hidden_dim, num_heads)
        self.fc1 = nn.Linear(hidden_dim * num_heads, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, obs, state=None, info={}):
        logits = []
        for observation in obs.observation:
            # TODO: Need here we move tensors to CUDA, cannot just put it in Batch because of data time -> slows down
            x = torch.as_tensor(observation[2], device=self.device, dtype=torch.float32)
            edge_index = torch.as_tensor(observation[0], device=self.device, dtype=torch.int)
            x = self.conv1(x, edge_index)
            x = x[observation[3]].view(x[observation[3]].size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            logits.append(x.flatten())
        logits = torch.stack(logits)
        return logits, state


