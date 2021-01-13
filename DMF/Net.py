import torch
from torch import nn
from torch.nn import functional as F


class SimpleNet(nn.Module):
    def __init__(self, u_dim, v_dim, latent_dim):
        super(SimpleNet, self).__init__()
        self.dense1_u = nn.Linear(v_dim, latent_dim)
        self.dense1_v = nn.Linear(u_dim, latent_dim)

    def forward(self, u , v):
        p = F.relu(self.dense1_u(u))        
        q = F.relu(self.dense1_v(v))

        out = torch.sum(p * q, dim=-1)
        return out / (torch.norm(p, dim=-1) * torch.norm(q, dim=-1))

