import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class SimpleNet(nn.Module):
    def __init__(self, u_dim, v_dim, latent_dim):
        super(SimpleNet, self).__init__()
        self.dense1_u = nn.Linear(v_dim, latent_dim)
        self.dense1_v = nn.Linear(u_dim, latent_dim)

    def forward(self, u , v):
        p = F.relu(self.dense1_u(u))        
        q = F.relu(self.dense1_v(v))

        out = torch.sum(p * q, dim=-1)
        return out

class Skip(nn.Module):
    def __init__(self, u_dim, v_dim, latent_dim):
        super(Skip, self).__init__()
        self.dense1_u = nn.Linear(v_dim, latent_dim * 10)
        self.dense2_u = nn.Linear(v_dim + latent_dim * 10, latent_dim)

        self.dense1_v = nn.Linear(u_dim, latent_dim * 10)
        self.dense2_v = nn.Linear(u_dim + latent_dim * 10, latent_dim)

    def forward(self, u , v):
        p = torch.sigmoid(self.dense1_u(u))        
        q = torch.sigmoid(self.dense1_v(v))

        p = torch.cat((u, p),-1)
        q = torch.cat((v, q),-1)

        p = torch.sigmoid(self.dense2_u(p))
        q = torch.sigmoid(self.dense2_v(q))

        out = torch.sum(p * q, dim=-1)
        return out


class ClipY:
    def __call__(self, sample):

        u, v, y = sample["u"], sample["v"], sample["y"]
        if y > 3751.3125:
            y = 3751.3125

        y = 1 + np.log(1+y)
        y = np.array(y).astype(np.float32)
       
        sample = {"u":u, "v":v, "y": y}

        return sample
