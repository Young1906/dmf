from torch import nn 
from torch.nn import functional as F

class Net(nn.Module):
    def __init__(self, u_dim, v_dim, latent_dim):
        super(Net, self).__init__()

        # Layer for user's transformation
        self.norm0_u  = nn.LayerNorm(v_dim)
        # self.dense1_u = nn.Linear(v_dim, latent_dim )
        # self.norm1_u  = nn.LayerNorm(latent_dim)
        # self.dense2_u = nn.Linear(latent_dim + v_dim, latent_dim)
 
        # Layers of item's transformation
        self.norm0_v  = nn.LayerNorm(u_dim)
        # self.dense1_v = nn.Linear(u_dim, latent_dim)
        # self.norm1_v  = nn.LayerNorm(latent_dim)
        # self.dense2_v = nn.Linear(latent_dim + u_dim, latent_dim)

        # Activation function
        self.a = nn.LeakyReLU(.001)

    def u_presentation(self, u):
        # Normalizing input
        u = self.norm0_u(u)

        # f(W.x)
        u = self.a(self.dense1_u(u))
        return u

    def v_presentation(self, v):
        # normalizing input
        v = self.norm0_v(v)
        
        # f(Wx)
        v = self.a(self.dense1_v(v))
        return v

    def forward(self, u, v):
        u, v = self.u_presentation(u), self.v_presentation(v)
        out = torch.sum(u * v, dim = -1)
        return out
