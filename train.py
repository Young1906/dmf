from matplotlib import pyplot as plt
from matplotlib import animation
import torch
from torch import nn 
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
from torchvision import transforms
import numpy as np
from torch.nn import functional as F
import torch.optim as optim
from tqdm import tqdm
from SparseMatrix import SparseMatrix
import logging, queue, threading

class Net(nn.Module):
    def __init__(self, u_dim, v_dim, latent_dim):
        super(Net, self).__init__()

        # Layer for user's transformation
        self.dense1_u = nn.Linear(v_dim, latent_dim * 3)
        self.dense2_u = nn.Linear(latent_dim * 3, latent_dim * 2)
        self.dense3_u = nn.Linear(latent_dim * 2, latent_dim)

        # Layers of item's transformation
        self.dense1_v = nn.Linear(u_dim, latent_dim * 3)
        self.dense2_v = nn.Linear(latent_dim * 3, latent_dim * 2)
        self.dense3_v = nn.Linear(latent_dim * 2, latent_dim)

    def forward(self, u, v):
        u = F.relu(self.dense1_u(u))
        u = F.relu(self.dense2_u(u))
        u = F.relu(self.dense3_u(u))
        v = F.relu(self.dense1_v(v))
        v = F.relu(self.dense2_v(v))
        v = F.relu(self.dense3_v(v))

        # out = torch.sigmoid(torch.sum(u * v, dim=-1))
        out = torch.sum(u * v, dim = -1)
        return out


class ClipY:
    def __call__(self, sample):

        u, v, y = sample["u"], sample["v"], sample["y"]

        # adaptation for implicit learning
        # paper: https://web.stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf
        alpha = .5
        # y = 1 + alpha * np.log(1 + y)
        # y = np.array(y, dtype=np.float32)
        
        if y > 19:
            y = 20
            # y = np.array(1., dtype=np.float32)

        sample = {"u":u, "v":v, "y":y}
        
        return sample


all_transform = transforms.Compose([ 
        ClipY()
    ])



def train():
    #config:
    BATCH_TO_PRINT = 1000
    BATCH_TO_VALIDATE = 3000
    TRAIN_RATIO = .80

    # Loading the dataset
    dataset = SparseMatrix("dataset.csv", transform = all_transform)

    train_size = int(len(dataset)*TRAIN_RATIO)
    test_size  = len(dataset) - train_size
    train, test = random_split(dataset, [train_size, test_size])

    latent_dim = 30
    u_dim, v_dim = dataset.u_dim, dataset.v_dim
    
    train_loader = DataLoader(train, 
            batch_size = 16,
            shuffle=True,
            num_workers=1
            )

    test_loader = DataLoader(test,
            batch_size = 16,
            shuffle=False,
            num_workers =2)

    # Init network
    net = Net(u_dim, v_dim, latent_dim)
    crit = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(),
            lr=1e-3, 
            momentum=.9
            # weight_decay=1e-3
            )

    min_val_loss = 1e9

    for i in range(2):
        rloss = 0.
        racc  = 0.
        for j, sample in enumerate(train_loader):
            u, v, y = sample["u"], sample["v"], sample["y"]

            optimizer.zero_grad()

            out = net(u, v)
            loss = crit(out, y)
            loss.backward()

            optimizer.step()
            loss_i = loss.item()
            rloss+= loss_i/BATCH_TO_PRINT

            
            if j % BATCH_TO_PRINT == (BATCH_TO_PRINT - 1):
                # cal loss on validation
                if j % BATCH_TO_VALIDATE == (BATCH_TO_VALIDATE - 1):
                    with torch.no_grad():
                        vloss = 0
                        for vsample in test_loader:
                            v_u, v_v, v_y = vsample["u"], vsample["v"], vsample["y"] 
                            vout = net(v_u, v_v)

                            vpred = ((vout > .5) * 1).numpy()
                            loss = crit(vout, v_y)
                            vloss+= loss.item()/test_size

                    print(f"epoch {i+1}, batch {j+1}, loss = {rloss:.3f}, val loss = {vloss:.3f}")
    
                    if vloss < min_val_loss:
                        torch.save({
                            "model": net.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "vloss": vloss,
                            "rloss": rloss
                        }, "./checkpoint.pth")
                        min_val_loss = vloss
                else:
                    print(f"epoch {i+1}, batch {j+1}, loss = {rloss:.3f}")
                rloss = 0
                racc = 0




if __name__ == "__main__":
    train()
