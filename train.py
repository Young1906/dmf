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
import datetime

class Net(nn.Module):
    def __init__(self, u_dim, v_dim, latent_dim):
        super(Net, self).__init__()

        # Layer for user's transformation
        self.norm0_u  = nn.LayerNorm(v_dim)
        self.dense1_u = nn.Linear(v_dim, latent_dim )
        self.norm1_u  = nn.LayerNorm(latent_dim)
        self.dense2_u = nn.Linear(latent_dim + v_dim, latent_dim)
        self.norm2_u  = nn.LayerNorm(latent_dim)
        self.dense3_u = nn.Linear(latent_dim + v_dim, latent_dim)

        # Layers of item's transformation
        self.norm0_v  = nn.LayerNorm(u_dim)
        self.dense1_v = nn.Linear(u_dim, latent_dim)
        self.norm1_v  = nn.LayerNorm(latent_dim)
        self.dense2_v = nn.Linear(latent_dim + u_dim, latent_dim)
        self.norm2_v  = nn.LayerNorm(latent_dim)
        self.dense3_v = nn.Linear(latent_dim + u_dim, latent_dim)


    def forward(self, u, v):

        # Normalizing input
        u = self.norm0_u(u)

        # f(W.x)
        u_ = torch.tanh(self.dense1_u(u))
        u_ = self.norm1_u(u_)
        
        # concat
        u_  = torch.cat((u, u_), -1)

        # f(W.x)
        u_ = torch.tanh(self.dense2_u(u_))
        u_ = self.norm2_u(u_)

        # concat
        u_ = torch.cat((u, u_), -1)

        # f(Wx)
        u_ = torch.tanh(self.dense3_u(u_))
        
        # normalizing input
        v = self.norm0_v(v)
        
        # f(Wx)
        v_ = torch.tanh(self.dense1_v(v))
        v_ = self.norm1_v(v_)

        # concat
        v_ = torch.cat((v, v_), -1)

        # f(Wx)
        v_ = torch.tanh(self.dense2_v(v_))
        v_ = self.norm2_v(v_)

        # concat
        v_ = torch.cat((v, v_), -1)

        # f(Wx):
        v_ = torch.tanh(self.dense3_v(v_))


        out = torch.sum(u_ * v_, dim = -1)
        # out = torch.sigmoid(self.out(u * v))
        return out


class ClipY:
    def __call__(self, sample):

        u, v, y = sample["u"], sample["v"], sample["y"]

        if y > 3751.3125:
            y = 3751.3125

        y = 1 + np.log(1+y)
        sample = {"u":u, "v":v, "y": torch.tensor(y).float()}
        
        return sample


all_transform = transforms.Compose([ 
        ClipY()
    ])



def train():
    #config:
    BATCH_TO_PRINT = 1000
    BATCH_TO_VALIDATE = 3000
    TRAIN_RATIO = .98

    # device:
    device = torch.device("cuda:0")

    # Loading the dataset
    dataset = SparseMatrix("dataset/dataset02.csv", transform = all_transform)

    train_size = int(len(dataset)*TRAIN_RATIO)
    test_size  = len(dataset) - train_size
    train, test = random_split(dataset, [train_size, test_size])

    latent_dim = 60
    u_dim, v_dim = dataset.u_dim, dataset.v_dim
    
    train_loader = DataLoader(train, 
            batch_size = 16,
            shuffle=True,
            num_workers=1
            )

    test_loader = DataLoader(test,
            batch_size = 16,
            shuffle=False,
            num_workers = 1)

    nbatch_test = len(test_loader)

    # Init network
    net = Net(u_dim, v_dim, latent_dim)
    net.to(device)

    crit = nn.MSELoss()
    # crit = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),
            lr=1e-3, 
            momentum=.9
            # weight_decay=1e-3
            )

    min_val_loss = 1e9

    for i in range(10):
        rloss = 0.
        racc  = 0.
        for j, sample in enumerate(train_loader):
            u, v, y = sample["u"].to(device), sample["v"].to(device), sample["y"].to(device)

            optimizer.zero_grad()

            out = net(u, v)
            loss = crit(out, y)
            loss.backward()

            optimizer.step()
            # metrics
            loss_i = loss.item()
            rloss+= loss_i/BATCH_TO_PRINT
            

            # with torch.no_grad():
                # pred = out.cpu().numpy().argmax(axis=-1)
                # racc+= np.sum(pred==y)/BATCH_TO_PRINT

            
            if j % BATCH_TO_PRINT == (BATCH_TO_PRINT - 1):
                # cal loss on validation
                if j % BATCH_TO_VALIDATE == (BATCH_TO_VALIDATE - 1):
                    with torch.no_grad():
                        vloss = 0
                        vacc = 0
                        for vsample in test_loader:
                            v_u, v_v, v_y = vsample["u"].to(device), vsample["v"].to(device), vsample["y"].to(device)
                            vout = net(v_u, v_v)

                            # vpred = vout.cpu().numpy().argmax(axis=-1)
                            vloss_ = crit(vout, v_y)
                            vloss+= vloss_.item()/nbatch_test
                            # vacc += np.sum(vpred == v_y)/nbatch_test

                    print(f"{datetime.datetime.now()}: epoch {i+1}, batch {j+1}, loss = {rloss:.3f}, val loss = {vloss:.3f}")
    
                    if vloss < min_val_loss:
                        torch.save({
                            "model": net.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "vloss": vloss,
                            "rloss": rloss
                        }, "./checkpoint.pth")
                        print("Saved checkpoint!!")
                        min_val_loss = vloss

                else:
                    print(f"{datetime.datetime.now()}: epoch {i+1}, batch {j+1}, loss = {rloss:.3f}")
                rloss = 0
                racc = 0




if __name__ == "__main__":
    train()
