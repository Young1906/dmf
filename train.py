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
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

class Net(nn.Module):
    def __init__(self, u_dim, v_dim, latent_dim):
        super(Net, self).__init__()

        # Layer for user's transformation
        self.norm0_u  = nn.LayerNorm(v_dim)
        self.dense1_u = nn.Linear(v_dim, latent_dim )
        self.norm1_u  = nn.LayerNorm(latent_dim)
        self.dense2_u = nn.Linear(latent_dim + v_dim, latent_dim)

        # Layers of item's transformation
        self.norm0_v  = nn.LayerNorm(u_dim)
        self.dense1_v = nn.Linear(u_dim, latent_dim)
        self.norm1_v  = nn.LayerNorm(latent_dim)
        self.dense2_v = nn.Linear(latent_dim + u_dim, latent_dim)

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


class ClipY:
    def __call__(self, sample):

        u, v, y, index = sample["u"], sample["v"], sample["y"], sample["index"]

        if y > 3751.3125:
            y = 3751.3125

        y = 1 + np.log(1+y)
        sample = {"u":u, "v":v, "y": torch.tensor(y).float(), "index":index}
        
        return sample


all_transform = transforms.Compose([ 
        ClipY()
    ])



def train(latent_dim):
    #config:
    BATCH_TO_PRINT = 100
    BATCH_TO_VALIDATE = 200
    TRAIN_RATIO = .99

    # device:
    device = torch.device("cuda:0")

    # Loading the dataset
    dataset = SparseMatrix("dataset/dataset.csv", transform = all_transform)

    train_size = int(len(dataset)*TRAIN_RATIO)
    test_size  = len(dataset) - train_size
    train, test = random_split(dataset, [train_size, test_size])

    latent_dim = latent_dim
    u_dim, v_dim = dataset.u_dim, dataset.v_dim
    
    train_loader = DataLoader(train, 
            batch_size = 32,
            shuffle=True,
            num_workers=1
            )

    test_loader = DataLoader(test,
            batch_size = 32,
            shuffle=False,
            num_workers = 1)

    nbatch_test = len(test_loader)
    nbatch_train = len(train_loader)

    # Init network
    net = Net(u_dim, v_dim, latent_dim)
    net.to(device)

    # crit = nn.MSELoss()
    crit = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),
            lr=1e-6, 
            momentum=.9
            # weight_decay=1e-3
            )

    min_val_loss = 1e9

    for i in range(5):
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
            

            if j % BATCH_TO_PRINT == (BATCH_TO_PRINT - 1):
                # cal loss on validation
                if j % BATCH_TO_VALIDATE == (BATCH_TO_VALIDATE - 1):
                    with torch.no_grad():
                        vloss = 0
                        vacc = 0
                        for vsample in test_loader:
                            v_u, v_v, v_y = vsample["u"].to(device), vsample["v"].to(device), vsample["y"].to(device)
                            # print(vsample["index"])
                            vout = net(v_u, v_v)

                            # vpred = vout.cpu().numpy().argmax(axis=-1)
                            vloss_ = crit(vout, v_y)
                            vloss+= vloss_.item()/nbatch_test
                            # vacc += np.sum(vpred == v_y)/nbatch_test
                    writer.add_scalar("Loss/test", vloss, j)
    
                    if vloss < min_val_loss:
                        torch.save({
                            "model": net.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "vloss": vloss,
                            "rloss": rloss
                        }, f"./checkpoint_{latent_dim}.pth")
                        print("Saved checkpoint!!")
                        min_val_loss = vloss
                        

                else:
                    writer.add_scalar("Loss/train", rloss, j)
                    # print(f"{datetime.datetime.now()}: epoch {i+1}, batch {j+1}, loss = {rloss:.3f}")
                rloss = 0
                racc = 0




if __name__ == "__main__":
    train(50)
