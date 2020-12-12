import torch
from torch import nn 
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
from torchvision import transforms
import numpy as np
from torch.nn import functional as F
import torch.optim as optim
from tqdm import tqdm

class Dset(Dataset):
    def __init__(self, csv_path, transform=None):
        super(Dset, self).__init__()
        
        # transform
        self.transform = transform

        # data from csv file
        csv = pd.read_csv(csv_path)
        Y = pd.pivot_table(
                csv,
                index="customer_id",
                columns="sku",
                values="qty").fillna(0)
        
        self.u_dim, self.v_dim = Y.shape
        self.len, _ = csv.shape
        def gen():
            for i in tqdm(Y.index):
                for j in Y.columns:
                    y = Y.loc[i, j]
                
                    if y:
                        u = Y.loc[i,:].values
                        v = Y.loc[:,j].values

                        self.len+=1

                        yield{
                            "u": np.array(u).astype(np.float32),
                            "v": np.array(v).astype(np.float32),
                            "y": np.array(y).astype(np.float32)
                         }
        self.data = gen()


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = next(self.data)


        if self.transform:
            sample = self.transform(sample)

        return sample


class Net(nn.Module):
    def __init__(self, u_dim, v_dim, latent_dim):
        super(Net, self).__init__()
        self.dense1_u = nn.Linear(v_dim, latent_dim)
        self.dense1_v = nn.Linear(u_dim, latent_dim)

    def forward(self, u, v):
        u = F.relu(self.dense1_u(u))
        v = F.relu(self.dense1_v(v))

        out = torch.sum(u * v, dim=-1)
        return out




if __name__ == "__main__":

    # Loading the dataset
    dataset = Dset("dataset.csv")

    latent_dim = 50
    u_dim, v_dim = dataset.u_dim, dataset.v_dim
    
    dataloader = DataLoader(dataset, 
            batch_size = 4,
            shuffle=True,
            num_workers=1
            )

    # Init network
    net = Net(u_dim, v_dim, latent_dim)
    crit = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-6, momentum=.9)

    for i in range(5):
        rloss = 0.
        for j, sample in enumerate(dataloader):
            u, v, y = sample["u"], sample["v"], sample["y"]

            out = net(u, v)
            loss = crit(out, y)
            loss.backward()

            optimizer.step()
            rloss+=loss.item()

            if j % 1000 == 999:
                print(f"epoch {i+1}, batch {j+1}, loss = {rloss}")
                rloss = 0


