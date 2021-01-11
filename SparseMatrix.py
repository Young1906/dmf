import torch
from torch import nn 
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
from torchvision import transforms
import numpy as np
from torch.nn import functional as F
import torch.optim as optim
from tqdm import tqdm
import sys, pickle


class SparseMatrix(Dataset):
    """
    Construct dataset from a very sparse matrix
    """
    def __init__(self, csv_path, transform=None):
        
        super(SparseMatrix, self).__init__()
        
        # transform
        self.transform = transform

        csv = pd.read_csv(csv_path)
        self.Y = pd.pivot(
                csv,
                index="customer_id",
                columns="sku",
                values="qty"
        ).fillna(0.)

        self.u_dim, self.v_dim = self.Y.shape

        customer_id = self.Y.index
        sku = self.Y.columns

        self.dataset = []

        try:
            with open("dataset.pkl", "rb") as f:
                self.dataset = pickle.load(f)
        except Exception as e:
            for c in tqdm(customer_id, leave=False):
                for k in sku:
                    y = self.Y.loc[c, k]

                    """
                    u, v = self.Y.loc[c, :], self.Y.loc[:,k]
                    if (np.sum(np.array(u) > 0) < 5) or (np.sum(np.array(v) > 0) < 5):
                        continue
                    """
                                

                    if y:
                        self.dataset.append((c, k , y)) 

                    else: 
                        # Randomly adding y=0 in to dataset
                        
                        if np.random.uniform() < .0004:
                            self.dataset.append((c, k, y))
            with open("dataset.pkl", "wb") as f:
                pickle.dump(self.dataset, f)
                

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # loading customer_id, sku, and Y_{u,v}
        c, k, y = self.dataset[idx]
        u, v = self.Y.loc[c,:], self.Y.loc[:,k]
        
        # masking interaction
        u.at[k] = 0
        v.at[c] = 0

        u, v = u.values, v.values

        # Loading vector
        u = np.array(u).astype(np.float32)
        v = np.array(v).astype(np.float32)
        y = np.array(y).astype(np.float32)


        sample = {"u": u, "v":v, "y": y, "index":[c, k]}


        if self.transform:
            sample = self.transform(sample)

        return sample

if __name__ == "__main__":
    dataset = SparseMatrix("dataset.csv", transform=None)
    dataloader = DataLoader(dataset,
            batch_size = 4,
            num_workers = 1,
            shuffle=True)

    for i, sample in enumerate(dataloader):
        print(sample)
