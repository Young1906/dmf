import random
import torch
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import pickle


class SparseMatrix(Dataset):
    """
    Custom dataloader for Deep Matrix Factorization
    """
    def __init__(self, csv_path, i, j, v, neg_r = 1, transform=None):
        """
        Args:

        Return:

        """
        super(SparseMatrix, self).__init__()

        # Transform
        self.transform = transform

        # fnname:
        fn_name = csv_path.split("/")[-1].replace(".", "_")

        # Read the data
        csv = pd.read_csv(csv_path)
        
        self.Y = pd.pivot(
            csv, 
            index = i,
            columns = j,
            values = v
        ).fillna(0.)

        # List of user and items
        self.users, self.items = self.Y.index, self.Y.columns
        self.u_dim, self.v_dim = self.Y.shape
        
        # Calculating sparsity of Y
        N_Interaction = len(self.users) * len(self.items)
        self.sparsity = csv.shape[0] / N_Interaction


        try:
            with open(f"cache/{fn_name}_cache.pkl", "rb") as f:
                self.train, self.test = pickle.load(f)
        
        except OSError:
            # Positive interaction
            pos_samples = []
            
            for _, row in tqdm(csv.iterrows()):
                u, v, _, r = row.values
                u, v = int(u), int(v)
                pos_samples.append((u, v, r))

            # Randomly sample negative interaction
            N_neg = int(N_Interaction * neg_r * self.sparsity)
            neq_u, neq_v = np.random.choice(self.users, N_neg), np.random.choice(self.items, N_neg)
            neq_samples = list(zip(neq_u, neq_v, [0]*N_neg))

            # All samples = pos + neg samples
            samples = [*pos_samples, *neq_samples]
            n_sample = len(samples)

            # Split
            test_size = int(n_sample * .15)
            random.shuffle(samples)
            self.test = samples[:test_size]
            self.train = samples[test_size:]

            # Cache this
            with open(f"cache/{fn_name}_cache.pkl", "wb") as f:
                pickle.dump((self.train, self.test), f)

    def __len__(self):
        return len(self.train)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        i, j, y = self.train[idx]

        # Vector from interaction matrix
        u, v = self.Y.loc[i,:], self.Y.loc[:, j]

        u = np.array(u).astype(np.float32)
        v = np.array(v).astype(np.float32)
        y = np.array(y).astype(np.float32)
        
        # Implicit feedback y = 1 if r_ij > 0 else 0
        # y = 1. if y else 0.
        
        sample = {"u": u, "v": v, "y": y}

        if self.transform:
            sample = self.transform(sample)

        return sample
