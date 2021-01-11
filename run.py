from train import Net, all_transform
import torch
from SparseMatrix import *
import pandas as pd

checkpoint = torch.load("checkpoint_50.pth")
net = Net(25106, 9091, 50)
net.load_state_dict(checkpoint["model"])
net.eval()

# device:
device = torch.device("cuda:0")
net.to(device)

if __name__ == "__main__":
    dataset = SparseMatrix("dataset/dataset02.csv", transform = all_transform)
    U = [] 
    with torch.no_grad():
        # Infering user presentation
        for u in dataset.Y.index:
            u_ = dataset.Y.loc[u].values
            u_ = np.array(u_).astype(np.float32)
            u_ = torch.tensor(u_).to(device)

            U.append(net.u_presentation(u_).cpu().numpy())
    
    U = pd.DataFrame(U, index=dataset.Y.index)
    U.to_csv("u.csv")
    print(U)
