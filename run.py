from DMF import SparseMatrix, SimpleNet
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch import nn
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter



if __name__ == "__main__":

    writer = SummaryWriter(flush_secs=10)

    # Load dataset
    dataset = SparseMatrix("dataset/dataset.csv",
        "customer_id", "sku", "qty"
    )
    
    # Train test split
    train_size = int(len(dataset)*.99)
    test_size  = len(dataset) - train_size

    train, test = random_split(dataset, [train_size, test_size])

    # Dataloader
    train_loader = DataLoader(train, 
            batch_size = 32,
            shuffle=True,
            num_workers=1
            )

    test_loader = DataLoader(test,
            batch_size = 32,
            shuffle=False,
            num_workers = 1)


    # device:
    device = torch.device("cuda:0")

    # Init network
    net = SimpleNet(dataset.u_dim, dataset.v_dim, 20)
    net.to(device)
    
    # And loss
    crit = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(),
        lr=1e-3, 
        momentum=.9
    )
    gstep = 0
    for e in range(5):
        for i, sample in enumerate(train_loader):
            rloss = 0.
            racc = 0.

            u, v, y = sample["u"].to(device), sample["v"].to(device), sample["y"].to(device)
            
            optimizer.zero_grad()
            out = net(u, v)

            with torch.no_grad():
                pred = out.cpu().numpy() > .5
                racc += np.mean(pred == y.cpu().numpy())


            loss = crit(out, y)
            loss.backward()

            optimizer.step()
            loss_i = loss.item()
            rloss+=loss_i
            gstep += 1

            if gstep % 200 == 0:
                writer.add_scalar("Loss/Train", rloss, gstep)
                writer.add_scalar("Acc/Train", racc, gstep)
                rloss = 0
                racc = 0

                if gstep % 1000 == 0:
                    with torch.no_grad():
                        vloss, vacc = 0., 0.
                        for _, vsample in enumerate(test_loader):
                            u, v, y = sample["u"].to(device), sample["v"].to(device), sample["y"].to(device)

                            out = net(u, v)
                            pred = out.cpu().numpy() > .5

                            loss = crit(out, y)
                            loss_i = loss.item()

                            vloss+=loss_i/test_size
                            vacc += np.sum(pred == y.cpu().numpy())/test_size

                            writer.add_scalar("Loss/validate", vloss, gstep)
                            writer.add_scalar("Acc/validate", vacc, gstep)
    writer.close()



