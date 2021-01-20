from DMF import SparseMatrix, SimpleNet
import torch
from torch.autograd import Variable

if __name__ == "__main__":
    sm = SparseMatrix("dataset/test.csv",
            "customer_id",
            "sku",
            "qty",
            )

    net = SimpleNet(sm.u_dim, sm.v_dim, 10)
    params= net.parameters()

    reg = Variable(torch.tensor(0.), requires_grad = True)
    for param in params:
        reg = reg + torch.sum(param)

    print(reg)


