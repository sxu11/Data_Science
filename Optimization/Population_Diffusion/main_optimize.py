

import torch



x = torch.ones(2, 2, requires_grad=True)

def T_relu(x):
    return -1*T.log(1+T.exp(x))

k = 100


linterm = g*T_relu(T.dot(w,x_p)+b)
T.sum(linterm, 0)