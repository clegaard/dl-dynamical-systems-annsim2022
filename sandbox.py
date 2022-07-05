from tqdm import tqdm
from torch.linalg import norm
import matplotlib.pyplot as plt
import torch
a = 0.5
b = 1.0
x = torch.arange(0,1,0.1)
y = a*x+b

a_est = torch.tensor(0.1) # initial guess, 0-dimension tensor represents scalar
b_est = torch.tensor(0.5) # initial guess, 0-dimension tensor represents scalar
y_est = a_est*x+b_est

fig,ax = plt.subplots()
ax.plot(x,y,label="true")
ax.plot(x,y_est,label="estimated")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.legend()

n_epochs = 1000 # number of training iterations
lr = 0.001 # learning rate
a_est.requires_grad = True
b_est.requires_grad = True

print(a_est.requires_grad)

for i in tqdm(range(n_epochs)):
    y_est = a_est * x + b_est
    loss = torch.linalg.norm(y - y_est) / x.shape[0]
    loss.backward()
    with torch.no_grad():
        a_est -=  lr*a_est.grad
        b_est -=  lr*b_est.grad
        a_est.grad = None
        b_est.grad = None  