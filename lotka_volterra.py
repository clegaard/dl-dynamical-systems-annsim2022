from importlib.metadata import requires
from turtle import color
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


if __name__ == "__main__":
    h = 0.001
    t_start = 0.0
    t_end = 5.0
    t = torch.arange(t_start, t_end + h, h)
    Δt = t[1:] - t[:-1]
    x0 = 2.0
    y0 = 1.0
    α = 1.0
    β = 1.0
    γ = 1.0
    δ = 1.0

    # Fully mechanistic model

    X = [x0]
    Y = [y0]
    x = x0
    y = y0

    for h in Δt:
        dx = α * x - β * x * y
        dy = δ * x * y - γ * y
        x = x + h * dx
        y = y + h * dy
        X.append(x)
        Y.append(y)

    fig, ax = plt.subplots()
    ax.plot(t, X, label="prey", color="blue")
    ax.plot(t, Y, label="predators", color="red")
    ax.set_xlabel("t[s]")
    ax.set_ylabel("population(t)")
    ax.set_title(f"Lotka-Volterra α={α},β={β},γ={γ},δ={δ}")
    ax.legend()

    # Parameter estimation
    n_epochs = 100
    α_estimate = torch.tensor(3.0, requires_grad=True)
    α_estimate.requires_grad = True
    lr = 0.001  # learning rate
    losses = []

    X = torch.tensor(X)
    Y = torch.tensor(Y)

    for _ in tqdm(range(n_epochs)):
        x = torch.tensor(x0, requires_grad=True)
        y = torch.tensor(y0, requires_grad=True)
        X_estimate = [x]
        Y_estimate = [y]

        for h in Δt:
            dx = α_estimate * x - β * x * y
            dy = δ * x * y - γ * y
            x = x + h * dx
            y = y + h * dy
            X_estimate.append(x)
            Y_estimate.append(y)

        X_estimate = torch.stack(X_estimate)
        Y_estimate = torch.stack(Y_estimate)

        loss = torch.linalg.norm(X - X_estimate) + torch.linalg.norm(Y - Y_estimate)
        losses.append(loss.item())
        loss.backward()
        with torch.no_grad():
            α_estimate -= lr * α_estimate.grad
            α_estimate.grad.zero_()

    X_estimate = X_estimate.detach().cpu()
    Y_estimate = Y_estimate.detach().cpu()

    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss(epoch)")

    fig, ax = plt.subplots()
    ax.plot(t, X, label="prey", color="blue")
    ax.plot(t, Y, label="predators", color="red")
    ax.plot(t, X_estimate, label="prey estimate", color="red", linestyle="dotted")
    ax.plot(t, Y_estimate, label="predators estimate", color="blue", linestyle="dotted")
    ax.set_xlabel("t[s]")
    ax.set_ylabel("population(t)")
    ax.legend()
    ax.set_title(f"Lotka-Volterra α={α_estimate:.2f},β={β},γ={γ},δ={δ}")

    plt.show()
