import jax.numpy as jnp
from jax import grad, jit, vmap


from tqdm import tqdm
import matplotlib.pyplot as plt


def solve_euler(f, t, x0):
    ΔT = t[1:] - t[:-1]
    x_cur = x0
    X = [x_cur]

    for t, Δt in tqdm(zip(t[1:], ΔT)):
        dy = f(t, x_cur)
        x_new = x_cur + Δt * dy
        X.append(x_new)
        x_cur = x_new

    return jnp.stack(X, axis=1)


from jax.lax import scan


def solve_euler_scan(f, t, x0):
    step_sizes = t[1:] - t[:-1]

    def f_scan(x_cur, t_and_step):
        t, step = t_and_step
        dydt = f(t, x_cur)
        x_new = x_cur + step * dydt  # account for step size
        return x_new, x_new

    _, X = scan(f_scan, init=x0, xs=(t[1:], step_sizes))
    X = jnp.concatenate((x0.reshape(-1, 1), X.T), axis=1)
    return X


# def solve_euler_scan(f, t, x0):
#     step_sizes = t[:1] - t[:-1]

#     def f_scan(tx_carry, slice):
#         t_cur, x_cur = tx_carry
#         dydt = f(t_cur, x_cur)
#         x_new = x_cur + 0.001 * dydt  # account for step size
#         return (t_cur, x_new), x_new

#     _, X = scan(f_scan, init=(t[0], x0), xs=t)
#     return X.T


def f(t, x):
    x, y = x
    dx = α * x - β * x * y
    dy = δ * x * y - γ * y
    return jnp.array((dx, dy))


if __name__ == "__main__":
    h = 0.001
    t_start = 0.0
    t_end = 5.0
    t = jnp.arange(t_start, t_end + h, h)
    x0 = 2.0
    y0 = 1.0
    α = 1.0
    β = 1.0
    γ = 1.0
    δ = 1.0

    z0 = jnp.array((x0, y0))

    X = solve_euler_scan(f, t, z0)

    fig, ax = plt.subplots()
    ax.plot(t, X[0], label="prey", color="blue")
    ax.plot(t, X[1], label="predators", color="red")
    ax.set_xlabel("t[s]")
    ax.set_ylabel("population(t)")
    ax.set_title(f"Lotka-Volterra α={α}, β={β}, γ={γ}, δ={δ}")
    ax.legend()
    plt.show()

    # # Parameter estimation
    # n_epochs = 100
    # α_estimate = torch.tensor(3.0, requires_grad=True)
    # α_estimate.requires_grad = True
    # lr = 0.001  # learning rate
    # losses = []

    # X = torch.tensor(X)
    # Y = torch.tensor(Y)

    # for _ in tqdm(range(n_epochs)):
    #     x = torch.tensor(x0, requires_grad=True)
    #     y = torch.tensor(y0, requires_grad=True)
    #     X_estimate = [x]
    #     Y_estimate = [y]

    #     for h in Δt:
    #         dx = α_estimate * x - β * x * y
    #         dy = δ * x * y - γ * y
    #         x = x + h * dx
    #         y = y + h * dy
    #         X_estimate.append(x)
    #         Y_estimate.append(y)

    #     X_estimate = torch.stack(X_estimate)
    #     Y_estimate = torch.stack(Y_estimate)

    #     loss = torch.linalg.norm(X - X_estimate) + torch.linalg.norm(Y - Y_estimate)
    #     losses.append(loss.item())
    #     loss.backward()
    #     with torch.no_grad():
    #         α_estimate -= lr * α_estimate.grad
    #         α_estimate.grad.zero_()

    # X_estimate = X_estimate.detach().cpu()
    # Y_estimate = Y_estimate.detach().cpu()

    # fig, ax = plt.subplots()
    # ax.plot(losses)
    # ax.set_xlabel("epoch")
    # ax.set_ylabel("loss(epoch)")

    # fig, ax = plt.subplots()
    # ax.plot(t, X, label="prey", color="blue")
    # ax.plot(t, Y, label="predators", color="red")
    # ax.plot(t, X_estimate, label="prey estimate", color="red", linestyle="dotted")
    # ax.plot(t, Y_estimate, label="predators estimate", color="blue", linestyle="dotted")
    # ax.set_xlabel("t[s]")
    # ax.set_ylabel("population(t)")
    # ax.legend()
    # ax.set_title(f"Lotka-Volterra α={α_estimate:.2f},β={β},γ={γ},δ={δ}")

    # plt.show()
