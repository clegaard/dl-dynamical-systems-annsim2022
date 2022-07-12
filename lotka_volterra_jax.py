import jax.numpy as jnp
from jax import grad, jit, value_and_grad, vmap
from torch import solve


from tqdm import tqdm
import matplotlib.pyplot as plt


def solve_euler(f, t, x0):
    ΔT = t[1:] - t[:-1]
    x_cur = x0
    X = [x_cur]

    for t, Δt in zip(t[1:], ΔT):
        dy = f(t, x_cur)
        x_new = x_cur + Δt * dy
        X.append(x_new)
        x_cur = x_new

    return jnp.stack(X, axis=1)


from jax.lax import scan


def solve_euler_scan(f, t, x0, f_parameters):
    step_sizes = t[1:] - t[:-1]

    def f_scan(x_cur, t_and_step):
        t, step = t_and_step
        dydt = f(t, x_cur, f_parameters)
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


def f(t, x):  # fully defined ODE
    x, y = x
    dx = α * x - β * x * y
    dy = δ * x * y - γ * y
    return jnp.array((dx, dy))


def f(t, x, α):
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

    X = solve_euler_scan(f, t, z0, α)

    fig, ax = plt.subplots()
    ax.plot(t, X[0], label="prey", color="blue")
    ax.plot(t, X[1], label="predators", color="red")
    ax.set_xlabel("t[s]")
    ax.set_ylabel("population(t)")
    ax.set_title(f"Lotka-Volterra α={α}, β={β}, γ={γ}, δ={δ}")
    ax.legend()
    plt.show()

    # Parameter estimation
    n_epochs = 1000
    α_estimate = 3.0
    lr = 0.0001  # learning rate
    losses = []

    def loss_f(α):
        x_predicted = solve_euler_scan(f, t, z0, α)
        return jnp.linalg.norm(X - x_predicted)

    @jit
    def optimization_step(α):
        loss, dldα = value_and_grad(loss_f)(α)
        return loss, α - lr * dldα

    for _ in tqdm(range(n_epochs)):

        loss, α_estimate = optimization_step(α_estimate)
        losses.append(loss)

    x_predicted = solve_euler_scan(f, t, z0, α_estimate)

    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss(epoch)")

    fig, ax = plt.subplots()
    ax.plot(t, X[0], label="prey", color="blue")
    ax.plot(t, X[1], label="predators", color="red")
    ax.plot(t, x_predicted[0], label="prey estimate", color="red", linestyle="dotted")
    ax.plot(
        t, x_predicted[1], label="predators estimate", color="blue", linestyle="dotted"
    )
    ax.set_xlabel("t[s]")
    ax.set_ylabel("population(t)")
    ax.legend()
    ax.set_title(f"Lotka-Volterra α={α_estimate:.2f},β={β},γ={γ},δ={δ}")

    plt.show()
