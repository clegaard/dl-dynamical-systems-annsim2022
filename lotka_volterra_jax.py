import jax.numpy as jnp
from jax import grad, jit, value_and_grad, vmap
from jax import random
from jax.lax import scan
import matplotlib.pyplot as plt
from tqdm import tqdm


def solve_euler(f, t, y0, args):
    step_sizes = t[1:] - t[:-1]
    y_cur = y0
    Y = [y_cur]

    for t, step_size in zip(t[1:], step_sizes):
        dydt = f(t, y_cur, *args)
        y_new = y_cur + step_size * dydt
        Y.append(y_new)
        y_cur = y_new

    return jnp.stack(Y, axis=1)


def solve_euler_scan(f, t, y0, args):
    step_sizes = t[1:] - t[:-1]

    def f_scan(y_cur, t_and_step):
        t, step_size = t_and_step
        dydt = f(t, y_cur, *args)
        y_new = y_cur + step_size * dydt
        return y_new, y_new

    _, Y = scan(f_scan, init=y0, xs=(t[1:], step_sizes))
    Y = jnp.concatenate((y0.reshape(-1, 1), Y.T), axis=1)
    return Y


def f(t, y, a, b, d, c):
    x, y = y
    dx = a * x - b * x * y
    dy = c * x * y - d * y
    return jnp.array((dx, dy))


def optimization_step_a(y_target, t, y0, a, b, d, c):
    def loss(a):
        y_predicted = solve_euler_scan(f, t, y0, (a, b, d, c))
        return jnp.linalg.norm(y_target - y_predicted)

    loss, dLda = value_and_grad(loss)(a)
    return loss, a - lr * dLda


def optimization_step_abcd(y_target, t, y0, a, b, c, d):
    def loss(a, b, c, d):
        y_predicted = solve_euler_scan(f, t, y0, (a, b, c, d))
        return jnp.linalg.norm(y_target - y_predicted)

    loss, dL = value_and_grad(loss, argnums=(0, 1, 2, 3))(a, b, c, d)
    dLda, dLdb, dLdc, dLdd = dL
    return loss, a - lr * dLda, b - lr * dLdb, c - lr * dLdc, d - lr * dLdd


def predict(params, inputs):
    for W, b in params:
        outputs = jnp.dot(inputs, W.T) + b
        inputs = jnp.tanh(outputs)
    return outputs


def initialize_mlp(sizes, key):
    keys = random.split(key, len(sizes))

    def initialize_layer(m, n, key, scale=1e-2):
        w_key, b_key = random.split(key)
        return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

    return [initialize_layer(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


if __name__ == "__main__":
    h = 0.001
    t_start = 0.0
    t_end = 5.0
    t = jnp.arange(t_start, t_end + h, h)
    y0 = jnp.array((2.0, 1.0))
    a = 1.0
    b = 1.0
    d = 1.0
    c = 1.0

    Y = solve_euler_scan(f, t, y0, (a, b, d, c))

    fig, ax = plt.subplots()
    ax.plot(t, Y[0], label="prey", color="blue")
    ax.plot(t, Y[1], label="predators", color="red")
    ax.set_xlabel("t[s]")
    ax.set_ylabel("population(t)")
    ax.set_title(f"Lotka-Volterra a={a}, b={b}, d={d}, c={c}")
    ax.legend()
    plt.show()

    # Parameter estimation only a
    n_epochs = 1000
    a_estimate = 3.0
    lr = 0.00005  # learning rate
    losses = []

    optimization_step_a = jit(optimization_step_a)

    for _ in tqdm(range(n_epochs)):

        loss, a_estimate = optimization_step_a(Y, t, y0, a_estimate, b, d, c)
        losses.append(loss)

    x_predicted = solve_euler_scan(f, t, y0, (a_estimate, b, d, c))

    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss(epoch)")

    fig, ax = plt.subplots()
    ax.plot(t, Y[0], label="prey", color="blue")
    ax.plot(t, Y[1], label="predators", color="red")
    ax.plot(t, x_predicted[0], label="prey estimate", color="red", linestyle="dotted")
    ax.plot(
        t, x_predicted[1], label="predators estimate", color="blue", linestyle="dotted"
    )
    ax.set_xlabel("t[s]")
    ax.set_ylabel("population(t)")
    ax.legend()
    ax.set_title(f"Lotka-Volterra a={a_estimate:.2f},b={b},c={c},d={d}")

    plt.show()

    # Parameter estimation a,b,c,d
    n_epochs = 1000
    a_estimate = 3.0
    b_estimate = 0.2
    c_estimate = 4.5
    d_estimate = 2.1
    lr = 0.001  # learning rate
    losses = []

    optimization_step_abcd = jit(optimization_step_abcd)

    for _ in tqdm(range(n_epochs)):

        loss, a_estimate, b_estimate, c_estimate, d_estimate = optimization_step_abcd(
            Y, t, y0, a_estimate, b_estimate, c_estimate, d_estimate
        )
        losses.append(loss)

    x_predicted = solve_euler_scan(
        f, t, y0, (a_estimate, b_estimate, c_estimate, d_estimate)
    )

    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss(epoch)")

    fig, ax = plt.subplots()
    ax.plot(t, Y[0], label="prey", color="blue")
    ax.plot(t, Y[1], label="predators", color="red")
    ax.plot(t, x_predicted[0], label="prey estimate", color="red", linestyle="dotted")
    ax.plot(
        t, x_predicted[1], label="predators estimate", color="blue", linestyle="dotted"
    )
    ax.set_xlabel("t[s]")
    ax.set_ylabel("population(t)")
    ax.legend()
    ax.set_title(
        f"Lotka-Volterra a={a_estimate:.2f},b={b_estimate:.2f},c={c_estimate:.2f},d={d_estimate:.2f}"
    )

    plt.show()
