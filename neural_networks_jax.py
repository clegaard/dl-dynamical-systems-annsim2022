from random import random
import jax.numpy as jnp
from jax import grad, jit, value_and_grad, vmap
from jax.lax import scan
from jax import random


def predict(params, inputs):
    for W, b in params:
        outputs = jnp.dot(inputs, W.T) + b
        inputs = jnp.tanh(outputs)
    return outputs


def loss_f(params, inputs, targets):
    preds = predict(params, inputs)
    return jnp.sum((preds - targets) ** 2)


def initialize_mlp(sizes, key):
    keys = random.split(key, len(sizes))

    def initialize_layer(m, n, key, scale=1e-2):
        w_key, b_key = random.split(key)
        return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

    return [initialize_layer(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


if __name__ == "__main__":
    key = random.PRNGKey(1)
    params = initialize_mlp([2, 32, 32, 2], key)

    for W, b in params:
        print(f"{W.shape}, {b.shape}")

    predict(params, jnp.empty((10, 2)))
