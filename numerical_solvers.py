from scipy.integrate import solve_ivp
from tqdm import tqdm
import torch

if __name__ == "__main__":

    t_start = 0.0
    t_end = 1.0
    step_size = 0.001
    T = torch.arange(t_start, t_end, step_size)
    ΔT = T[1:] - T[:-1]
    y0 = torch.tensor((0.0, 0.0))

    def f(t, y):
        pass

    y = 0
    Y = [y0]

    for t, Δt in zip(T[1:], ΔT):

        dydt = f(t, Δt)
        y += Δt * dydt
