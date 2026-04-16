import os
import sys
# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import torch

from acm.optimiser import ACM
from experiments.utils import set_seed

def rosenbrock(tensor):
    x, y = tensor[0], tensor[1]
    return (1.0 - x)**2 + 100.0 * (y - x**2)**2

def get_trajectory(optimizer_class, start_coords, steps, **kwargs):
    p = torch.tensor(start_coords, requires_grad=True)
    optimizer = optimizer_class([p], **kwargs)
    trajectory = [p.detach().numpy().copy()]

    for _ in range(steps):
        optimizer.zero_grad()
        loss = rosenbrock(p)
        loss.backward()
        optimizer.step()
        trajectory.append(p.detach().numpy().copy())
    return np.array(trajectory)

def main():
    set_seed(42)
    start_point = [-1.5, 2.0]
    iterations = 2500

    traj_sgd  = get_trajectory(torch.optim.SGD, start_point, iterations, lr=0.001)
    traj_adam = get_trajectory(torch.optim.Adam, start_point, iterations, lr=0.05)
    # ACM uses lr=0.1 because the G^-1 tensor rigorously controls the step size
    traj_acm  = get_trajectory(ACM, start_point, iterations, lr=0.1, kappa=5.0)

    x_grid = np.linspace(-2.0, 2.0, 400)
    y_grid = np.linspace(-1.0, 3.0, 400)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = (1.0 - X)**2 + 100.0 * (Y - X**2)**2

    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, np.log(Z), levels=50, cmap='viridis', alpha=0.8)
    plt.colorbar(label='Log(Loss)')

    plt.plot(traj_sgd[:, 0], traj_sgd[:, 1], color='red', label='SGD', linewidth=2, alpha=0.8)
    plt.plot(traj_adam[:, 0], traj_adam[:, 1], color='cyan', label='Adam', linewidth=2, alpha=0.8)
    plt.plot(traj_acm[:, 0], traj_acm[:, 1], color='white', label='ACM (Ours)', linewidth=2, linestyle='--')

    plt.scatter(*start_point, color='black', marker='o', s=100)
    plt.scatter(1, 1, color='gold', marker='*', s=300, edgecolor='black', label='Global Min')

    plt.xlim([-2.0, 2.0])
    plt.ylim([-1.0, 3.0])
    plt.title('Optimisation Trajectory on the Rosenbrock Function', fontsize=16)
    plt.legend(loc='upper left')
    plt.savefig('rosenbrock.pdf', format='pdf', bbox_inches='tight', dpi=300)
    print("Saved rosenbrock.pdf")

if __name__ == '__main__':
    main()
