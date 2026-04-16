import os
import random

import numpy as np
import torch
from torch.optim import Optimizer


# ==========================================
# STRICT REPRODUCIBILITY FUNCTION
# ==========================================
def set_seed(seed=42):
    """
    Set random seeds for reproducibility.

    This ensures identical random states across PyTorch, NumPy, and Python's
    built-in hashing.

    Args:
        seed (int, optional): The random seed to use (default: 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Executing on: {device}")
set_seed(42)


class ACM(Optimizer):
    """
    Adaptive Circular Manifold (ACM) Optimiser.

    This optimiser dynamically scales gradient steps using a strictly Tikhonov-regularised
    diagonal Riemannian metric tensor instead of the traditional variance square-root constraint.
    By operating on a density-dependent geometrical manifold, it effectively mitigates extreme
    overshooting in steep topographies and prevents stagnation on flat plateaus.

    Args:
        params (iterable): An iterable of parameters to optimise or dicts defining parameter groups.
        lr (float, optional): The base learning rate. Due to strict geometric braking, this is
            typically set higher than Adam (default: 0.001).
        kappa (float, optional): The manifold boundary strictness coefficient, controlling the
            sensitivity of the metric tensor formulation (default: 10.0).
        beta1 (float, optional): The exponential decay rate for the tangent space inertia estimates (default: 0.9).
        beta2 (float, optional): The exponential decay rate for the directional density (default: 0.99).
        weight_decay (float, optional): The decoupled weight decay coefficient for L2 regularisation (default: 0.0).
    """
    def __init__(self, params, lr=0.001, kappa=10.0, beta1=0.9, beta2=0.99, weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, kappa=kappa, beta1=beta1, beta2=beta2, weight_decay=weight_decay)
        super(ACM, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single geometric step on the loss surface.

        Args:
            closure (callable, optional): A callable structure that re-evaluates the mathematical
                bound and returns the loss scalar.

        Returns:
            loss (float | None): The loss evaluated by the closure subroutine.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # 1. Decoupled Weight Decay
                if group['weight_decay'] != 0:
                    p.data.mul_(1.0 - group['lr'] * group['weight_decay'])

                grad = p.grad.data
                state = self.state[p]

                # State initialisation
                if len(state) == 0:
                    state['step'] = 0  # We keep step ONLY for momentum bias correction
                    state['momentum'] = torch.zeros_like(p.data)
                    state['density'] = torch.zeros_like(p.data)

                state['step'] += 1
                momentum, density = state['momentum'], state['density']
                beta1, beta2 = group['beta1'], group['beta2']
                kappa = group['kappa']
                lr = group['lr']
                step = state['step']

                # 2. Tangent Space Inertia & Directional Density
                momentum.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                density.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # 3. Momentum Bias Correction (Satisfies Theorem 1 Assumption)
                bias_correction1 = 1.0 - beta1 ** step
                m_hat = momentum / bias_correction1

                # 4. The ACM Metric Tensor (Raw Density - Satisfies Lemma 1 Bounds)
                metric_tensor = 1.0 / (1.0 + kappa * density)

                # 5. Geodesic Update
                p.data.addcmul_(m_hat, metric_tensor, value=-lr)

        return loss
