"""
Unit tests for the Adaptive Circular Manifold (ACM) optimiser.

These tests ensure mathematical correctness, convergence properties, and
the retention of strict deterministic reproducibility as prescribed for
the ICHORA 2026 benchmark standards.
"""

import pytest
import torch

from acm.optimiser import ACM
from experiments.utils import set_seed


def test_acm_initialisation():
    """
    Ensure the optimiser cleanly rejects invalid hyperparameters.
    """
    tensor = torch.tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(ValueError):
        ACM([tensor], lr=-0.01)
    with pytest.raises(ValueError):
        ACM([tensor], weight_decay=-0.1)


def test_reproducibility():
    """
    Check that deterministic seeding produces identical results.
    """

    def run_simulation():
        set_seed(42)
        params = torch.nn.Parameter(torch.tensor([5.0, -3.0], dtype=torch.float32))
        opt = ACM([params], lr=0.1, kappa=5.0)

        for _ in range(5):
            opt.zero_grad()
            loss = (params ** 2).sum()
            loss.backward()
            opt.step()

        return params.data.clone()

    result_one = run_simulation()
    result_two = run_simulation()

    assert torch.allclose(result_one,
                          result_two), "Deterministic seeding failed to produce identical gradient trajectories."


def test_mathematical_bounds_and_step():
    """
    Check that the metric tensor is calculated precisely as
    1 / (1 + kappa * density) without epsilon or square roots.
    """
    set_seed(100)
    params = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float32))
    opt = ACM([params], lr=0.1, kappa=10.0, beta1=0.9, beta2=0.99)

    # Calculate initial loss and propagate backward
    loss = params ** 2
    loss.backward()
    grad = params.grad.data  # Expected gradient: 4.0

    # Manual mathematical trace of Step 1
    expected_momentum = 0.9 * 0.0 + 0.1 * 4.0  # 0.4
    expected_density = 0.99 * 0.0 + 0.01 * (4.0 ** 2)  # 0.16

    bias_correction1 = 1.0 - 0.9 ** 1  # 0.1
    m_hat = expected_momentum / bias_correction1  # 0.4 / 0.1 = 4.0

    # Evaluate the ACM metric tensor constraint
    metric_tensor = 1.0 / (1.0 + 10.0 * expected_density)  # 1.0 / (1.0 + 10.0 * 0.16) = 1.0 / 2.6 = 0.384615

    expected_update = -0.1 * m_hat * metric_tensor  # -0.1 * 4.0 * 0.384615 = -0.153846
    expected_param = 2.0 + expected_update  # 1.846154

    opt.step()

    assert torch.allclose(params.data, torch.tensor([expected_param], dtype=torch.float32)), \
        "The geodesic parameter update diverged from the theoretical mathematical approximation."


def test_decoupled_weight_decay():
    """
    Check that decoupled weight decay modifies the parameters
    before gradient steps and density updates.
    """
    set_seed(2026)
    params = torch.nn.Parameter(torch.tensor([10.0], dtype=torch.float32))
    opt = ACM([params], lr=0.1, weight_decay=0.1)

    opt.zero_grad()
    loss = params * 2.0
    loss.backward()

    opt.step()

    # Mathematical trace for decoupled regularisation
    # Step 1: Decentralised multiplication -> 10.0 * (1.0 - 0.1 * 0.1) = 9.9
    # Step 2: Gradient calculation
    # For iteration 1, gradient = 2.0, m_hat = 2.0, density = 0.01 * 4 = 0.04
    # kappa = 10.0 (default)
    # metric_tensor = 1.0 / (1.0 + 10.0 * 0.04) = 1.0 / 1.4 = 0.714285
    # step = -0.1 * 2.0 * 0.714285 = -0.142857
    # updated param = 9.9 - 0.142857 = 9.757143

    assert 9.8 > params.data > 9.7, \
        "Decoupled weight decay was not applied correctly."
