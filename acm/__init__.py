"""
Adaptive Circular Manifold (ACM) Optimiser.

This module provides the official PyTorch implementation of the ACM optimiser,
as presented at the 8th International Congress on Human-Computer Interaction,
Optimization and Robotic Applications (ICHORA 2026). It operates by abandoning
the variance square-root scaling of Adam in favour of a strictly Tikhonov-regularised
diagonal Riemannian metric tensor.
"""

from .optimiser import ACM

__all__ = ['ACM']
