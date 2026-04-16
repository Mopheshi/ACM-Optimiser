# Adaptive Circular Manifold (ACM) Optimiser

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

Official PyTorch implementation of the **Adaptive Circular Manifold (ACM)** optimiser, as presented at the 8th International Congress on Human-Computer Interaction, Optimization and Robotic Applications (ICHORA 2026).

**Paper:** [Adaptive Circular Manifold (ACM): A Density-Dependent Geometric Metric for Robust Gradient Descent](https://github.com/Mopheshi/ACM-Optimiser) *(Accepted at ICHORA 2026)*

---

## 📖 Abstract

Standard optimisation methods, such as basic Gradient Descent and adaptive variants like Adam, typically rely on fixed geometries or sub-Riemannian scaling metrics. This rigidity causes major failures in AI training: aggressive overshooting in steep, dense regions and stagnation on flat plateaus. 

The **Adaptive Circular Manifold (ACM)** is an $\mathcal{O}(N)$ approximation of Natural Gradient Descent that incorporates decoupled weight decay. ACM abandons the variance square-root scaling of Adam in favour of a strictly Tikhonov-regularised diagonal Riemannian metric tensor. By measuring the density of the local gradient, ACM dynamically shrinks or expands its manifold trust region, inherently preventing geometric collapse while guaranteeing $\mathcal{O}(1/\sqrt{T})$ convergence.

### Key Features
* **Riemannian Metric Tensor:** Replaces the flawed square-root denominator of Adam with a mathematically bounded geometric constraint ($G^{-1}_t = 1 / (1 + \kappa \rho_t)$).
* **Decoupled Weight Decay:** Natively supports $\lambda$ regularisation, achieving state-of-the-art parity with AdamW.
* **Massive Variance Reduction:** Empirically proven to reduce statistical variance by over 3$\times$ compared to Adam on heavily imbalanced datasets.
* **Structural Immunity to Noise:** Safely navigates extreme (40%) label noise environments without catastrophic overfitting.

---

## 🛠 Installation

You can install the dependencies and use the optimiser directly from this repository:

```bash
git clone https://github.com/Mopheshi/ACM-Optimiser.git
cd ACM-Optimiser
pip install -r requirements.txt
````

*(Optional: If you have configured the `setup.py`, you can install it as a package using `pip install -e .`)*

-----

## 🚀 Quick Start

ACM acts as a direct, drop-in replacement for any standard PyTorch optimiser (e.g., `torch.optim.Adam`).

```python
import torch
from acm.optimiser import ACM

# 1. Initialise your model
model = MyNeuralNetwork()

# 2. Drop in the ACM optimiser
# Note: ACM requires a slightly higher learning rate than Adam due to its strict geometric braking.
# The 'kappa' parameter controls the manifold sensitivity (default: 10.0, optimal for ResNet: 5.0).
optimiser = ACM(
    model.parameters(), 
    lr=0.005,           # Base learning rate
    kappa=5.0,          # Manifold boundary strictness
    beta1=0.9,          # Momentum decay
    beta2=0.99,         # Density decay
    weight_decay=0.01   # Decoupled weight decay
)

# 3. Standard PyTorch training loop
loss = criterion(model(inputs), targets)
loss.backward()
optimiser.step()
optimiser.zero_grad()
```

-----

## 📂 Repository Structure

```text
acm-optimiser/
│
├── acm/                        # Main Python package
│   ├── __init__.py             
│   └── optimiser.py            # Contains the core ACM class definition
│
├── experiments/                # Scripts to reproduce ICHORA 2026 benchmarks
│   ├── utils.py                # Training engines and reproducibility utilities
│   ├── run_rosenbrock.py       # Geometric trajectory mapping on Rosenbrock
│   ├── run_fashionmnist.py     # 40% label noise overfitting tests
│   ├── run_cora.py             # GCN stability on non-Euclidean topologies
│   └── run_cassava.py          # ResNet-18 statistical variance and hardware scaling
│
├── notebooks/                  
│   └── acm_demo.ipynb          # Interactive Jupyter notebook for getting started
│
├── tests/                      
│   └── test_optimiser.py       # PyTest suite for mathematical bounds and gradient checks
│
├── CITATION.cff                # GitHub integration for automated citations
├── LICENSE                     # MIT License
├── README.md                   
└── requirements.txt            # Project dependencies
```

-----

## 🔬 Reproducing the Paper Experiments

To guarantee full reproducibility of the claims made in the ICHORA 2026 paper, run the benchmark scripts located in the `experiments/` directory. All scripts are configured with deterministic seeding (`seed=42`, `100`, `2026`).

```bash
# Example: Run the Rosenbrock Topography Test
python experiments/run_rosenbrock.py

# Example: Run the ResNet-18 Statistical Validation (Requires Kaggle Cassava Dataset)
python experiments/run_cassava.py
```

-----

## 📜 Citation

If you find this code or the ACM framework useful in your research, please consider citing our ICHORA 2026 paper:

```bibtex
@inproceedings{edward2026acm,
  title={Adaptive Circular Manifold (ACM): A Density-Dependent Geometric Metric for Robust Gradient Descent},
  author={Edward, Ndachimya Magaji and Daniel, Jibrailu and Dauda, Yohanna},
  booktitle={8th International Congress on Human-Computer Interaction, Optimization and Robotic Applications (ICHORA)},
  year={2026},
  organization={IEEE}
}
```

## 📄 License

This project is licensed under the MIT Licence - see the [LICENCE](LICENCE) file for details.
