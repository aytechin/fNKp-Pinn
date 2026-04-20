# fNKp-Galerkin: Spectral-Galerkin PINN for Fractional Diffusion Equations

A spectral-physics-informed neural network for bivariate fractional diffusion equations using finite bivariate biorthogonal N-Konhauser polynomials. The method achieves **exact** evaluation of the Riemann-Liouville spatial derivative without Grünwald-Letnikov quadrature.

[![arXiv](https://img.shields.io/badge/arXiv-2504.XXXXX-red)](https://arxiv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Key Innovation

The method uses an **x^q-weighted** ansatz:

$$\hat{u}_{\theta}(t,x) = x^{q}\sum_{s=0}^{S} a_s K_s^{(p,q)}(t,x)$$

which enables the **exact identity**:

$$D_x^\beta[x^q K_s^{(p,q)}] = x^{q-\beta} K_s^{(p, q-\beta)}$$

This makes the spatial RL derivative **quadrature-free** — unlike standard fPINN which relies on Grünwald-Letnikov approximation.

## Installation

```bash
pip install torch numpy scipy
```

## Quick Start

```python
import torch
from fnkp_core_v2 import LinearHeadFNKpPINN, InSpanTarget

# Create model with S=2 basis functions
model = LinearHeadFNKpPINN(S=2, upsilon=1, p_init=10.0, q_init=0.8)

# In-span target (exact recovery benchmark)
target = InSpanTarget(S_star=2, p0=10.0, q0=0.8,
                 coeffs=torch.tensor([1.0, 0.3, -0.2]))

# Forward pass
t = torch.rand(100)
w = torch.rand(100)
u_hat = model(t, w)
```

## Running Experiments

```bash
# Run all experiments
python run_extended.py

# Run specific experiments
python run_extended.py --only IN        # In-span benchmark
python run_extended.py --only OUT       # Out-of-span benchmark
python run_extended.py --only SS        # S-convergence
python run_extended.py --only FDM     # Classical FDM baseline
```

## Key Results

| Benchmark | fNKp-Galerkin | fPINN | bML-PINN |
|-----------|-------------|-------|----------|
| **In-span** (rel L2) | **2.08×10⁻⁷** | 6.72×10⁻² | 7.19×10⁻² |
| **Out-of-span** | **36.4%** | 40.3% | 39.9% |

### In-Span Performance
- fNKp achieves **near-machine precision** (~10⁻⁷) on in-span targets
- **5+ orders of magnitude** better than fPINN/bML baselines
- **3+ orders of magnitude** better than classical FDM at N=320

### S-Convergence
- **Matched S** (S=S*) is the optimal regime
- Under-parametrised (S<S*): expressivity limited
- Over-parametrised (S>S*): optimisation limited (direct Galerkin solve recommended)

## Repository Structure

```
.
├── fnkp_core_v2.py          # Core library
├── run_extended.py          # Experiment runner
├── results_extended.pt      # Saved results (94 experiments)
├── README.md
└── LICENSE
```

### Core Classes

- `StableFNKpBasis` — Biorthogonal N-Konhauser polynomial basis with caching
- `LinearHeadFNKpPINN` — Linear-head Galerkin ansatz (main method)
- `MLPHeadFNKpPINN` — MLP-head variant (ablation)
- `VanillaFPINN` — Standard fPINN baseline
- `BMLPINN` — bML-PINN baseline
- `FDMReference` — Classical fractional FDM solver
- `InSpanTarget` — Exact-solution benchmark generator

### Experiments (exp codes)

| Code | Description |
|------|------------|
| IN | In-span benchmark (sweet-spot) |
| OUT | Out-of-span benchmark |
| SS | S-convergence study |
| FDM | Classical FDM baseline |
| NQ | Caputo quadrature sensitivity |
| FAIR | fPINN GL order sensitivity |
| L1 | L1 regularisation at S>S* |
| OOS_S | S-convergence on out-of-span |

## Theory Highlights

1. **Zero Residual**: If u* lies in the x^q-weighted fNKp span, the interior residual vanishes identically (Proposition)

2. **Loss-to-Error Bound**: Non-asymptotic bound combining approximation error and optimisation rate (Theorem 2)

3. **S-Refinement**: Monotone approximation but κ_S-dependent optimisation (Theorem 3)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- SciPy (for FDM reference and Gauss-Jacobi quadrature)

## License

MIT License — see LICENSE file.

## Citation

```bibtex
@article{fnkp2025,
  title={A Spectral-Galerkin Physics-Informed Network for Bivariate Fractional Diffusion Equations Based on Finite N-Konhauser Polynomials},
  author={},
  journal={arXiv:2504.XXXXX},
  year={2025}
}
```

## References

- Güldogan Lekesiz, Çekim, Özarslan (2025) — Finite bivariate biorthogonal N-Konhauser polynomials
- Raissi, Perdikaris, Karniadakis (2019) — Physics-informed neural networks
- Meerschaert, Tadjeran (2004) — Finite difference methods for fractional PDEs