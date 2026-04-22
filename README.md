# Robust NMF (PyTorch)

A PyTorch implementation of Robust Non-negative Matrix Factorization (NMF) with sparse error correction.

## Overview

This implementation decomposes a matrix `A` into low-rank components `W` and `H`, plus a sparse error matrix `S`:

```
A ≈ W × H + S
```

The algorithm is robust to outliers and corruptions in the input data.

## Requirements

- Python 3.8+
- PyTorch 1.8+
- CUDA (optional, for GPU acceleration)

## Installation

```bash
pip install torch
```

## Usage

```python
import torch
from rnmf_torch import robust_nmf_torch

# Create input matrix (with outliers)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
A = torch.rand(1000, 2000, device=device)

# Add sparse outliers
mask = torch.rand_like(A) < 0.01
A[mask] += 50.0

# Run Robust NMF
W, H, S = robust_nmf_torch(A, k=15, lamb=0.5)

# Reconstruct
L = W @ H
print(f"Reconstruction error: {torch.linalg.norm(A - L - S):.4f}")
print(f"Sparsity of S: {(S > 0).float().mean().item():.4f}")
```

## API

### `robust_nmf_torch(A, k, lamb=None, tol=1e-6, max_outer_iter=200, inner_nmf_iter=20)`

**Parameters:**
- `A`: Input matrix (2D torch.Tensor)
- `k`: Rank of the low-rank approximation
- `lamb`: Sparsity regularization parameter (default: `1/sqrt(max(m,n))`)
- `tol`: Convergence tolerance (default: `1e-6`)
- `max_outer_iter`: Maximum outer iterations (default: `200`)
- `inner_nmf_iter`: Inner NMF iterations per outer loop (default: `20`)

**Returns:**
- `W`: Left factor matrix (m × k)
- `H`: Right factor matrix (k × n)
- `S`: Sparse error matrix (m × n)

## Algorithm

1. Initialize `W` and `H` using truncated SVD
2. Alternating updates:
   - Clean input: `A_clean = ReLU(A - S)`
   - Update `H` and `W` using multiplicative update rules
   - Update sparse error `S` via soft thresholding
3. Converge when relative loss change < tolerance

## License

MIT License
