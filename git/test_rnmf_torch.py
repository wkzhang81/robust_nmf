import torch
from rnmf_torch import robust_nmf_torch


def test_robust_nmf_basic():
    """Test basic functionality of robust NMF"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = torch.rand(100, 200, device=device)
    
    W, H, S = robust_nmf_torch(A, k=10, lamb=0.5)
    
    L = W @ H
    reconstruction_error = torch.linalg.norm(A - L - S, ord='fro')
    
    assert W.shape == (100, 10), f"Expected W shape (100, 10), got {W.shape}"
    assert H.shape == (10, 200), f"Expected H shape (10, 200), got {H.shape}"
    assert S.shape == A.shape, f"Expected S shape {A.shape}, got {S.shape}"
    assert reconstruction_error < 1.0, f"Reconstruction error too high: {reconstruction_error}"
    print("✓ Basic test passed")


def test_robust_nmf_with_outliers():
    """Test robust NMF with sparse outliers"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = torch.rand(100, 200, device=device)
    
    # Add sparse outliers
    mask = torch.rand_like(A) < 0.01
    A[mask] += 50.0
    
    W, H, S = robust_nmf_torch(A, k=10, lamb=0.5)
    
    L = W @ H
    reconstruction_error = torch.linalg.norm(A - L - S, ord='fro')
    sparsity = (S > 0).float().mean().item()
    
    assert reconstruction_error < 5.0, f"Reconstruction error too high: {reconstruction_error}"
    print(f"✓ Outlier test passed (sparsity: {sparsity:.4f})")


def test_robust_nmf_custom_params():
    """Test robust NMF with custom parameters"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = torch.rand(50, 80, device=device)
    
    W, H, S = robust_nmf_torch(
        A, 
        k=5, 
        lamb=0.1,
        tol=1e-4,
        max_outer_iter=100,
        inner_nmf_iter=10
    )
    
    assert W.shape == (50, 5), f"Expected W shape (50, 5), got {W.shape}"
    assert H.shape == (5, 80), f"Expected H shape (5, 80), got {H.shape}"
    print("✓ Custom params test passed")


if __name__ == "__main__":
    print("Running tests...")
    test_robust_nmf_basic()
    test_robust_nmf_with_outliers()
    test_robust_nmf_custom_params()
    print("\nAll tests passed!")
