import torch
import torch.nn.functional as F

def robust_nmf_torch(A, k, lamb=None, tol=1e-6, max_outer_iter=200, inner_nmf_iter=20):
    m, n = A.shape
    device = A.device
    
    if lamb is None:
        lamb = 1.0 / torch.sqrt(torch.tensor(max(m, n), dtype=torch.float32, device=device))
        
    A_norm = torch.linalg.norm(A, ord='fro')
    
    print("Initiation with low rank SVD...")
    U, s, V = torch.svd_lowrank(A, q=k)   
    sqrt_s = torch.sqrt(s)
    W = torch.abs(U * sqrt_s)
    H = torch.abs((V * sqrt_s).T)     
    S = torch.zeros_like(A)
    eps = 1e-9 
    prev_loss = float('inf')
    
    for i in range(max_outer_iter):
        A_clean = F.relu(A - S) 
        
        for _ in range(inner_nmf_iter):
            WtA = W.T @ A_clean
            WtWH = (W.T @ W) @ H
            H = H * (WtA / (WtWH + eps))
            AHt = A_clean @ H.T
            WHHt = W @ (H @ H.T)
            W = W * (AHt / (WHHt + eps))

        L = W @ H
        residual = A - L
        S = torch.sign(residual) * F.relu(torch.abs(residual) - lamb)
        
        current_loss = torch.linalg.norm(A - L - S, ord='fro') / A_norm
        loss_change = torch.abs(prev_loss - current_loss)
        
        if i % 5 == 0:
            print(f"Iter {i:3d} | Rel Loss: {current_loss:.5f} | Loss Change: {loss_change:.6e}")
            
        if loss_change < tol:
            print(f"✅ converge in iteration {i} !!!!")
            break
        prev_loss = current_loss
    return W, H, S


# ================= test =================
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# A = torch.rand(1000, 2000, device=device) 
# mask = torch.rand_like(A) < 0.01
# A[mask] += 50.0 

# W, H, S = robust_nmf_torch(A, k=15, lamb=0.5)

# L = W @ H
# print(f"reconstitutional error: ||A - L - S||_F : {torch.linalg.norm(A - L - S):.4f}")
# print(f"ratio of non-zeros in S: {(S > 0).float().mean().item():.4f}")