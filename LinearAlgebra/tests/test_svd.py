import numpy as np

def low_rank_approx(X, k):
    U,S,Vt = np.linalg.svd(X, full_matrices=False)
    k = min(k, len(S))
    return U[:,:k] @ np.diag(S[:k]) @ Vt[:k,:]

def test_low_rank_error_decreases():
    X = np.random.randn(12, 8)
    e2 = np.linalg.norm(X - low_rank_approx(X, 2), 'fro')
    e4 = np.linalg.norm(X - low_rank_approx(X, 4), 'fro')
    assert e4 <= e2 + 1e-9