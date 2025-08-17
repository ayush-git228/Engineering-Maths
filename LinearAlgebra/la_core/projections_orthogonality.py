# %% [markdown]
# Projections & Orthogonality
# - projection matrix P = A (A^T A)^{-1} A^T (orthogonal projection onto col(A))
# - projection via QR or least-squares for numerical stability
# - checks: idempotent & symmetric for orthogonal projection matrices

# %%
import numpy as np
from typing import Tuple

# %%
def orthogonal_projection_matrix(A: np.ndarray) -> np.ndarray:
    """
    Construct orthogonal projection matrix onto Col(A).
    For numerical stability, uses QR when A has full column rank, otherwise uses pseudo-inverse.
    Returns P with shape (n, n) where A is (n, k).
    """
    n, k = A.shape[0], A.shape[1]
    if k == 0:
        return np.zeros((n, n))
    # Prefer QR if columns likely independent
    if np.linalg.matrix_rank(A) == k:
        Q = np.linalg.qr(A, mode='reduced')[0]  # n x k orthonormal
        return Q @ Q.T
    # Fallback: use pseudo-inverse
    A_pinv = np.linalg.pinv(A)
    return A @ A_pinv

# %%
def project_vector(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Project y onto column space of A using projection matrix.
    """
    P = orthogonal_projection_matrix(A)
    return P @ y

# %%
def is_projection_matrix(P: np.ndarray, tol: float = 1e-10) -> Tuple[bool, bool]:
    """
    Return (is_idempotent, is_symmetric) within tolerance.
    For orthogonal projection both True.
    """
    idempotent = np.allclose(P @ P, P, atol=tol)
    symmetric = np.allclose(P.T, P, atol=tol)
    return idempotent, symmetric

# %%
# Script example
if __name__ == "__main__":
    A = np.array([[1.,0.],[1.,1.],[0.,1.]])  # columns span a 2-dim subspace of R^3
    y = np.array([2., 0., 1.])

    P = orthogonal_projection_matrix(A)
    y_hat = project_vector(A, y)
    idemp, symm = is_projection_matrix(P)

    print("Projection matrix P shape:", P.shape)
    print("P @ P == P?", idemp)
    print("P symmetric?", symm)
    print("Original y:", y)
    print("Projected y_hat:", y_hat)
    print("Residual (y - y_hat):", y - y_hat)