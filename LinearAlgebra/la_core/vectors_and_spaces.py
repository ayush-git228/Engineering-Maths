# %% [markdown]
# Vectors & Vector Spaces — GATE-DA friendly
# - span, linear independence, basis, coordinate projection, norms
# - small runnable examples and helper functions to use from agent code

# %%
import numpy as np
from typing import Iterable, Tuple

# %%
def is_linear_independent(vecs: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Check linear independence of column vectors in `vecs`.
    vecs: shape (n, m) where columns are m vectors in R^n
    Returns True if columns are linearly independent.
    """
    if vecs.ndim != 2:
        raise ValueError("vecs must be 2D matrix with columns as vectors")
    s = np.linalg.svd(vecs, compute_uv=False)
    rank = int(np.sum(s > tol))
    return rank == vecs.shape[1]

# %%
def span_contains(A: np.ndarray, y: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Check if vector y is in the column-space (span) of columns of A.
    Uses least-squares to solve min ||Ax - y||_2 and checks residual.
    """
    if A.ndim != 2 or y.ndim != 1:
        raise ValueError("A must be 2D, y must be 1D")
    if A.shape[0] != y.shape[0]:
        raise ValueError("y must have the same number of rows as A")

    # Solve least squares: min ||Ax - y||
    x, *_ = np.linalg.lstsq(A, y, rcond=None)
    residual = np.linalg.norm(A @ x - y)

    return bool(residual <= tol)

# %%
def basis_from_columns(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Return a basis (as columns) for col(A) using SVD.
    """
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    r = int(np.sum(S > tol))
    return U[:, :r]

# %%
def project_onto_basis(B: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Orthogonally project y onto col(B). B has basis vectors as columns (preferably independent).
    """
    # P = B(B^T B)^{-1} B^T
    BtB = B.T @ B
    if np.linalg.matrix_rank(BtB) < BtB.shape[0]:
        # fallback using lstsq to avoid inversion in singular case
        coeffs, *_ = np.linalg.lstsq(B, y, rcond=None)
        return B @ coeffs
    return B @ np.linalg.inv(BtB) @ B.T @ y

# %%
def orthonormalize_columns(A: np.ndarray) -> np.ndarray:
    """
    Return an orthonormal basis for columns of A using QR (economy).
    """
    Q, R = np.linalg.qr(A, mode='reduced')
    return Q

# %%
# Quick usage examples when run as script
if __name__ == "__main__":
    # define two column vectors in R^3
    v1 = np.array([1., 0., 0.])
    v2 = np.array([1., 1., 0.])
    A = np.column_stack([v1, v2])  # shape (3,2)

    print("Columns of A are linearly independent?", is_linear_independent(A))
    y = np.array([2., 1., 0.])
    print("Is y in span(A)?", span_contains(A, y))

    B = basis_from_columns(A)
    print("Basis for col(A) (columns):\n", B)

    y_proj = project_onto_basis(B, np.array([2., 3., 5.]))
    print("Projection of [2,3,5] onto span(A):", y_proj)

    print("Orthonormal basis via QR:\n", orthonormalize_columns(A))
