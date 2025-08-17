# %% [markdown]
# Rank & Nullity utilities
# - rank, null_space basis, determinant, linear independence check functions
# - small-run examples for unit tests and agent checks

# %%
import numpy as np
from typing import Tuple

# %%
def matrix_rank(A: np.ndarray, tol: float = 1e-10) -> int:
    """
    Compute numerical rank using singular values.
    """
    s = np.linalg.svd(A, compute_uv=False)
    return int(np.sum(s > tol))

# %%
def null_space(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Return an orthonormal basis for null space of A as columns.
    If null space is {0}, returns array with shape (n, 0).
    """
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    r = int(np.sum(S > tol))
    # Vt shape (m,m) if A is (n,m) with full_matrices=False, Vt is (m,m) or (m,min(n,m))
    null = Vt[r:].T if r < Vt.shape[0] else np.zeros((A.shape[1], 0))
    return null

# %%
def determinant(A: np.ndarray) -> float:
    """
    Determinant (works for square A).
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("Determinant defined only for square matrices")
    return float(np.linalg.det(A))

# %%
def rank_nullity_check(A: np.ndarray, tol: float = 1e-10) -> Tuple[int, int]:
    """
    Returns (rank, nullity) for matrix A of shape (n, m).
    """
    r = matrix_rank(A, tol=tol)
    nullity = A.shape[1] - r
    return r, nullity

# %%
# Examples when run as script
if __name__ == "__main__":
    A = np.array([[1.,2.,3.],[2.,4.,6.],[1.,1.,1.]])
    r = matrix_rank(A)
    ns = null_space(A)
    det_2x2 = determinant(A[:2,:2])
    print("Matrix:\n", A)
    print("Rank:", r)
    print("Null-space basis (columns):\n", ns)
    print("Nullity:", ns.shape[1])
    print("Determinant of top-left 2x2:", det_2x2)
