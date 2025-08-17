import numpy as np
from la_core.lu_decomposition import lu_decomposition, lu_solve

def test_pa_equals_lu_and_solve():
    np.random.seed(1)
    A = np.random.randn(5, 5)
    b = np.random.randn(5)
    P, L, U = lu_decomposition(A)
    np.testing.assert_allclose(P @ A, L @ U, atol=1e-7)
    x = lu_solve(P, L, U, b)
    np.testing.assert_allclose(A @ x, b, atol=1e-7)