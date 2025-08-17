import numpy as np
from la_core.gaussian_elimination import gaussian_elimination

def test_gauss_solution_matches_numpy():
    np.random.seed(0)
    for _ in range(5):
        A = np.random.randn(4, 4)
        b = np.random.randn(4)
        x, _ = gaussian_elimination(A, b)
        np.testing.assert_allclose(A @ x, b, atol=1e-7)