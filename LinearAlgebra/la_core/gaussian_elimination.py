import numpy as np
from typing import Tuple

def gaussian_elimination(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve Ax=b via Gaussian elimination with partial pivoting.
    Returns (x, U) where U is the upper-triangular matrix after elimination.
    """
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = A.shape[0]
    for k in range(n - 1):
        i_max = k + np.argmax(np.abs(A[k:, k]))
        if A[i_max, k] == 0:
            continue
        if i_max != k:
            A[[k, i_max]] = A[[i_max, k]]
            b[[k, i_max]] = b[[i_max, k]]
        for i in range(k + 1, n):
            if A[k, k] == 0:
                continue
            m = A[i, k] / A[k, k]
            A[i, k:] -= m * A[k, k:]
            b[i]     -= m * b[k]
            
    x = np.zeros(n)
    for i in reversed(range(n)):
        s = b[i] - A[i, i+1:] @ x[i+1:]
        if abs(A[i, i]) < 1e-14:
            if abs(s) < 1e-12:
                x[i] = 0.0
            else:
                raise np.linalg.LinAlgError("Inconsistent system")
        else:
            x[i] = s / A[i, i]
    return x, A

if __name__ == "__main__":
    A = np.array([[3.,2.,-4.],[2.,3.,3.],[5.,-3.,1.]])
    b = np.array([3.,15.,14.])
    x, U = gaussian_elimination(A, b)
    print("x:", x)
    print("A@x:", A @ x)
