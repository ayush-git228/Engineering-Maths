import numpy as np
from typing import Tuple

def lu_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PA = LU (partial pivoting). Returns P, L, U.
    """
    A = A.astype(float).copy()
    n = A.shape[0]
    P = np.eye(n)
    L = np.zeros_like(A)
    U = A.copy()
    for k in range(n):
        i_max = k + np.argmax(np.abs(U[k:, k]))
        if abs(U[i_max, k]) < 1e-14:
            L[k, k] = 1.0
            continue
        if i_max != k:
            U[[k, i_max]] = U[[i_max, k]]
            P[[k, i_max]] = P[[i_max, k]]
            L[[k, i_max], :k] = L[[i_max, k], :k]
        L[k, k] = 1.0
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
            U[i, k] = 0.0
    return P, L, U

def lu_solve(P: np.ndarray, L: np.ndarray, U: np.ndarray, b: np.ndarray) -> np.ndarray:
    Pb = P @ b
    y = np.zeros_like(b, dtype=float)
    for i in range(L.shape[0]):
        y[i] = Pb[i] - L[i, :i] @ y[:i]
    x = np.zeros_like(b, dtype=float)
    for i in reversed(range(U.shape[0])):
        x[i] = (y[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]
    return x

if __name__ == "__main__":
    A = np.array([[4.,3.],[6.,3.]])
    b = np.array([10.,12.])
    P,L,U = lu_decomposition(A)
    x = lu_solve(P,L,U,b)
    print("x:", x)