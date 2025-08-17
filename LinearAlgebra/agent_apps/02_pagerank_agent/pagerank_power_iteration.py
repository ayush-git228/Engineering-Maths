import numpy as np

def stochasticize(A: np.ndarray, damping=0.85) -> np.ndarray:
    n = A.shape[0]
    col_sums = A.sum(axis=0, keepdims=True)
    M = np.where(col_sums == 0, 1.0/n, A / np.where(col_sums == 0, 1, col_sums))
    G = damping * M + (1 - damping) * (np.ones((n, n)) / n)
    return G

def power_iteration(P: np.ndarray, iters=200, tol=1e-12) -> np.ndarray:
    n = P.shape[0]
    x = np.ones(n) / n
    for _ in range(iters):
        x_new = P @ x
        if np.linalg.norm(x_new - x, 1) < tol:
            x = x_new
            break
        x = x_new
    return x / x.sum()

if __name__ == "__main__":
    A = np.array([[0,1,1],[1,0,0],[1,1,0]], float)
    P = stochasticize(A, damping=0.85)
    scores = power_iteration(P)
    print("scores:", scores)