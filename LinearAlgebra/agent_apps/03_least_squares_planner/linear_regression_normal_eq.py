import numpy as np

def fit_normal_eq(X: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    d = X.shape[1]
    A = X.T @ X + ridge * np.eye(d)
    b = X.T @ y
    return np.linalg.solve(A, b)

if __name__ == "__main__":
    X = np.array([[1,1],[2,1],[1,3],[3,2],[4,3]], float)
    y = np.array([2,3,4,6,8], float)
    w = fit_normal_eq(X, y, ridge=0.01)
    print("w:", w)