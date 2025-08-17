import numpy as np
from typing import List

class LinearScorer:
    """
    s = W x  (linear model). W learned via normal equations with optional ridge.
    """
    def __init__(self, d_in: int, d_out: int, ridge: float = 0.0):
        self.W = np.zeros((d_out, d_in))
        self.ridge = ridge

    def fit(self, X: np.ndarray, Y: np.ndarray):
        # X: [N,d_in], Y: [N,d_out]
        d = X.shape[1]
        A = X.T @ X + self.ridge * np.eye(d)
        B = X.T @ Y
        self.W = np.linalg.solve(A, B).T

    def score(self, x: np.ndarray) -> np.ndarray:
        return (self.W @ x.reshape(-1,1)).ravel()

def choose_actions(scorer: LinearScorer, x: np.ndarray, actions: List[str], top_k=2) -> List[str]:
    s = scorer.score(x)
    idx = np.argsort(-s)[:top_k]
    return [actions[i] for i in idx]