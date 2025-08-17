import numpy as np
from typing import List, Tuple

class VectorStore:
    """
    In-memory vector DB (cosine similarity, top-k).
    Stores L2-normalized vectors.
    """
    def __init__(self, dim: int | None = None):
        self.ids: List[str] = []
        self.vecs: List[np.ndarray] = []
        self.dim = dim

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v if n == 0 else (v / n)

    def add(self, doc_id: str, v: np.ndarray) -> None:
        if self.dim is None: self.dim = v.shape[0]
        if v.shape[0] != self.dim: raise ValueError("dimension mismatch")
        self.ids.append(doc_id)
        self.vecs.append(self._normalize(v))

    def topk(self, q: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
        if not self.vecs: return []
        qn = self._normalize(q)
        M = np.vstack(self.vecs)        # [N, d]
        sims = M @ qn                   # cosine since unit vectors
        idx = np.argsort(-sims)[:k]
        return [(self.ids[i], float(sims[i])) for i in idx]

if __name__ == "__main__":
    vs = VectorStore()
    vs.add("la",  np.array([0.9,0.1,0.0]))
    vs.add("prob",np.array([0.1,0.9,0.0]))
    vs.add("agent",np.array([0.3,0.3,0.9]))
    print("top-2:", vs.topk(np.array([0.85,0.15,0.0]), k=2))
