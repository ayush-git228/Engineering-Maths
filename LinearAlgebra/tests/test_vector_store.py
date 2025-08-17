import numpy as np
from agentkit.vector_store import VectorStore

def test_topk_ordering():
    vs = VectorStore()
    vs.add("a", np.array([1,0,0]))
    vs.add("b", np.array([0.9,0.1,0]))
    vs.add("c", np.array([0,1,0]))
    q = np.array([1,0,0])
    top = vs.topk(q, k=2)
    assert top[0][0] == "a"
    assert len(top) == 2