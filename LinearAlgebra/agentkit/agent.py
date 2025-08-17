import numpy as np
from typing import Dict, Callable
from .planner import LinearScorer, choose_actions
from .tools import TOOLS

class Agent:
    """
    Deterministic agent: perceive (x) → score tools → execute top-k → return outputs.
    """
    def __init__(self, feature_dim: int, top_k: int = 2, ridge: float = 0.0):
        self.scorer = LinearScorer(d_in=feature_dim, d_out=len(TOOLS), ridge=ridge)
        self.actions = [t.name for t in TOOLS]
        self.tools_map: Dict[str, Callable] = {t.name: t for t in TOOLS}
        self.top_k = top_k

    def train(self, X: np.ndarray, Y_pref: np.ndarray):
        self.scorer.fit(X, Y_pref)

    def act(self, x: np.ndarray, tool_kwargs: Dict[str, Dict] | None = None):
        chosen = choose_actions(self.scorer, x, self.actions, top_k=self.top_k)
        outputs = {}
        for name in chosen:
            kw = {} if tool_kwargs is None else tool_kwargs.get(name, {})
            outputs[name] = self.tools_map[name](**kw)
        return {"chosen": chosen, "outputs": outputs}

if __name__ == "__main__":
    np.random.seed(0)
    agent = Agent(feature_dim=3, top_k=1)
    X = np.random.randn(20,3)
    Y = np.vstack([2*X[:,0] + 0.1, 2*X[:,1] + 0.1]).T
    agent.train(X, Y)
    res = agent.act(np.array([2.0, 0.1, 0.0]), {"calculator":{"expression":"2*(3+4)"}})
    print(res)