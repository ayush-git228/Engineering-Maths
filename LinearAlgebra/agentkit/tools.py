import numpy as np
from typing import Any, Dict

class Tool:
    def __init__(self, name: str, fn, desc: str):
        self.name = name
        self.fn = fn
        self.desc = desc
    def __call__(self, **kwargs) -> Any:
        return self.fn(**kwargs)

def calculator(expression: str) -> Dict[str, Any]:
    import re
    if not re.fullmatch(r"[0-9\\.\+\\-\\*\\/\\(\\) ]+", expression):
        raise ValueError("Unsupported characters")
    return {"result": eval(expression, {"__builtins__": {}}, {})}

def dot_product(a: list[float], b: list[float]) -> Dict[str, Any]:
    va, vb = np.array(a, float), np.array(b, float)
    if va.shape != vb.shape: raise ValueError("shape mismatch")
    return {"result": float(va @ vb)}

TOOLS = [
    Tool("calculator", calculator, "Safe arithmetic (+,-,*,/)."),
    Tool("dot_product", dot_product, "Compute vector dot product."),
]