import numpy as np
from typing import List, Dict

def bag_of_words_matrix(docs: List[str], vocab: List[str]) -> np.ndarray:
    V = {w:i for i, w in enumerate(vocab)}
    X = np.zeros((len(docs), len(vocab)))
    for i, d in enumerate(docs):
        for w in d.lower().split():
            if w in V:
                X[i, V[w]] += 1
    return X

def low_rank_approx(X: np.ndarray, k: int) -> np.ndarray:
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    k = min(k, len(S))
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

def summarize(docs: List[str], k: int = 2) -> List[str]:
    vocab = sorted({w for d in docs for w in d.lower().split()})
    X = bag_of_words_matrix(docs, vocab)
    Xk = low_rank_approx(X, k)
    scores = Xk.sum(axis=1)
    idx = np.argsort(-scores)[:max(1, min(k, len(docs)))]
    return [docs[i] for i in idx]

if __name__ == "__main__":
    docs = open("data/wiki_tiny_corpus.txt","r",encoding="utf-8").read().strip().splitlines()
    for s in summarize(docs, k=3):
        print("-", s)