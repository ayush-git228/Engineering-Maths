# Cheatsheets

## Identities
- (AB)^T = B^T A^T
- (A^T A) invertible iff columns of A are linearly independent.
- Projection: P = A(A^T A)^{-1}A^T, P^2 = P, P^T = P.
- Normal equations: X^T X w = X^T y
- SVD: A = U Σ V^T, rank(A) = #non-zero singular values.

## Pitfalls
- Ill-conditioning: prefer `np.linalg.solve` to explicit inverse.
- Rank deficiency: add small `ridge*I` when solving normal equations.
- Normalization for cosine similarity is essential to avoid length bias.

## Exam quick notes
- Rank-nullity: dim(Null(A)) = n - rank(A)
- Symmetric matrices have real eigenvalues and orthogonal eigenvectors.
- PSD check: eigenvalues of (Q+Q^T)/2 are ≥ 0.