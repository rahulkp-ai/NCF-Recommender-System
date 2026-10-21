# NCF Mathematical Formulation

## Problem Definition

- Users: U = {1, ..., M}, M = 6040 (MovieLens 1M)
- Items: I = {1, ..., N}, N = 3706
- Interaction matrix: Y ∈ {0,1}^(M×N)
- Sparsity: ~95% unobserved

## Hypothesis

A NumPy NCF and PyTorch NCF trained on identical data with identical
hyperparameters will converge to equivalent Hit@10 within ±2%,
but PyTorch will train 10-50x faster due to vectorised autograd.

## Model: f(u, i | Θ) = σ(MLP([p_u ⊕ q_i]))

- p_u = P[u,:], P ∈ R^(M×k) — user embedding
- q_i = Q[i,:], Q ∈ R^(N×k) — item embedding
- Concat: z_0 = [p_u; q_i] ∈ R^(2k)
- Hidden: z*l = ReLU(W_l z*{l-1} + b_l)
- Output: ŷ = σ(w^T z_L + b)

## Loss: Binary Cross Entropy with Negative Sampling

L = -∑*{(u,i)∈Y+} log(ŷ*{ui}) - ∑*{(u,j)∈Y-} log(1 - ŷ*{uj})
Negative sampling ratio: 4 negatives per positive

## Evaluation Protocol

- Leave-one-out: hold last interaction per user as test
- Sample 99 random negatives per test interaction
- Rank all 100, compute: Hit@10, NDCG@10
