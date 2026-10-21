# Formal Problem Statement

Given a set of users \(U\), items \(I\), and an implicit interaction matrix

\[
Y \in \{0,1\}^{|U| \times |I|}
\]

learn a function

\[
f : U \times I \rightarrow \hat{y}
\]

that predicts unobserved interactions, where \(f\) is parameterized by a deep neural network combining embedding lookups with multi-layer perceptron (MLP) fusion.

# Core Hypothesis

A Neural Collaborative Filtering (NCF) model implemented from scratch using NumPy will converge to equivalent predictive performance as a PyTorch-optimized implementation, but with measurably longer training time and greater numerical sensitivity. This demonstrates that automatic differentiation (autograd) provides engineering efficiency without sacrificing theoretical correctness.

# Research Contribution

A side-by-side empirical comparison of:

- Manual backpropagation (NumPy implementation)
- Automatic differentiation (PyTorch implementation)

within a Neural Collaborative Filtering framework, integrated into a production recommendation system featuring a hybrid cold-start strategy.
