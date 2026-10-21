# Neural Collaborative Filtering (NCF) — Literature Review Notes

**Reference:** He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T.-S. (2017). _Neural Collaborative Filtering._

---

## 1. What is the Key Weakness of Matrix Factorization (MF) that NCF Addresses?

The primary weakness of traditional **Matrix Factorization (MF)** is its reliance on the **inner product** (a linear kernel) as the interaction function between user and item latent vectors.

### Why is this a problem?

The inner product performs a relatively simple linear interaction between latent features and may fail to capture:

- Complex user preferences
- Non-linear user-item relationships
- High-order feature interactions

As a result, recommendation quality can degrade, especially when latent dimensions are limited.

### NCF Solution

Neural Collaborative Filtering replaces the fixed inner product with a **learnable neural architecture**, enabling the model to capture richer and more complex interaction patterns.

---

## 2. What Does the Interaction Function φ Generalize?

The interaction function **φ (phi)** generalizes the interaction mechanism used in **traditional Matrix Factorization (MF)**.

The paper introduces **Generalized Matrix Factorization (GMF)** and proves that MF is a special case of GMF.

### Conditions Under Which GMF Becomes MF

If:

- The activation function is the **identity function**
- The output layer weight vector **h** is a vector of ones

Then:

\[
\phi(\mathbf{p_u}, \mathbf{q_i})
=
\mathbf{p_u} \odot \mathbf{q_i}
\]

and GMF reduces exactly to standard Matrix Factorization.

### Key Insight

MF is not discarded; instead, it is generalized within a neural framework.

---

## 3. What Layer Sizes Are Used in the Paper's Experiments?

For the **Multi-Layer Perceptron (MLP)** component, the paper adopts a **tower architecture** where each hidden layer has half the size of the previous layer.

### General Structure

```text
Input Layer
      ↓
Hidden Layer 1
      ↓
Hidden Layer 2
      ↓
Hidden Layer 3
      ↓
Prediction Layer
```

### Example Configuration

For:

- Predictive Factors = **32**
- Concatenated Embedding Size = **64**

The hidden layers are:

```text
32 → 16 → 8
```

### Other Configurations Evaluated

| Predictive Factors | Hidden Layer Sizes |
| ------------------ | ------------------ |
| 16                 | 16 → 8 → 4         |
| 32                 | 32 → 16 → 8        |
| 64                 | 64 → 32 → 16       |
| 128                | 128 → 64 → 32      |

This progressively compressed structure helps learn increasingly abstract interaction patterns.

---

## 4. What Sampling Ratio Is Used?

The authors investigate negative sampling ratios ranging from:

```text
1 : 1  to  10 : 1
```

where:

- Positive Instance = observed interaction
- Negative Instance = unobserved interaction

### Default Ratio Used

The paper uses:

```text
4 Negative Samples : 1 Positive Sample
```

or

```text
4 : 1
```

for most baseline comparisons.

### Observation

Experimental results show that performance is typically best when the ratio is between:

```text
3 : 1 and 6 : 1
```

---

## 5. What Hit@10 Score Does the Best Model Achieve on MovieLens 1M?

The best-performing model in the paper is:

### NeuMF (Neural Matrix Factorization)

NeuMF combines:

- Generalized Matrix Factorization (GMF)
- Multi-Layer Perceptron (MLP)

into a unified architecture.

### Performance on MovieLens 1M

| Model     | Hit@10      |
| --------- | ----------- |
| GMF       | Lower       |
| MLP       | Lower       |
| eALS      | Lower       |
| BPR       | Lower       |
| **NeuMF** | **≈ 0.705** |

### Result

\[
\text{Hit@10} \approx 0.705
\]

or approximately:

```text
70.5%
```

This demonstrates that NeuMF significantly outperforms both traditional collaborative filtering methods and its individual neural components.

---

# Key Takeaways

- Matrix Factorization is limited by its linear inner-product interaction.
- NCF introduces neural networks to model non-linear user-item interactions.
- GMF generalizes classical Matrix Factorization.
- The MLP architecture follows a decreasing tower structure (e.g., 32 → 16 → 8).
- A 4:1 negative sampling ratio is used as the default setting.
- NeuMF achieves a Hit@10 score of approximately **70.5%** on the MovieLens 1M dataset.
