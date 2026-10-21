# Chapter 1 — Introduction

## Chapter Overview

This chapter introduces the research context, motivation, and scope of the present study, which investigates Neural Collaborative Filtering (NCF) through a combined theoretical, implementational, and systems-engineering lens. The chapter situates the work within the broader evolution of recommender systems, from classical collaborative filtering to neural approaches, and articulates the specific problem addressed by this thesis: the disconnect between the mathematical formulation of neural recommendation models and the reproducible, empirically validated engineering practice required to bring such models into a deployable system. The chapter concludes by stating the research questions, objectives, contributions, and scope that structure the remainder of the thesis.

## 1.1 Background

The task of recommending relevant items to users from large catalogues is a foundational problem in information filtering, arising in domains ranging from e-commerce to media streaming. As digital catalogues have grown from thousands to millions of items, manual curation has become infeasible, and algorithmic recommendation has become the primary mechanism through which users discover relevant content. Recommender systems research has, over three decades, progressed through several distinct methodological generations: neighbourhood-based collaborative filtering, latent-factor matrix factorization, and, more recently, neural network-based approaches that relax the linearity assumptions inherent in earlier models.

This progression is not merely of academic interest. Each generation of methods has addressed a specific representational limitation of its predecessor. Neighbourhood methods do not scale well to large, sparse interaction matrices. Matrix factorization (MF) methods scale considerably better by projecting users and items into a shared low-dimensional latent space, but the prediction function — an inner product of latent vectors — imposes a restrictive geometric structure on the space of representable preference orderings. Neural Collaborative Filtering, introduced by He et al. [1], was proposed specifically to relax this constraint by replacing the fixed inner product with a learned, potentially non-linear interaction function parameterised by a neural network.

## 1.2 Recommender Systems

A recommender system can be formally characterised as a function that estimates, for a given user and a given item, a score reflecting the likelihood or strength of the user's preference for that item, and that subsequently ranks items by this score to produce a recommendation list. Recommender systems are broadly classified according to the signal they exploit: **content-based systems**, which recommend items similar to those a user has previously engaged with, based on item attributes; **collaborative filtering systems**, which exploit patterns of interaction across a population of users without requiring explicit item attributes; and **hybrid systems**, which combine both signal types to mitigate the weaknesses of each approach in isolation.

The present study engages with all three categories. Its central research component is a collaborative-filtering model (NCF); its cold-start-handling component is a content-based genre-similarity model; and its production recommendation engine is an explicit hybrid of collaborative, content-based, and popularity-based signals (Chapter 9).

## 1.3 Collaborative Filtering

Collaborative filtering (CF) is founded on the assumption that users who have interacted similarly with items in the past will exhibit similar preferences in the future. Two principal families of CF methods exist. **Memory-based (neighbourhood) methods** compute similarities directly between users or between items using the raw interaction matrix, and produce predictions as similarity-weighted aggregates of known ratings. **Model-based methods** instead learn a compact parametric representation of users and items — most commonly through matrix factorization — from which predictions are generated. Model-based methods, particularly MF, became the dominant paradigm following their prominent use in the Netflix Prize competition, owing to their favourable scalability and their capacity to generalise beyond the sparsity of the observed interaction matrix.

A well-documented limitation of MF, formalised by He et al. [1], is that the inner product — used to combine user and item latent vectors into a predicted score — is a fixed, linear similarity function. This function cannot represent arbitrary preference orderings among items for a given user, a limitation that becomes material when latent dimensionality is constrained and that motivates the departure from purely linear interaction functions.

## 1.4 Neural Collaborative Filtering

Neural Collaborative Filtering generalises matrix factorization by replacing the fixed inner product with a learned function realised by a neural network operating over user and item embeddings. In its most general form, as proposed by He et al. [1], NCF fuses two complementary pathways: a Generalized Matrix Factorization (GMF) pathway, which generalises the inner product via an element-wise (Hadamard) product followed by a learned linear projection, and a Multi-Layer Perceptron (MLP) pathway, which learns non-linear interaction patterns through a sequence of fully connected layers with non-linear activations. This thesis investigates the theoretical formulation of the full NCF framework (Chapter 2) and implements and empirically validates the MLP-based interaction-function variant of this framework (Chapters 6–8), a distinction that is stated explicitly here and revisited with full transparency in Chapters 6, 7, and 14, so that the boundary between the general theory and what was actually implemented and evaluated in this project is never ambiguous.

## 1.5 Motivation for the Study

Three considerations motivate this study.

First, the pedagogical value of implementing a neural recommendation model **from first principles** — without reliance on an automatic differentiation framework — is substantial for demonstrating genuine mathematical understanding of forward propagation, backpropagation, and gradient-based optimisation, as opposed to competent use of a high-level deep learning library. A from-scratch implementation forces explicit engagement with the chain rule through embedding lookups, linear transformations, and non-linear activations, and with the numerical considerations (e.g., the simplified gradient of a sigmoid-BCE composition) that framework-provided automatic differentiation otherwise conceals.

Second, an implementation that is only theoretically plausible is of limited scientific value unless its correctness can be independently verified. This motivates two complementary verification strategies pursued in this work: analytical gradient verification against numerically estimated gradients (via central finite differences), and empirical parity assessment against an established deep learning framework (PyTorch) trained under matched configuration.

Third, recommendation research frequently terminates at the point of offline evaluation, leaving a gap between algorithmic contribution and the engineering practices — model serving, monitoring, containerisation, testing — required to operate such a model as part of a real system. This study explicitly extends the investigation beyond model training to include a production-oriented system architecture, while treating this engineering layer as _supporting_ rather than _supplanting_ the research contribution.

## 1.6 Problem Statement

Given a set of users $U$, a set of items $I$, and an implicit-feedback interaction matrix $Y \in \{0,1\}^{|U| \times |I|}$, the problem addressed by this thesis is threefold:

1. To formulate, implement from first principles, and analytically verify a neural interaction function $\hat{y}_{ui} = f(u, i \mid \Theta)$ that estimates the probability of user–item interaction, without reliance on automatic differentiation.
2. To assess whether this from-scratch implementation behaves consistently — in terms of training convergence and recommendation quality — with an architecturally equivalent implementation built on an established deep learning framework, and to characterise the computational trade-offs between the two.
3. To address the cold-start limitation inherent to collaborative filtering — the inability to generate meaningful recommendations for users or items with little or no interaction history — through a hybrid recommendation strategy, and to integrate the resulting system into a reproducible, monitored, production-oriented architecture.

## 1.7 Research Gap

The literature on Neural Collaborative Filtering, reviewed in depth in Chapter 3, is extensive with respect to model architecture and offline accuracy benchmarking, but comparatively sparse with respect to three specific concerns that this thesis addresses jointly:

- **Verification methodology.** Published NCF studies typically report predictive performance without disclosing or verifying the correctness of the underlying gradient computations at the implementation level, since virtually all rely on automatic differentiation frameworks in which gradient correctness is assumed rather than demonstrated.
- **Implementation parity.** Comparisons between hand-derived (from-scratch) and framework-based implementations of the same neural architecture, evaluated under matched hyperparameters and identical data pipelines, are rare in the accessible recommender-systems literature, despite their pedagogical and diagnostic value.
- **Research-to-production continuity.** Academic NCF studies rarely extend their evaluation into a documented, testable, observable production architecture; conversely, industrial system papers (e.g., Wide & Deep [11], YouTube DNN [12]) rarely expose the underlying model's mathematical derivation or verification methodology in comparable depth.

This thesis is positioned at the intersection of these three gaps, as elaborated further in Chapter 3.

## 1.8 Research Questions

The study is organised around the following research questions:

- **RQ1.** Does a from-scratch NumPy implementation of Neural Collaborative Filtering, with manually derived and analytically verified backpropagation, converge to a solution that is behaviourally consistent with an architecturally equivalent PyTorch implementation trained under matched hyperparameters?
- **RQ2.** What is the magnitude and nature of the computational trade-off (training time, convergence behaviour) between the from-scratch and framework-based implementations?
- **RQ3.** To what extent does a hybrid recommendation strategy — combining the trained NCF model with popularity-based and content-based signals — address the cold-start problem for new users?
- **RQ4.** Can the resulting recommendation model be integrated into a reproducible, tested, and observable production-oriented system architecture without the engineering layer compromising the traceability of the underlying research claims?

## 1.9 Research Objectives

Corresponding to the research questions above, the objectives of this study are:

1. To derive the mathematical formulation of the implemented neural interaction function, including the forward propagation, loss function, and backpropagation equations (addresses RQ1).
2. To implement this formulation from first principles in NumPy and verify the correctness of the resulting gradients using numerical differentiation (addresses RQ1).
3. To implement an architecturally equivalent model in PyTorch under matched configuration and to empirically compare training convergence, recommendation quality, and computational performance between the two implementations (addresses RQ1, RQ2).
4. To design and implement a hybrid recommendation strategy that blends collaborative, content-based, and popularity-based signals as a function of user interaction history, and to empirically characterise its behaviour across cold-start, partially warm, and fully warm user states (addresses RQ3).
5. To integrate the resulting recommendation components into a documented, containerised, tested, and monitored system architecture, while maintaining a clear separation between research and production code (addresses RQ4).

## 1.10 Contributions of the Study

The contributions of this thesis, each traceable to specific artefacts examined during the project audit preceding this thesis (see Chapter 4 for the full evidence mapping), are:

- A mathematically documented, from-scratch NumPy implementation of an NCF-style neural interaction function, comprising custom embedding layers, linear layers, activation functions, a Binary Cross-Entropy loss with a numerically stable combined sigmoid-BCE gradient, and both SGD-with-momentum and Adam optimisers implemented without an automatic differentiation framework.
- Gradient verification of the implementation via central finite-difference numerical differentiation, providing direct evidence — rather than an assumption inherited from a framework — that the manually derived backpropagation is correct, within the scope tested.
- An architecturally matched PyTorch implementation, trained under a shared data pipeline and matched hyperparameters, enabling a controlled empirical comparison of convergence behaviour, recommendation quality, and training time between hand-derived and framework-based optimisation.
- A hybrid recommendation engine addressing the cold-start problem through an interaction-count-dependent blending of collaborative, content-based, and popularity signals, with its qualitative behaviour empirically demonstrated across representative user states.
- A production-oriented system architecture — comprising a FastAPI backend, an isolated model-serving component, a Next.js frontend, containerised deployment, and Prometheus/Grafana-based observability — that operationalises the research artefacts while preserving a documented boundary between research and production code.

## 1.11 Scope of the Study

This study is scoped as follows. The dataset used throughout is MovieLens-1M, a widely used benchmark for implicit-feedback recommendation research; no additional datasets are evaluated. The neural architecture investigated is the MLP-based interaction-function variant of NCF, applying a single concatenated embedding representation followed by a feedforward network; the full dual-pathway GMF+MLP NeuMF fusion architecture is presented at the theoretical level (Chapter 2) but is **not** the architecture implemented and empirically evaluated in this work — this distinction is maintained consistently throughout the thesis. No hyperparameter search or formal ablation study was conducted; the scratch-versus-PyTorch comparison is a parity and efficiency study at a fixed, shared configuration. The production architecture is evaluated on the basis of its documented design, implemented code, and available test suite; it is not evaluated under production load, and no formal scalability benchmarking was performed. Kubernetes and Terraform artefacts present in the repository are explicitly reference scaffolding and are discussed as proposed future infrastructure rather than as deployed systems.

## 1.12 Thesis Organisation

The remainder of this thesis is organised as follows. Chapter 2 establishes the theoretical foundations of collaborative filtering, neural recommendation, and the full NCF framework, including its GMF and MLP pathways and the mathematics of backpropagation. Chapter 3 provides a critical literature review situating this work within the evolution of recommender systems research and articulates the research gap in greater depth. Chapter 4 details the research methodology, experimental protocol, and reproducibility strategy. Chapter 5 describes the MovieLens-1M dataset and its preprocessing. Chapter 6 presents the mathematical formulation and from-scratch implementation of the evaluated NCF variant, including gradient verification. Chapter 7 presents the PyTorch implementation and the empirical assessment of parity between the two implementations. Chapter 8 reports the experimental evaluation and benchmarking results. Chapter 9 presents the hybrid recommendation and cold-start strategy. Chapters 10 through 12 describe the production architecture, deployment and observability infrastructure, and testing methodology, respectively. Chapter 13 interprets the findings and answers the research questions. Chapter 14 discusses limitations and threats to validity. Chapter 15 concludes the thesis and identifies directions for future research.

## Chapter Summary

This chapter has situated the present study within the trajectory of recommender systems research, from neighbourhood-based collaborative filtering through matrix factorization to Neural Collaborative Filtering. It has articulated a research problem spanning three dimensions — mathematically verified from-scratch implementation, empirical framework parity, and production-oriented integration — and has stated the corresponding research questions, objectives, contributions, and scope. A central methodological commitment introduced here, and maintained throughout the thesis, is the explicit separation between the theoretical NCF framework (which includes a GMF pathway) and the specific MLP-based variant that was implemented and empirically evaluated in this work.
