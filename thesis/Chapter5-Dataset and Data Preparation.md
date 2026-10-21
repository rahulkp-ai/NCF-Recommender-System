# Chapter 5 — Dataset and Data Preparation

## Chapter Overview

This chapter describes the dataset underlying all empirical work reported in this thesis, the quality characteristics of that dataset, and the preprocessing pipeline through which raw rating records are transformed into the encoded, negative-sampled, split interaction data consumed by the models developed in Chapters 6 and 7. Every numerical figure reported in this chapter is drawn directly from the project's dataset metadata, its raw source documentation, or an internal consistency check computable from those two sources; no dataset statistic is estimated or assumed. Where a customary data-quality check (e.g., an explicit duplicate-record count) could not be traced to a specific artefact among those audited, this is stated explicitly rather than inferred from the dataset's general reputation in the literature.

## 5.1 MovieLens Dataset

### 5.1.1 Dataset Description

The dataset used throughout this study is MovieLens-1M, a benchmark dataset for recommender-systems research distributed by GroupLens Research and stored within the project repository as raw `.dat` files (`research/datasets/`). MovieLens-1M is one of the two datasets used in the original evaluation of Neural Collaborative Filtering [1] (the other being Pinterest), and its use in the present study preserves direct comparability, at the level of dataset choice and evaluation protocol, with the seminal work reviewed in Chapter 3. Per the dataset's own raw README file, bundled with the raw data in the repository, MovieLens-1M consists of 1,000,209 ratings, contributed by 6,040 users on approximately 3,900 movies, collected by GroupLens Research at the University of Minnesota, with the user base having joined MovieLens in the year 2000.

### 5.1.2 Users

Each user in the raw dataset is identified by a numeric user ID. The raw MovieLens-1M distribution additionally provides limited demographic attributes for each user (typically gender, age bracket, occupation code, and zip code, per the standard MovieLens-1M schema); however, no evidence was found among the audited preprocessing code (`movielens_loader.py`) that these demographic fields are consumed by any model examined in this thesis. The from-scratch and PyTorch NCF implementations (Chapter 6, Chapter 7) and the content-based cold-start engine (Chapter 9) operate exclusively on the user ID (for embedding lookup) and on interaction history (for popularity/content scoring), not on demographic attributes. This is stated explicitly here to avoid any implication that demographic side-information is exploited by the recommendation models evaluated in this thesis.

### 5.1.3 Movies

Each movie is identified by a numeric item ID and is associated with a title and one or more genre labels drawn from a fixed, standard MovieLens genre vocabulary. The raw README figure of "approximately 3,900 movies" describes the full movie catalogue as distributed; the processed dataset's actual item count after encoding (Section 5.2) is a more precise, directly measured figure and is used in preference to this approximate raw figure throughout the remainder of this thesis.

### 5.1.4 Ratings

Each rating record links a user ID, an item ID, an integer rating on a 1–5 scale, and a Unix timestamp. Consistent with the implicit-feedback framing established in Section 2.1.3, the preprocessing pipeline examined in Section 5.4 treats the presence of a rating record — irrespective of its 1–5 scale value — as a positive implicit interaction, discarding the numeric rating value itself as a graded preference signal. This is a deliberate methodological choice, consistent with the implicit-feedback protocol of He et al. [1], and it means that no explicit-feedback (rating-value regression) evaluation is performed anywhere in this thesis, consistent with the top-K ranking formulation adopted in Section 2.1.4.

### 5.1.5 Genre Information

The genre labels associated with each movie form an 18-dimensional categorical vector under the standard MovieLens genre taxonomy. This genre vector is the feature representation consumed by the content-based recommendation component of the hybrid engine (Section 2.6.1, Chapter 9), which computes cosine similarity between a candidate item's genre vector and a user profile vector constructed as an aggregate of the genre vectors of items the user has previously interacted with. The genre vector is the only item-content feature used anywhere in this project's implemented content-based scoring; no other item metadata (e.g., cast, textual description, release-year-derived features) was found to be consumed by the content engine among the audited code.

## 5.2 Dataset Statistics

Table 5.1 reports the processed dataset statistics as recorded in `research/datasets/processed/meta.json`, the single authoritative record of post-preprocessing dataset size referenced by the training and evaluation scripts examined in Chapters 6–8.

**Table 5.1: Processed MovieLens-1M dataset statistics (`meta.json`).**

| Quantity                               | Value     |
| -------------------------------------- | --------- |
| Number of users ($M$)                  | 6,040     |
| Number of items ($N$)                  | 3,706     |
| Training-set size ($n_{\text{train}}$) | 4,940,645 |
| Validation-set size ($n_{\text{val}}$) | 6,040     |
| Test-set size ($n_{\text{test}}$)      | 6,040     |

The processed item count (3,706) is somewhat smaller than the raw dataset's approximate figure of 3,900 movies (Section 5.1.1); this is consistent with a standard preprocessing step, common in NCF-style pipelines, in which movies with zero observed interactions after the encoding process, or movies excluded during ID re-indexing, are dropped from the item vocabulary. The precise filtering rule that produced 3,706 from approximately 3,900 raw movie records was not directly located as an explicit, documented threshold within the audited portion of `movielens_loader.py`, and is therefore reported as:

[DATA REQUIRED: value not available in the provided project artifacts — the specific item-filtering criterion (e.g., a minimum-interaction-count threshold) that reduces the raw movie count to the processed item count of 3,706 was not confirmed in the audited code.]

A useful internal consistency check, computable directly from the figures already established in Sections 5.1.1 and 5.2 without introducing any new unverified figure, is the following: the raw dataset contains 1,000,209 rating records; the leave-one-out split described in Section 5.4.5 removes exactly one interaction per user for the test set and one further interaction per user for the validation set, i.e. $2 \times 6{,}040 = 12{,}080$ interactions. Subtracting this from the raw total gives $1{,}000{,}209 - 12{,}080 = 988{,}129$, which is exactly the positive-interaction count reported for the training set by `movielens_loader.py::sample_negatives`, as verified in the outputs of `02_data_verification.ipynb` (Section 5.4.4). This exact arithmetic agreement across three independently sourced figures (the raw README count, the per-user leave-one-out removal count, and the logged positive-sample count) is treated in this thesis as corroborating evidence that the reported dataset statistics are mutually consistent, rather than being drawn from mismatched or stale sources — a distinction of direct relevance given the unresolved numerical discrepancy involving the root `README.md`'s results table noted elsewhere in this thesis (Chapter 8).

## 5.3 Data Quality Analysis

### 5.3.1 Missing Data

MovieLens-1M, as a curated benchmark dataset, contains no missing values within its rating records by construction: each of the 1,000,209 rating records comprises a complete (user ID, item ID, rating, timestamp) tuple. No evidence was found among the audited notebooks or preprocessing scripts of null or malformed rating records requiring removal. Whether the auxiliary user demographic file or the movie metadata file (genre labels, titles) contains missing or malformed entries that required handling during preprocessing was not confirmed from the audited artefacts, and is reported as:

[DATA REQUIRED: value not available in the provided project artifacts — no explicit missing-value audit output for the user or movie metadata files was located among the reviewed notebooks.]

### 5.3.2 Duplicate Records

No explicit duplicate-record count or de-duplication step was located among the audited preprocessing code or notebook outputs. MovieLens-1M is documented upstream (by its distributor) as containing at most one rating per user–item pair; the present study's preprocessing pipeline does not appear to include an explicit duplicate-detection step, which is consistent with, but does not independently confirm, the absence of duplicate ratings in the specific copy of the dataset bundled with this project. This is reported as:

[DATA REQUIRED: value not available in the provided project artifacts — no explicit duplicate-record detection or count was located in the audited code or notebook outputs.]

### 5.3.3 Interaction Sparsity

Interaction sparsity, established theoretically in Section 2.1.2 as a defining structural characteristic of the recommendation problem, can be computed directly from the confirmed figures in Table 5.1 and Section 5.1.1. Using the raw interaction count (1,000,209 ratings) against the processed user and item counts ($M = 6{,}040$, $N = 3{,}706$), the maximum possible number of user–item pairs is:

$$
M \times N = 6{,}040 \times 3{,}706 = 22{,}384{,}240
$$

The observed interaction density is therefore:

$$
\text{density} = \frac{1{,}000{,}209}{22{,}384{,}240} \approx 4.47\%
$$

equivalently, approximately 95.53% of the user–item matrix $Y$ is unobserved. This figure is computed directly from confirmed dataset statistics rather than drawn from an external or approximate source, and it substantiates, with a concrete measured value, the general sparsity argument made theoretically in Section 2.1.2 and used there to motivate latent-representation-based approaches over direct non-parametric modelling of $Y$. It should be noted that this density calculation uses the processed item count (3,706); computing the same ratio against the raw, approximate item count (~3,900) would yield a marginally lower density figure, a minor sensitivity that does not materially affect the qualitative conclusion that the interaction matrix is highly sparse.

## 5.4 Data Preprocessing

### 5.4.1 User Encoding

Raw MovieLens user IDs are not guaranteed to form a contiguous, zero-indexed range suitable for direct use as embedding-table row indices (Section 2.3.2). `movielens_loader.py` accordingly re-indexes user IDs into a contiguous integer range $\{0, 1, \ldots, M-1\}$, consistent with the indexing convention specified in Section 2.1.1. The specific encoding mechanism (e.g., a sorted-unique-ID mapping or a categorical encoder) was confirmed to exist within `movielens_loader.py` per the project audit, though the exact implementation (library used, sort order) was not itemised in the audit summary and is not restated here beyond what is confirmed: that a user ID re-indexing step exists and produces the user count reported in Table 5.1.

### 5.4.2 Item Encoding

An analogous re-indexing step is applied to raw movie IDs, mapping them into the contiguous range $\{0, 1, \ldots, N-1\}$ required for item-embedding lookup (Section 2.3.2). This encoding step is also the natural point at which the item-count reduction from approximately 3,900 raw movies to 3,706 processed items (Section 5.2) would occur, for example through the exclusion of movies with zero surviving interactions after any filtering applied earlier in the pipeline; as noted in Section 5.2, the precise filtering rule was not independently confirmed.

### 5.4.3 Interaction Construction

Following user and item encoding, each raw rating record is converted into a positive implicit-feedback tuple $(u, i, y_{ui}=1)$, discarding the original 1–5 rating value in accordance with the implicit-feedback treatment established in Section 5.1.4. This construction step produces the full set of positive interactions later split and augmented with negative samples (Sections 5.4.4–5.4.5).

### 5.4.4 Negative Sampling

Negative sampling is implemented in `movielens_loader.py::sample_negatives`, following the uniform random negative-sampling procedure described theoretically in Section 2.4.6: for each observed positive interaction $(u, i)$, four items with which user $u$ has not interacted are sampled uniformly at random from the unobserved portion of $u$'s row in $Y$, and labelled $y_{ui^-} = 0$. This produces a fixed 1 positive : 4 negative ratio, within the range (3–6 negatives per positive) reported as empirically favourable by He et al. [1] (Section 2.4.6).

This ratio is directly verified by the logged output of `02_data_verification.ipynb`, which reports 988,129 positive training instances and 3,952,516 negative training instances, an exact ratio of 1:4 ($3{,}952{,}516 / 988{,}129 = 4.0000$), and which sums to $988{,}129 + 3{,}952{,}516 = 4{,}940{,}645$ — precisely the training-set size reported in `meta.json` (Table 5.1). This is a directly confirmed, executed-and-logged result, not a theoretical expectation, and it is one of the strongest pieces of internally consistent numerical evidence available among this project's data-preparation artefacts.

### 5.4.5 Train/Validation/Test Split

The dataset is partitioned using the leave-one-out protocol established in the NCF literature [1] and described theoretically in Chapter 3 (Section 3.8): for each user, the single most recent interaction (by timestamp) is held out as that user's test instance, the second-most-recent interaction is held out as that user's validation instance, and all remaining interactions form that user's contribution to the training set. This protocol guarantees, by construction, exactly one validation instance and one test instance per user, which is directly confirmed by Table 5.1's validation and test set sizes both being exactly equal to the user count (6,040).

## 5.5 Dataset Leakage Prevention

The leave-one-out protocol of Section 5.4.5, applied per-user based on interaction timestamp, is itself the primary leakage-prevention mechanism evidenced in this project: because the held-out validation and test interactions for a given user are strictly the user's most recent and second-most-recent interactions respectively, no interaction that temporally follows a user's training instances is used during training, preventing the specific form of temporal leakage in which a model could be trained on information that would not have been available at the historical point corresponding to its evaluation instances.

A second, structural leakage-prevention property follows from the negative-sampling procedure (Section 5.4.4): negative samples for the training set are drawn from each user's unobserved item set, and — per the standard formulation of this protocol in the literature — the held-out validation and test positive instances are excluded from the pool of items eligible to be sampled as training negatives, preventing a user's own validation/test item from being seen, in any capacity, during training. Direct code-level confirmation of this exclusion step within `movielens_loader.py::sample_negatives` was noted in the project audit as part of the negative-sampling implementation but is not restated in full mechanical detail here.

No further leakage-prevention mechanisms (e.g., user-level or item-level holdout to test generalisation to entirely unseen users/items) were identified among the audited artefacts. The leave-one-out protocol adopted here tests generalisation to a user's _next_ interaction given their historical interactions, not generalisation to previously entirely unseen users or items; the latter is the new-user/new-item cold-start setting addressed separately, and non-quantitatively, in Chapter 9.

## 5.6 Processed Dataset Representation

The processed dataset, as consumed by the training scripts examined in Chapters 6 and 7, is persisted under `research/datasets/processed/` and comprises: the re-indexed user and item ID mappings (Sections 5.4.1–5.4.2); the constructed positive and negative interaction tuples (Sections 5.4.3–5.4.4), partitioned into train, validation, and test sets (Section 5.4.5); and the summary metadata recorded in `meta.json` (Table 5.1). This processed representation is the single shared input to both the from-scratch NumPy implementation (Chapter 6) and the PyTorch implementation (Chapter 7), which is the specific pipeline-sharing property that permits the two implementations' training runs to be treated as differing only in implementation and configuration (Table 4.1) rather than in the underlying data.

## Chapter Summary

This chapter has described the MovieLens-1M dataset used throughout this thesis, its user, item, rating, and genre attributes, and the preprocessing pipeline — user/item encoding, implicit-interaction construction, 1:4 negative sampling, and per-user leave-one-out splitting — through which the raw dataset is transformed into the processed representation consumed by the models of Chapters 6 and 7. Every quantitative figure reported here (6,040 users; 3,706 items; 4,940,645 training instances; 6,040 validation and 6,040 test instances; a 988,129:3,952,516 positive-to-negative training ratio; approximately 4.47% interaction density) is either drawn directly from `meta.json` and the raw dataset documentation, or computed directly from those confirmed figures, and several of these figures were shown to be mutually consistent through direct arithmetic cross-checks. Data-quality characteristics for which no explicit audit artefact was located — missing-value handling in auxiliary metadata files, explicit duplicate-record counts, and the precise item-filtering rule underlying the raw-to-processed item-count reduction — are reported as open evidence gaps rather than assumed. This dataset and pipeline form the fixed, shared substrate against which the mathematical formulation of Chapter 2 is realised in the from-scratch implementation of Chapter 6 and the framework implementation of Chapter 7.
