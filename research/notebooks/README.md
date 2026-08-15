# `research/notebooks/`

Numbered sequence telling the research story in order — read/run
top to bottom:

1. `01_ncf_formulation.ipynb` — mathematical formulation (problem
   definition, embeddings, forward pass, loss, gradients, hypothesis,
   evaluation protocol)
2. `02_data_verification.ipynb` — MovieLens-1M data sanity checks
3. `03_model_parity_verification.ipynb` — confirms the NumPy and
   PyTorch implementations agree on identical inputs before the full
   training comparison
4. `04_pytorch_comparison.ipynb` — loss curves, Hit@10 curves, and
   training-time bar charts (writes `../evaluation/figures/thesis/*`)
5. `05_hybrid_test.ipynb` — exercises the hybrid engine (collaborative +
   content + popularity) end-to-end

`figures/` — supporting diagrams referenced by the notebooks
(autograd computation graphs, the hybrid engine's architecture, etc.)
