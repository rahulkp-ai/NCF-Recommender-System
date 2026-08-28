# IEEE Publication Source & Compilation Guide

This directory contains the source files, assets, and build instructions for the Neural Collaborative Filtering (NCF) research paper. The paper details the architecture, PyTorch benchmarking, and production microservice implementation of the hybrid recommender system.

This publication is referenced in the portfolio's **Publications** section and linked within `docs/user-guide/README.md`.

---

## 🛠️ Build Requirements

To compile the Markdown paper into the standard IEEE two-column PDF format, ensure you have the following tools installed:

- **[Pandoc](https://pandoc.org/)** ($\ge 2.12$)
- **pdfLaTeX** (via TeX Live, MacTeX, or MikTeX)
- **IEEEtran LaTeX Package** (`IEEEtran.cls`)

---

## Compilation Command

Run the following command from this directory to generate `paper.pdf`:

```bash
pandoc paper.md \
  -V documentclass=IEEEtran \
  -V classoption=conference \
  --pdf-engine=pdflatex \
  -o paper.pdf
```
