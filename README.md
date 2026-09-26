# FALCON++

Code for **FALCON++: Shorter Signatures without NTRU Smoothing Estimates**.  
**Hao Yan and Nicholas Zhao**, Imperial College London.  
Contact: `{h.yan22,n.zhao22}@imperial.ac.uk` · [中文说明](README.zh-CN.md)

This compact repository reproduces all six main-text tables and runs the complete KeyGen → Sign → Verify workflow for **I-1245** (n=512, q=509) and **V-117** (n=1024, q=1949).

## Run

Use Python **3.13**, the validated version. From the repository root:

```sh
python -m pip install -r requirements.txt
python reproduce_paper.py
python demo_falconpp.py --seed example-run
```

Only NumPy and mpmath are required. No LaTeX or plotting tools are needed. Dependencies are pinned to the validated versions; a Python virtual environment is recommended.

## Table reproduction

`python reproduce_paper.py` recomputes the numerical quantities and entropy bounds, exports CSV/Markdown tables, and checks **all 88 displayed data cells**. A mismatch makes the command fail.

| Table | Content | Output |
|---|---|---|
| 1 | Matched-width sampling iterations | `results/tables/table1.csv` |
| 2 | Key/signature sizes and cost estimates | `results/tables/table2.csv` |
| 3 | Selected parameters | `results/tables/table3.csv` |
| 4 | Modulus and size comparison | `results/tables/table4.csv` |
| 5 | Moment orders and acceptance bounds | `results/tables/table5.csv` |
| 6 | Heuristic costs and chi-BDD diagnostic | `results/tables/table6.csv` |

`results/table_verification.json` records each comparison and its source: **38 calculated cells, 26 design inputs and 24 cited literature inputs**. `data/paper_tables.tex` contains verbatim table excerpts for comparison, with the original manuscript's SHA-256. It is not used as computational answers. Literature comparison inputs and their references are in `data/literature_baselines.json`.

Unrounded numerical results are in `results/selected_parameters.json`; entropy enclosures are in `results/entropy.json`. Tables 2 and 4 use the paper's **419/921-byte entropy estimates**, recomputed from the stated formula and bounds. The workflow demo reports actual encoded lengths separately.

## Complete workflow

```sh
python demo_falconpp.py --parameter I-1245
python demo_falconpp.py --parameter V-117
python demo_falconpp.py --wire-format padded --seed example-run
```

The default runs both sets with unpadded canonical rANS. The optional padded mode exposes the existing conservative fixed-length format. `--seed` enables repeatable experiments; omitting it uses OS-seeded randomness.

KeyGen generates the NTRU trapdoor and expanded sampler tree. Signing uses a fresh salt, hashes to a syndrome, samples a Klein–GPV preimage, applies clipped correction and a norm check, then encodes the signature. Verification decodes, rehashes and checks the reconstructed vector. The demo checks both a valid signature and rejection of a modified message, writes aggregate results to `results/workflow.json`, and does not save private keys.

## Files

- `reproduce_paper.py`, `verify_entropy.py`, `export_tables.py`, `check_paper_tables.py`: numerical reproduction and comparison.
- `demo_falconpp.py`: runnable complete scheme example.
- `implement/falconpp/`: required key generation, arithmetic, sampling, signing and codec modules.
- `implement/security/`: required numerical models.
- `data/`: two small reference-input files.

The release excludes the paper, figures, historical campaign drivers, development tests and precomputed output. Runtime results are generated locally. Core numerical and signing algorithms retain their original source attribution; source-reference details for comparison data are included in the JSON input. This packaging adds no new license grant.

For GitHub web upload, extract the ZIP and drag **all contents inside `FALCONpp-github/`** into the repository's upload area. Keep `.gitignore` and both subdirectories. This compact package fits in one upload.
