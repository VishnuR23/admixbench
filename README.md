# Admixture-Graph Validation Pipeline

End-to-end validation backbone for a research project whose eventual goal is a deep-learning model
that predicts **admixture graphs** from **f-statistics**. This repo is **not** the ML model — it
proves the moving parts work together first:

> known admixture graph → simulate SNP genotypes (msprime) → compute f-statistics (ADMIXTOOLS 2)
> → infer a graph blindly (ADMIXTOOLS 2) → check the inference recovers the truth.

Everything lives in one documented notebook: [`admixture_graph_validation.ipynb`](admixture_graph_validation.ipynb).

## The validated result

Running the notebook top-to-bottom (from a clean state) produces:

| Step | Check | Result |
|---|---|---|
| 5 | Admixture signal `f3(P3; P2, P5)` | **Z = −4.17** (clearly negative) → gate passes |
| 6 | qpgraph fit of the **true** topology | worst residual **\|Z\| = 0.49** → round-trip sound |
| 7 | `find_graphs` blind search (1 admixture) | best graph **score 5.084** |
| 8 | Recovered vs truth | **PASS** — best-scoring graph is the truth (5.084 = true-topology score) |

The simulated graph: **6 sampled populations = 5 ingroup (P1–P5) + 1 outgroup (O)** with a single
admixture event at **P3** (mixing a P2-side and a P5-side lineage). Five ingroup populations plus the
outgroup is above the n ≥ 5 threshold needed to identify one admixture event.

## Pipeline steps

0. Environment checks · 1. Define the true graph · 2. Build msprime demography · 3. Simulate
genotypes · 4. Export to PLINK (snputils) · 5. f2/f3 + admixture-signal gate · 6. qpgraph sanity on
the true topology · 7. `find_graphs` blind search (top-k) · 8. Lenient graph comparison → PASS/FAIL.

Generated files land in `artifacts/` (git-ignored; the f2 blocks there are the **swappable seam**
where a fast analytic generator will later plug in for large-scale ML training data).

## Setup (macOS, reproducible)

Nothing was preinstalled; this is the exact path used.

### 1. Python environment (isolated venv)
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name admixgraph --display-name "Python (admixgraph .venv)"
```
`numpy` is pinned `<2` to avoid an ABI clash with prebuilt scipy/snputils wheels.

### 2. R + ADMIXTOOLS 2
```bash
brew install r gcc          # gcc provides gfortran, needed by some R deps
```
**Critical gotcha.** Homebrew's R 4.6 compiles C with `-std=gnu23`, which Apple clang 15 (Command
Line Tools) rejects — every pure-C R dependency fails to build. Override the C standard before
installing:
```bash
mkdir -p ~/.R
cat > ~/.R/Makevars <<'EOF'
CC = clang -std=gnu17
CC23 = clang -std=gnu17
EOF
```
Then install ADMIXTOOLS 2 (hard deps only — `dependencies=NA` avoids unrelated, lib-heavy Suggests):
```bash
Rscript -e 'install.packages(c("remotes","pgenlibr"), repos="https://cloud.r-project.org")'
Rscript -e 'remotes::install_github("uqrmaie1/admixtools", dependencies=NA, upgrade="never")'
```

## Run

Open the notebook with the **`admixgraph`** kernel and Run All, or headless:
```bash
PATH="/opt/homebrew/bin:$PATH" .venv/bin/python -m jupyter nbconvert --to notebook --execute \
  --inplace --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=admixgraph \
  admixture_graph_validation.ipynb
```
Make sure `Rscript` (`/opt/homebrew/bin`) is on `PATH` so the R subprocess steps can find ADMIXTOOLS 2.

## Design

See [`docs/superpowers/specs/2026-06-28-admixture-graph-validation-pipeline-design.md`](docs/superpowers/specs/2026-06-28-admixture-graph-validation-pipeline-design.md)
for the approved design and the decisions behind it.
