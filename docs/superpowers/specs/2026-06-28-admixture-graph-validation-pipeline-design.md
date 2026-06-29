# Admixture Graph Validation Pipeline — Design Spec

**Date:** 2026-06-28
**Status:** Approved (brainstorming complete) — build step-by-step on user's cue
**Scope:** The end-to-end *validation* pipeline only. NOT the deep-learning model. This backbone proves the moving parts (simulate → genotypes → f-statistics → ADMIXTOOLS 2 inference → compare to truth) work together, and becomes the substrate for later training-data generation.

---

## 1. Goal & non-goals

**Goal.** A single, well-documented Jupyter notebook that, end to end:
1. Defines a known true admixture graph.
2. Simulates real SNP genotypes under it (msprime).
3. Computes f-statistics from those genotypes (ADMIXTOOLS 2).
4. Infers a graph from the f-statistics (ADMIXTOOLS 2).
5. Compares the recovered graph to the known truth and emits PASS/FAIL.

**Non-goals.** No ML model. No large-scale data generation yet. No analytic f2 generator yet (the f2-on-disk artifact is the seam where it plugs in later).

**Working style.** One notebook, built one step at a time. The user hands over each step; do not jump ahead. Every step = one markdown cell (what it does + expected output) + code + printed sanity checks.

---

## 2. Key decisions (locked)

| Decision | Choice | Rationale |
|---|---|---|
| Data source | **msprime → real genotypes** | Gold-standard end-to-end validation; exercises sampling noise and every moving part the real data will. |
| Inference mode | **qpgraph sanity → then find_graphs** | Cheap deterministic check (score true topology, residuals ≈ 0) before the hard stochastic blind search; isolates failures. |
| Genotype I/O layer | **snputils** (SNPObject) | Lab-standard SNP tooling expected by PI; eases later training-data scale-up. Sits between tskit genotypes and ADMIXTOOLS 2. |
| R boundary | **subprocess (`Rscript`)**, file exchange, no rpy2 | Clean, debuggable seam; matches "R via subprocess" instruction. |
| Genotype format to ADMIXTOOLS 2 | **PLINK** (.bed/.bim/.fam) primary; EIGENSTRAT fallback | Read natively by `extract_f2`. |
| Swappable seam | **f2 blocks on disk** | Later analytic generator writes the same artifact; everything downstream is unchanged. |

---

## 3. Data flow

```
[Python] true graph spec (topology + drift lengths + admixture weight α)
   │
   ├─► msprime Demography ─► sim_ancestry + sim_mutations ─► genotypes (tskit)
   │                                                            │
   │                                          snputils (SNPObject) ─► PLINK .bed/.bim/.fam
   │                                                            │
[R via Rscript] ◄────────────────────────────────────────────────┘
   │   admixtools::extract_f2  ─►  f2 blocks (cached on disk)   ← SWAPPABLE SEAM
   │   A) qpgraph(true topology, f2)   → worst |residual| Z (sanity: small)
   │   B) find_graphs(f2, nadmix=1)    → top-k scoring graphs
   │   write results back as JSON/CSV
   │
[Python] read results ─► compare recovered vs truth (lenient) ─► PASS/FAIL report
```

---

## 4. The true graph

**6 sampled populations total = 5 ingroup (P1–P5) + 1 outgroup (O).** The simulation samples exactly these 6 — no more, no fewer. 5 ingroup populations with the outgroup is comfortably above the n ≥ 5 identifiability threshold for a single admixture event.

```
                root
                /  \
               O    anc                 O  = outgroup (constrains topology)
                    /  \
                anc_L   anc_R
                /  \     /  \
              P1  anc_L2 P4  anc_R2
                   /  \      /  \
                  P2  srcA  P5  srcB
                        \   /
                         P3 = α·srcA + (1-α)·srcB     (the single admixture event)
```

- **Leaves (sampled):** O, P1, P2, P3, P4, P5.
- **Internal nodes:** root, anc, anc_L, anc_R, anc_L2, anc_R2, srcA, srcB.
- **Admixture:** P3 = α·srcA + (1−α)·srcB, with srcA on the P2 side and srcB on the P5 side (well-separated sources → recoverable event). α ≈ 0.4 (final value set in Step 1).

**Internal-node wiring to verify when building (do not just accept the diagram):**
- anc_L2 → {P2, srcA}
- anc_R2 → {P5, srcB}
- P3 is the admixed child of (srcA, srcB).

**Drift values must earn trust.** Assign drift lengths so the admixture is *detectable*. Specifically the srcA and srcB branches must be long enough off their parents that the admixture signal survives. Acceptance gate (checked in Step 5): **f3(P3; P2, P5) is clearly negative** (convincingly negative Z-score, not marginal). If it isn't, the sources are too short off their parents — **retune drift and re-verify; do not proceed to the search.**

---

## 5. Step roadmap (build one at a time, on user's cue)

| # | Step | Key output / sanity check |
|---|---|---|
| 0 | **Environment setup** | Install/verify msprime, snputils, R, ADMIXTOOLS 2; print all versions. (None currently installed.) |
| 1 | **Define the true graph** | Topology + drift + α as a Python data structure — single source of truth. Verify internal wiring (§4). |
| 2 | **Build msprime demography** | Construct Demography from the spec; sanity-check the object (populations, events, times). |
| 3 | **Simulate genotypes** | sim_ancestry + sim_mutations; N diploids/pop; report SNP count, per-pop sample counts. |
| 4 | **Export via snputils → PLINK** | SNPObject → .bed/.bim/.fam with population labels in .fam; confirm 6 pops, sample/SNP dims. |
| 5 | **Compute f2 (ADMIXTOOLS 2)** | `extract_f2` via Rscript; sanity-check f2 magnitudes/signs. **GATE: f3(P3; P2, P5) clearly negative** — else retune drift (Step 1) and repeat. |
| 6 | **qpgraph sanity** | Score true topology; confirm worst |residual Z| is small (roundtrip is sound). |
| 7 | **find_graphs (blind search)** | nadmix=1; return **top-k** graphs with scores (not just rank-1). |
| 8 | **Compare to truth → PASS/FAIL** | **Lenient** isomorphism: did it recover the admixed population + broad structure? Report whether truth is **in the top-k set**. PASS does NOT require rank-1 exact equality. |

---

## 6. Success criteria (what "the pipeline works" means)

- Step 5 produces f-statistics with the expected sign structure, **including a clearly negative f3(P3; P2, P5)**.
- Step 6: qpgraph on the true topology fits well (worst residual Z small) — confirms simulate→f2→inference roundtrips.
- Step 8: a graph matching the truth (lenient: correct admixed pop + broad structure) appears in the **top-k** find_graphs results. Rank-1 exact recovery is a bonus, not required.

---

## 7. Open items

- **snputils**: chosen as the I/O layer; confirm exact format support (PLINK write path) at Step 4; EIGENSTRAT fallback if needed.
- Final drift values and α: set in Step 1, validated by the Step 5 f3 gate.
- top-k value (k) for find_graphs: decide at Step 7 (default suggestion: k = 5–10).
