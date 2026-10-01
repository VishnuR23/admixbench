# Run log

Every run that produced files in `results/` is recorded here: what was run, from which commit, and what
it cost. Machine: Apple M1, 8 cores, 8.6 GB RAM, R 4.6 with admixtools 2.0.10, msprime 1.3.4.

## 2026-09-30: M1 acceptance test

Commit `2e35e8f`. Command: `python scripts/run_m1.py acceptance` (seed 1, 1e8 bp).
Output: `results/m1_acceptance.csv`.

| graph | qpgraph score | worst residual (Z) |
|---|---|---|
| N* (true) | 1.14 | 1.07 |
| N1 (popE admixed, TreeMix) | 514.68 | 20.54 |
| N0 (ML tree) | 1818.19 | 30.53 |

Pass: N* < N1 < N0 with gaps over 50, and the N* worst residual is under 3. 549,051 SNPs, 50 blocks.
Simulation peaked at 1.0 GB of memory, which caps the stage 1 pool at 4 workers on this machine.

## 2026-09-30: pilot benchmark

Commit `ea569db` (grid locked). Command:
`python scripts/run_benchmark.py --replicates 3 --numstart 10 --out results/pilot`.
All five configs, seeds 1 to 3, 1e8 bp, 10 find_graphs runs per replicate. Wall time 3672 s
(stage 1: 857 s on 4 workers; stage 2: 2815 s).
Re-scored from cache at commit `2e21a73` (new identifiability flag and outcome column) into
`results/pilot/rescored/`. Timings come from `results/pilot/benchmark.csv`, because the re-score hit the
cache.

Per-stage cost, mean per replicate:

| config | SNPs | simulate | export | extract_f2 | qpgraph | one find_graphs run | find_graphs, 10 runs |
|---|---|---|---|---|---|---|---|
| trivial | 540k | 187 s | 2 s | 23 s | 1 s | 24 s | 45 s |
| easy | 623k | 205 s | 3 s | 20 s | 1 s | 40 s | 77 s |
| m1 | 550k | 163 s | 2 s | 12 s | 1 s | 36 s | 69 s |
| moderate | 679k | 179 s | 4 s | 43 s | 2 s | 132 s | 269 s |
| hard | 718k | 187 s | 3 s | 27 s | 1 s | 246 s | 476 s |

Outcomes (10 runs is a smoke test, not evidence):

| config | gates passed | recovered | alternative fits | search failure | max worst residual |
|---|---|---|---|---|---|
| trivial | 3/3 | 3 | 0 | 0 | 2.73 |
| easy | 3/3 | 3 | 0 | 0 | 1.89 |
| m1 | 3/3 | 3 | 0 | 0 | 1.85 |
| moderate | 3/3 | 2 | 1 | 0 | 2.63 |
| hard | 3/3 | 1 | 1 | 1 | 2.49 |

## 2026-09-30: pilot M1 sweep

Commit `2e35e8f`. Command:
`python scripts/run_m1.py sweep --lengths 1e7 3e7 1e8 --seeds 1 2 3 --numstart 10 --out results/pilot/m1_sweep.csv`.
Two arms per dataset: `pinned` (outpop=popE) and `free` (no outgroup, so popE can be inferred as
admixed). Result: A in 18 of 18 runs, E in none. One find_graphs call took 59 to 81 s when nothing else was
running. The 1e8 pinned searches were reused from the benchmark pilot cache.

## Cost model

- Stage 1 (simulate, export, extract_f2, qpgraph) costs about 215 s per replicate and runs 4 at a time,
  so about 54 s of wall time per replicate.
- find_graphs splits N runs over 7 cores ahead of time, so its wall time is about ceil(N/7) times the
  mean run time. At N=10 this predicted 48 / 79 / 71 / 263 / 492 s against measured 45 / 77 / 69 / 269 /
  476 s. A 50-run easy search in the walkthrough took 351 s against a predicted 349 s. The estimates below
  add 10% for uneven runs.

| config | per replicate, numstart 50 | per replicate, numstart 100 |
|---|---|---|
| trivial | 264 s | 448 s |
| easy | 406 s | 712 s |
| m1 | 358 s | 632 s |
| moderate | 1215 s | 2229 s |
| hard | 2219 s | 4113 s |
| all five | 4462 s (1.24 h) | 8134 s (2.26 h) |

Benchmark totals: numstart 50 gives 6.2 h at 5 replicates, 12.4 h at 10, 18.6 h at 15. numstart 100 gives
11.3 h at 5 replicates and 22.6 h at 10. Hard is half of every total.

M1 sweep at numstart 50: one search is about 314 s. Three lengths times 10 seeds times 2 arms is 60
searches. The 10 pinned searches at 1e8 bp are shared with the benchmark's m1 config, so 50 are new:
about 4.4 h, plus about 15 min of simulation at 1e7 and 3e7 bp.

## Proposed full run (not launched)

- Benchmark: all five configs, 10 replicates (seeds 1 to 10), numstart 50, 1e8 bp. About 12.4 h.
- M1 sweep: lengths 1e7, 3e7 and 1e8 bp, seeds 1 to 10, numstart 50, both arms. About 4.6 h.
- Total about 17 h, which leaves about 30% headroom under 24 h.
