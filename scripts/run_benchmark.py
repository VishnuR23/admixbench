"""Run the benchmark grid and write results/benchmark.csv and results/summary.csv.

    python scripts/run_benchmark.py                       # the grid as configured
    python scripts/run_benchmark.py --replicates 3 --numstart 10 --out results/pilot
"""
import argparse
import time

from admixbench.benchmark import load_grid, run_grid


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", default="configs/grid.yaml")
    p.add_argument("--configs", nargs="+", help="subset of configs to run")
    p.add_argument("--replicates", type=int)
    p.add_argument("--numstart", type=int)
    p.add_argument("--sequence-length", type=float)
    p.add_argument("--workers", type=int)
    p.add_argument("--cache", default="cache")
    p.add_argument("--out", default="results")
    a = p.parse_args()
    sim = {"sequence_length": a.sequence_length} if a.sequence_length else None
    t0 = time.time()
    df = run_grid(load_grid(a.grid), cache_root=a.cache, out_dir=a.out, configs=a.configs,
                  replicates=a.replicates, numstart=a.numstart, sim=sim, workers=a.workers,
                  log=lambda m: print(f"[{time.time() - t0:7.0f}s] {m}", flush=True))
    print(f"done: {len(df)} replicates in {time.time() - t0:.0f}s -> {a.out}/benchmark.csv")


if __name__ == "__main__":
    main()
