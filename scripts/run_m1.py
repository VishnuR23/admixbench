"""M1 (Molloy et al. 2021): acceptance test, then the sequence-length sweep.

    python scripts/run_m1.py acceptance
    python scripts/run_m1.py sweep --lengths 1e7 3e7 1e8 --seeds 1 2 3 --numstart 50

acceptance: simulate M1, score the true graph N*, the TreeMix graph N1 and the
ML tree N0 with qpgraph. Required: N* < N1 < N0 with large gaps and the N*
worst residual under 3 SE. Exits nonzero on failure.

sweep: find_graphs at numadmix=1 for each length and seed, in two arms.
"pinned" sets outpop=popE, as one normally would. "free" sets no outgroup,
so popE itself can be inferred as admixed, which is the trap.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from admixbench import admixtools, metrics
from admixbench.benchmark import prepare_data
from admixbench.graphspec import m1_alternatives, m1_graph

GAP = 50     # minimum score gap between N* < N1 < N0 to count as "large"


def acceptance(seed, length, cache, out):
    spec = m1_graph()
    t0 = time.time()
    data = prepare_data(spec, seed, {"sequence_length": length}, cache)
    print(f"M1 seed={seed} length={length:.0e}: {data['n_snps']} SNPs, {data['n_blocks']} blocks "
          f"({'cached' if data['cached'] else f'{time.time() - t0:.0f}s'})")
    graphs = {"N*": spec.to_networkx(), **m1_alternatives()}
    rows = []
    for name, g in graphs.items():
        r = admixtools.qpgraph_score(data["f2dir"], g, spec.outgroup, data["datadir"], f"m1_{name.strip('*')}")
        rows.append({"graph": name, "score": r["score"], "worst_residual": r["worst_residual"]})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    s = dict(zip(df.graph, df.score))
    checks = {
        "N* worst residual < 3": df.loc[df.graph == "N*", "worst_residual"].item() < 3,
        f"N1 - N* > {GAP}": s["N1"] - s["N*"] > GAP,
        f"N0 - N1 > {GAP}": s["N0"] - s["N1"] > GAP,
    }
    for k, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {k}")
    df["seed"], df["sequence_length"], df["n_snps"] = seed, length, data["n_snps"]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return all(checks.values())


def classify(clades):
    if clades == [["popA"]]:
        return "A"
    if clades == [["popE"]]:
        return "E"
    return "other"


def sweep(lengths, seeds, numstart, cache, out):
    spec = m1_graph()
    truth = spec.to_networkx()
    rows = []
    for length in lengths:
        for seed in seeds:
            data = prepare_data(spec, seed, {"sequence_length": length}, cache)
            gate = admixtools.qpgraph_true(data["f2dir"], spec, data["datadir"])
            for arm, outpop in (("pinned", spec.outgroup), ("free", None)):
                t0 = time.time()
                fg = admixtools.find_graphs(data["f2dir"], 1, outpop, numstart, seed, data["datadir"])
                best = admixtools.edges_to_networkx(fg["rows"][0]["edges"])
                H = metrics.canonicalize_admixture(best)
                clades = sorted(sorted(metrics._desc_leaves(H, n)) for n in H if H.in_degree(n) == 2)
                row = dict(sequence_length=length, seed=seed, arm=arm, numstart=numstart,
                           n_snps=data["n_snps"], gate_worst_residual=gate["worst_residual"],
                           true_score=gate["score"], best_score=fg["rows"][0]["score"],
                           admixed=";".join(",".join(c) for c in clades), outcome=classify(clades),
                           set_distance=metrics.set_distance(truth, best),
                           rank_of_truth=metrics.rank_of_truth(truth, [admixtools.edges_to_networkx(r["edges"])
                                                                       for r in fg["rows"]]),
                           t_find_graphs=time.time() - t0,
                           run_secs_mean=sum(fg["run_secs"]) / len(fg["run_secs"]))
                rows.append(row)
                print(f"L={length:.0e} seed={seed} {arm:6s}: admixed {row['admixed']:12s} -> {row['outcome']:5s} "
                      f"best {row['best_score']:.2f} vs true {row['true_score']:.2f}, rank {row['rank_of_truth']}, "
                      f"{row['t_find_graphs']:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)
    df = pd.DataFrame(rows)
    table = (df.groupby(["arm", "sequence_length"])["outcome"].value_counts().unstack(fill_value=0)
             .reindex(columns=["A", "E", "other"], fill_value=0))
    table["runs"] = table.sum(axis=1)
    print(table.to_string())
    table.to_csv(Path(out).with_name("m1_table.csv"))
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("acceptance")
    a.add_argument("--seed", type=int, default=1)
    a.add_argument("--length", type=float, default=1e8)
    a.add_argument("--out", default="results/m1_acceptance.csv")
    s = sub.add_parser("sweep")
    s.add_argument("--lengths", type=float, nargs="+", required=True)
    s.add_argument("--seeds", type=int, nargs="+", required=True)
    s.add_argument("--numstart", type=int, required=True)
    s.add_argument("--out", default="results/m1_sweep.csv")
    for x in (a, s):
        x.add_argument("--cache", default="cache")
    args = p.parse_args()
    if args.cmd == "acceptance":
        sys.exit(0 if acceptance(args.seed, args.length, args.cache, args.out) else 1)
    sweep(args.lengths, args.seeds, args.numstart, args.cache, args.out)


if __name__ == "__main__":
    main()
