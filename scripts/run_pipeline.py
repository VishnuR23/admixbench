"""Run one graph end to end and print the comparison with the truth.

    python scripts/run_pipeline.py                     # easy graph, 1e8 bp, 50 restarts
    python scripts/run_pipeline.py --config m1 --numstart 20
"""
import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from admixbench import admixtools, fstats, metrics
from admixbench.benchmark import load_grid, make_spec, prepare_data
from admixbench.plotting import plot_comparison


def sister_leaves(truth, x):
    """For admixed leaf x, one leaf under each source, for f3(x; a, b)."""
    H = metrics.canonicalize_admixture(truth)
    return [sorted(metrics._desc_leaves(H, u) - {x})[0] for u in H.predecessors(x)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="easy")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--numstart", type=int, default=50)
    p.add_argument("--sequence-length", type=float, default=1e8)
    p.add_argument("--cache", default="cache/pipeline")
    p.add_argument("--figure", default="results/figures/pipeline_comparison.png")
    a = p.parse_args()

    spec = make_spec(load_grid()["configs"][a.config], a.seed)
    truth = spec.to_networkx()
    t0 = time.time()
    data = prepare_data(spec, a.seed, {"sequence_length": a.sequence_length}, a.cache, keep_genotypes=True)
    print(f"graph {spec.name}: {len(spec.leaves)} leaves, {len(spec.admixtures)} admixture event(s)")
    print(f"SNPs: {data['n_snps']}  jackknife blocks: {data['n_blocks']}  ({time.time() - t0:.0f}s)")

    freqs = fstats.read_eigenstrat(data["prefix"])
    adm = metrics.admixed_accuracy(truth, truth)["true_admixed"]
    for x in sorted(adm):
        a1, b1 = sister_leaves(truth, x)
        s = fstats.f3(freqs, x, a1, b1)
        print(f"f3({x}; {a1}, {b1}) = {s.est:.5f}  SE {s.se:.5f}  Z {s.z:.2f}")

    gate = admixtools.qpgraph_true(data["f2dir"], spec, data["datadir"])
    print(f"qpgraph on the true topology: score {gate['score']:.3f}, worst residual |Z| "
          f"{gate['worst_residual']:.2f} -> {'PASS' if gate['passed'] else 'FAIL'} (bar: < 3)")

    t0 = time.time()
    fg = admixtools.find_graphs(data["f2dir"], len(spec.admixtures), spec.outgroup, a.numstart, a.seed,
                                data["datadir"])
    rows = fg["rows"]
    best = admixtools.edges_to_networkx(rows[0]["edges"])
    print(f"find_graphs: {a.numstart} runs, {len(rows)} distinct topologies, best score {rows[0]['score']:.3f} "
          f"({time.time() - t0:.0f}s)")

    acc = metrics.admixed_accuracy(truth, best)
    H = metrics.canonicalize_admixture(best)
    for x in sorted(acc["inferred_admixed"]):
        srcs = [sorted(metrics._desc_leaves(H, u) - {x}) for u in H.predecessors(x)]
        print(f"inferred admixed {x}: sources sit above {srcs[0]} and {srcs[1]}")
    ranked = [admixtools.edges_to_networkx(r["edges"]) for r in rows]
    print(f"true admixed {sorted(acc['true_admixed'])}, inferred {sorted(acc['inferred_admixed'])}, "
          f"sources correct {acc['sources_correct']}")
    print(f"topology equal: {metrics.topology_equality(truth, best)}  set distance: "
          f"{metrics.set_distance(truth, best)}  covariance distance: "
          f"{metrics.covariance_distance(gate['fitted'], best):.2e}  rank of truth: "
          f"{metrics.rank_of_truth(truth, ranked)}")

    Path(a.figure).parent.mkdir(parents=True, exist_ok=True)
    plot_comparison(truth, best, ("truth", f"find_graphs best (score {rows[0]['score']:.2f})")).savefig(
        a.figure, dpi=120, bbox_inches="tight")
    print(f"figure: {a.figure}")


if __name__ == "__main__":
    main()
