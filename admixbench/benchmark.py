"""Benchmark runner: simulate, extract f2, gate on the true graph, search, score.

Simulation, export, extract_f2 and the qpgraph gate are single-threaded, so
replicates run through them in parallel with multiprocessing. find_graphs
already uses every core, so searches then run one replicate at a time.
Everything is cached on disk, so a stopped run resumes where it left off.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import yaml

from . import admixtools, metrics
from .export import write_eigenstrat
from .graphspec import easy_graph, m1_graph, random_graph, trivial_graph
from .simulate import simulate

SIM_DEFAULTS = dict(samples_per_pop=10, sequence_length=1e8,
                    recombination_rate=1e-8, mutation_rate=1.25e-8)
FIXED = {"trivial": trivial_graph, "easy": easy_graph, "m1": m1_graph}


# ------------------------------------------------------------------ configs
def load_grid(path="configs/grid.yaml"):
    return yaml.safe_load(Path(path).read_text())


def make_spec(cfg, seed):
    """Fixed graphs ignore the seed. Random graphs draw their topology from it."""
    if cfg["graph"] == "random":
        return random_graph(cfg["n_leaves"], cfg["n_admix"], seed)
    return FIXED[cfg["graph"]]()


# -------------------------------------------------------------------- data
def data_key(spec, seed, sim):
    blob = json.dumps({"spec": repr(spec), "seed": seed, "sim": {k: float(v) for k, v in sorted(sim.items())}})
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def prepare_data(spec, seed, sim, cache_root, keep_genotypes=False):
    """Simulate, export and extract f2, or reuse the cached result.

    Returns a dict with ``datadir``, ``f2dir``, ``prefix``, ``n_snps``,
    ``n_blocks``, stage timings, and ``cached``.
    """
    sim = {**SIM_DEFAULTS, **sim}
    datadir = Path(cache_root) / f"{spec.name}_{data_key(spec, seed, sim)}"
    meta_path = datadir / "meta.json"
    if meta_path.exists():
        return {**json.loads(meta_path.read_text()), "cached": True}

    datadir.mkdir(parents=True, exist_ok=True)
    prefix = datadir / "sim"
    t0 = time.time()
    ts = simulate(spec, seed=seed, **sim)
    t1 = time.time()
    info = write_eigenstrat(ts, prefix)
    t2 = time.time()
    n_blocks = admixtools.extract_f2(prefix, datadir / "f2")
    t3 = time.time()
    if not keep_genotypes:
        Path(f"{prefix}.geno").unlink()
    meta = dict(datadir=str(datadir), f2dir=str(datadir / "f2"), prefix=str(prefix),
                n_snps=info["n_snps"], n_blocks=n_blocks,
                t_simulate=t1 - t0, t_export=t2 - t1, t_extract_f2=t3 - t2)
    meta_path.write_text(json.dumps(meta, indent=2))
    return {**meta, "cached": False}


# --------------------------------------------------------------- one replicate
def _stage1(job):
    """Data and the qpgraph gate for one replicate (runs in a worker)."""
    name, cfg, seed, sim, cache_root = job
    spec = make_spec(cfg, seed)
    data = prepare_data(spec, seed, sim, cache_root)
    t0 = time.time()
    gate = admixtools.qpgraph_true(data["f2dir"], spec, data["datadir"])
    data["t_qpgraph"] = time.time() - t0
    return name, seed, spec, data, gate


def _stage2(name, seed, spec, data, gate, numstart, stop_gen):
    """find_graphs and scoring for one replicate."""
    truth = spec.to_networkx()
    t0 = time.time()
    rows = admixtools.find_graphs(data["f2dir"], numadmix=len(spec.admixtures), outpop=spec.outgroup,
                                  numstart=numstart, seed=seed, workdir=data["datadir"], stop_gen=stop_gen)
    t_find = time.time() - t0
    ranked = [admixtools.edges_to_networkx(r["edges"]) for r in rows]
    best = ranked[0]
    acc = metrics.admixed_accuracy(truth, best)
    sources = list(acc["sources_correct"].values())
    return dict(
        config=name, seed=seed, graph=spec.name, n_snps=data["n_snps"], n_blocks=data["n_blocks"],
        qpgraph_true_score=gate["score"], qpgraph_worst_residual=gate["worst_residual"],
        gate_passed=gate["passed"], find_graphs_best_score=rows[0]["score"],
        score_gap=rows[0]["score"] - gate["score"], n_topologies=len(rows),
        topology_equal=metrics.topology_equality(truth, best),
        set_distance=metrics.set_distance(truth, best),
        covariance_distance=metrics.covariance_distance(gate["fitted"], best),
        admixed_exact=acc["exact_match"], admixed_precision=acc["precision"], admixed_recall=acc["recall"],
        sources_correct=all(sources) if sources else None, clades_exact=acc["clades_exact"],
        true_admixed=" ".join(sorted(acc["true_admixed"])),
        inferred_admixed=" ".join(sorted(acc["inferred_admixed"])),
        rank_of_truth=metrics.rank_of_truth(truth, ranked),
        identifiability_flag=metrics.identifiability_flag(gate["fitted"], best),
        numstart=numstart, t_simulate=data["t_simulate"], t_export=data["t_export"],
        t_extract_f2=data["t_extract_f2"], t_qpgraph=data["t_qpgraph"], t_find_graphs=t_find,
    )


def run_replicate(name, cfg, seed, cache_root, numstart, stop_gen=100, sim=None):
    """One replicate end to end, in this process. Returns the result row."""
    _, _, spec, data, gate = _stage1((name, cfg, seed, sim or {}, cache_root))
    return _stage2(name, seed, spec, data, gate, numstart, stop_gen)


# --------------------------------------------------------------------- grid
def run_grid(grid, cache_root="cache", out_dir="results", configs=None, replicates=None,
             numstart=None, sim=None, workers=None, log=print):
    """Run every (config, replicate) in ``grid`` and write benchmark.csv and
    summary.csv to ``out_dir``. Arguments override the grid file."""
    sim = {**grid.get("sim", {}), **(sim or {})}
    search = grid.get("search", {})
    stop_gen = search.get("stop_gen", 100)
    names = configs or list(grid["configs"])
    jobs = []
    for name in names:
        cfg = grid["configs"][name]
        n = replicates or cfg.get("replicates", grid["replicates"])
        jobs += [(name, cfg, seed, sim, cache_root) for seed in range(1, n + 1)]

    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    log(f"stage 1: {len(jobs)} replicates, data + qpgraph gate, {workers} workers")
    t0 = time.time()
    with Pool(workers) as pool:
        stage1 = pool.map(_stage1, jobs, chunksize=1)
    log(f"stage 1 done in {time.time() - t0:.0f}s")

    rows = []
    for name, seed, spec, data, gate in stage1:
        cfg = grid["configs"][name]
        ns = numstart or cfg.get("numstart", search["numstart"])
        t = time.time()
        rows.append(_stage2(name, seed, spec, data, gate, ns, stop_gen))
        r = rows[-1]
        log(f"{name} seed={seed}: gate {'PASS' if r['gate_passed'] else 'FAIL'} "
            f"(worst |Z| {r['qpgraph_worst_residual']:.2f}), best {r['find_graphs_best_score']:.2f} "
            f"vs true {r['qpgraph_true_score']:.2f}, set_dist {r['set_distance']}, "
            f"admixed {r['inferred_admixed'] or '-'} (true {r['true_admixed'] or '-'}), "
            f"rank {r['rank_of_truth']}, {time.time() - t:.0f}s")

    df = pd.DataFrame(rows)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "benchmark.csv", index=False)
    summarize(df).to_csv(out / "summary.csv")
    return df


def summarize(df):
    """Mean and sd per config of every numeric or boolean column."""
    cols = [c for c in df.columns if c not in ("seed",) and
            (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]))]
    num = df[["config"] + cols].copy()
    num[cols] = num[cols].astype(float)
    s = num.groupby("config", sort=False).agg(["mean", "std"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s.insert(0, "n", df.groupby("config", sort=False).size())
    return s
