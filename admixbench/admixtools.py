"""Bridge to ADMIXTOOLS 2 in R.

Each call writes a small R script and runs it with Rscript. stdout and stderr
are captured. A nonzero exit raises with the tail of stderr. No rpy2.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import networkx as nx

RSCRIPT = shutil.which("Rscript") or "/opt/homebrew/bin/Rscript"
DIAG = 1e-4
EDGE_COLUMNS = {"from", "to", "type", "weight", "low", "high"}
EDGE_TYPES = {"edge", "admix"}


class RError(RuntimeError):
    pass


def run_r(code, workdir, label):
    """Write ``code`` to ``workdir/_label.R``, run it, return stdout."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / f"_{label}.R"
    path.write_text(code)
    p = subprocess.run([RSCRIPT, str(path)], capture_output=True, text=True)
    if p.returncode != 0:
        tail = "\n".join(p.stderr.strip().splitlines()[-15:])
        raise RError(f"R step '{label}' failed (exit {p.returncode}):\n{tail}")
    return p.stdout


def r_available():
    """True if Rscript runs and admixtools and jsonlite load."""
    try:
        p = subprocess.run([RSCRIPT, "-e", "suppressMessages({library(admixtools); library(jsonlite)})"],
                           capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return p.returncode == 0


def _q(path):
    return json.dumps(str(path))       # a quoted string that R parses


# ------------------------------------------------------------------ extract_f2
def extract_f2(prefix, outdir):
    """Run admixtools::extract_f2 and return the number of jackknife blocks.

    adjust_pseudohaploid=FALSE is required. Left on, admixtools checks the
    first 1000 SNPs for heterozygotes, can decide simulated diploids are
    pseudohaploid, and silently halves their allele-frequency information. We
    saw one population's f2 to the outgroup come out at half its true value.
    """
    outdir = Path(outdir)
    out = run_r(f"""
suppressMessages(library(admixtools))
extract_f2({_q(prefix)}, outdir={_q(outdir)}, overwrite=TRUE, maxmiss=0,
           adjust_pseudohaploid=FALSE, minac2=FALSE, verbose=FALSE)
f2 <- f2_from_precomp({_q(outdir)}, verbose=FALSE)
cat(sprintf("n_blocks: %d\\n", dim(f2)[3]))
""", outdir.parent, "extract_f2")
    n_blocks = int(re.search(r"n_blocks: (\d+)", out).group(1))
    if n_blocks <= 1:
        raise RError(f"extract_f2 produced {n_blocks} jackknife block(s); positions are probably wrong")
    return n_blocks


# --------------------------------------------------------------------- qpgraph
def _r_edges(graph):
    """(from, to) rows for qpgraph. Internal nodes get plain names; leaves
    keep theirs."""
    internal = sorted(n for n in graph if graph.out_degree(n) > 0)
    name = {n: f"n{i}" for i, n in enumerate(internal)}
    name.update({n: n for n in graph if graph.out_degree(n) == 0})
    if len(set(name.values())) != len(name):
        raise ValueError("a leaf name collides with a generated internal node name")
    return [[name[u], name[v]] for u, v in graph.edges]


def qpgraph_score(f2dir, graph, outpop, workdir, label="qpgraph"):
    """Fit a fixed topology. Returns {"score", "worst_residual", "edges"}.

    The worst residual is the largest |Z| over the f4 residuals. ``edges`` are
    the fitted edges, in the find_graphs format. ``f2_se_norm`` is the
    Frobenius norm of the block-jackknife standard errors of the f2 matrix,
    the scale of the sampling noise. The f3 base population is set
    to ``outpop``, as find_graphs does. Left unset, qpgraph uses the first
    leaf in the edge list, and with few jackknife blocks the same topology
    then scores differently depending on edge order.
    """
    workdir = Path(workdir)
    edges_path = workdir / f"_{label}_edges.json"
    workdir.mkdir(parents=True, exist_ok=True)
    edges_path.write_text(json.dumps(_r_edges(graph)))
    out = run_r(f"""
suppressMessages({{library(admixtools); library(jsonlite)}})
f2 <- f2_from_precomp({_q(f2dir)}, verbose=FALSE)
e <- fromJSON({_q(edges_path)})
q <- qpgraph(f2, e, diag={DIAG}, f3basepop={_q(outpop)}, return_fstats=TRUE)
n <- dim(f2)[3]
loo <- sapply(seq_len(n), function(j) apply(f2[, , -j, drop=FALSE], 1:2, mean), simplify="array")
se <- sqrt((n - 1) / n * apply((loo - c(apply(loo, 1:2, mean)))^2, 1:2, sum))
cat(toJSON(list(score=q$score, worst_residual=max(abs(q$f4$z), na.rm=TRUE), f2_se_norm=norm(se, "F"),
                edges=as.data.frame(q$edges)), auto_unbox=TRUE, digits=NA, na="null", dataframe="columns"))
""", workdir, label)
    res = json.loads(out.strip().splitlines()[-1])
    edges = parse_find_graphs([{"score": res["score"], "hash": label, "edges": res["edges"]}])[0]["edges"]
    return {"score": float(res["score"]), "worst_residual": float(res["worst_residual"]),
            "f2_se_norm": float(res["f2_se_norm"]), "edges": edges}


def qpgraph_true(f2dir, spec, workdir):
    """Score the true topology on its own data. This is the sanity gate: a
    worst residual of 3 SE or more means the problem is upstream of the search,
    and no search result on this data should be trusted. Also returns
    ``fitted``, the true topology with lengths fitted to the data."""
    res = qpgraph_score(f2dir, spec.to_networkx(), spec.outgroup, workdir, "qpgraph_true")
    res["passed"] = res["worst_residual"] < 3
    res["fitted"] = edges_to_networkx(res["edges"])
    return res


# ----------------------------------------------------------------- find_graphs
def find_graphs(f2dir, numadmix, outpop, numstart, seed, workdir, stop_gen=100, cache=True):
    """Blind topology search with ``numstart`` independent runs of
    admixtools::find_graphs, in parallel over cores.

    find_graphs has no restart argument of its own (``numstart`` passed to it
    would reach qpgraph and set its optimizer starts instead). Each run here
    is one full hill-climb from a random start. All rows from all runs are
    pooled, deduplicated by topology hash keeping the best score, and sorted
    by score. Returns {"rows": [{"score", "hash", "run", "edges"}, ...],
    "run_secs": [wall seconds of each run]}. ``outpop=None`` leaves the
    outgroup free, so any population, the true outgroup included, can be
    inferred as admixed. The JSON is cached under ``workdir`` keyed on the
    arguments.
    """
    workdir = Path(workdir)
    args = dict(f2dir=str(f2dir), numadmix=numadmix, outpop=outpop, numstart=numstart,
                seed=seed, stop_gen=stop_gen, diag=DIAG)
    key = hashlib.sha256(json.dumps(args, sort_keys=True).encode()).hexdigest()[:12]
    out_path = workdir / f"find_graphs_{key}.json"
    if not (cache and out_path.exists()):
        run_r(f"""
suppressMessages({{library(admixtools); library(jsonlite); library(parallel)}})
f2 <- f2_from_precomp({_q(f2dir)}, verbose=FALSE)
one <- function(i) {{
  t0 <- proc.time()[["elapsed"]]
  r <- find_graphs(f2, numadmix={numadmix}, outpop={_q(outpop) if outpop else "NULL"},
                   stop_gen={stop_gen}, diag={DIAG}, verbose=FALSE)
  secs <- proc.time()[["elapsed"]] - t0
  lapply(seq_len(nrow(r)), function(j)
    list(score=r$score[j], hash=r$hash[j], run=i, secs=secs, edges=as.data.frame(r$edges[[j]])))
}}
RNGkind("L'Ecuyer-CMRG"); set.seed({seed})
res <- mclapply(seq_len({numstart}), one, mc.cores=max(1, detectCores() - 1), mc.set.seed=TRUE)
bad <- vapply(res, inherits, logical(1), "try-error")
if (any(bad)) stop(paste("find_graphs run failed:", res[bad][[1]]))
write_json(do.call(c, res), {_q(out_path)}, auto_unbox=TRUE, digits=NA, na="null", dataframe="columns")
""", workdir, f"find_graphs_{key}")
    raw = json.loads(out_path.read_text())
    run_secs = {r["run"]: r["secs"] for r in raw if "secs" in r}
    return {"rows": parse_find_graphs(raw), "run_secs": [run_secs[k] for k in sorted(run_secs)]}


def parse_find_graphs(raw):
    """Validate find_graphs JSON and return rows deduplicated by hash, sorted
    by score. Raises on any schema it does not recognise."""
    if not isinstance(raw, list) or not raw:
        raise ValueError("find_graphs output is empty or not a list")
    best = {}
    for row in raw:
        missing = {"score", "hash", "edges"} - set(row)
        if missing:
            raise ValueError(f"find_graphs row is missing {sorted(missing)}")
        cols = set(row["edges"])
        if cols != EDGE_COLUMNS:
            raise ValueError(f"unexpected find_graphs edge columns {sorted(cols)}; expected {sorted(EDGE_COLUMNS)}")
        e = row["edges"]
        n = len(e["from"])
        if any(len(e[c]) != n for c in ("to", "type", "weight")):
            raise ValueError("find_graphs edge columns have different lengths")
        types = set(e["type"])
        if not types <= EDGE_TYPES:
            raise ValueError(f"unexpected find_graphs edge types {sorted(types - EDGE_TYPES)}")
        if any(not isinstance(w, (int, float)) for w in e["weight"]):
            raise ValueError("find_graphs edge weights must all be numbers")
        edges = [dict(zip(("from", "to", "type", "weight"), r))
                 for r in zip(e["from"], e["to"], e["type"], e["weight"])]
        item = {"score": float(row["score"]), "hash": row["hash"], "run": row.get("run"), "edges": edges}
        if row["hash"] not in best or item["score"] < best[row["hash"]]["score"]:
            best[row["hash"]] = item
    return sorted(best.values(), key=lambda r: r["score"])


def edges_to_networkx(edges):
    """DiGraph from find_graphs edges. ``admix`` edges become admixture edges
    with ``proportion``; anything else is a drift edge with ``length``."""
    g = nx.DiGraph()
    for e in edges:
        if e["type"] == "admix":
            g.add_edge(e["from"], e["to"], admixture=True, proportion=e["weight"], length=0.0)
        else:
            g.add_edge(e["from"], e["to"], admixture=False, length=e["weight"])
    for n in g:
        g.nodes[n]["leaf"] = g.out_degree(n) == 0
    return g
