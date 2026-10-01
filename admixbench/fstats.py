"""f-statistics in Python: the analytic forward model and empirical estimates.

The empirical estimates are sanity checks for the walkthrough. ADMIXTOOLS 2
computes its own f2 for inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


# --------------------------------------------------------------- forward model
def _leaves(g):
    return sorted(n for n in g if g.out_degree(n) == 0)


def _ancestry_weights(g, leaf):
    """Fraction of ``leaf``'s ancestry that flows through each edge.

    One unit of ancestry starts at the leaf and is pushed up to the root. At an
    admixture node it splits by the incoming proportions. On a tree every edge
    on the root-to-leaf path gets weight 1. With admixture the weight on an
    edge is the summed product of proportions over all paths through it.
    """
    node_w = dict.fromkeys(g, 0.0)
    node_w[leaf] = 1.0
    edge_w = {}
    for n in reversed(list(nx.topological_sort(g))):
        w = node_w[n]
        if not w:
            continue
        preds = list(g.predecessors(n))
        for p in preds:
            share = w * g.edges[p, n]["proportion"] if len(preds) == 2 else w
            edge_w[p, n] = share
            node_w[p] += share
    return edge_w


def expected_f2(g):
    """Expected f2 between every pair of leaves, as a symmetric DataFrame.

    Each edge adds independent drift of variance ``length``. A leaf's allele
    frequency is the weighted sum of the drift on edges above it, so

        f2(A, B) = sum over edges e of length(e) * (w_A(e) - w_B(e))**2.

    On a tree this is the sum of lengths on the path between A and B. With
    admixture it sums over all paths, each weighted by the product of the
    proportions along it. Admixture edges count if they carry a length.
    """
    leaves = _leaves(g)
    edges = list(g.edges)
    length = np.array([g.edges[e].get("length", np.nan) for e in edges])
    if np.isnan(length).any():
        missing = [e for e, l in zip(edges, length) if np.isnan(l)]
        raise ValueError(f"expected_f2 needs a length on every edge; missing on {missing[:5]}")
    W = np.array([[_ancestry_weights(g, leaf).get(e, 0.0) for e in edges] for leaf in leaves])
    diff = W[:, None, :] - W[None, :, :]
    return pd.DataFrame((diff ** 2) @ length, index=leaves, columns=leaves)


# ------------------------------------------------------------ empirical stats
@dataclass
class FreqData:
    """Per-population allele frequencies plus what the estimators need."""
    pops: list
    freqs: np.ndarray        # pops x snps, reference allele frequency
    n_hap: np.ndarray        # haploid sample size per population
    blocks: np.ndarray       # jackknife block id per SNP

    def __getitem__(self, pop):
        i = self.pops.index(pop)
        return self.freqs[i], self.n_hap[i]


def read_eigenstrat(prefix, block_size=2e6):
    """FreqData from EIGENSTRAT files, with 2 Mb jackknife blocks by position."""
    prefix = Path(prefix)
    ind = [l.split() for l in open(f"{prefix}.ind").read().splitlines()]
    labels = np.array([r[2] for r in ind])
    raw = np.frombuffer(Path(f"{prefix}.geno").read_bytes(), dtype=np.uint8)
    geno = raw.reshape(-1, len(ind) + 1)[:, :-1] - ord("0")          # snps x individuals
    pos = np.loadtxt(f"{prefix}.snp", usecols=3, dtype=np.int64, ndmin=1)
    pops = list(dict.fromkeys(labels))
    freqs, n_hap = [], []
    for p in pops:
        cols = labels == p
        freqs.append(geno[:, cols].sum(axis=1) / (2 * cols.sum()))
        n_hap.append(2 * cols.sum())
    return FreqData(pops, np.array(freqs), np.array(n_hap), (pos // block_size).astype(np.int64))


@dataclass
class Stat:
    est: float
    se: float
    z: float


def jackknife(values, blocks):
    """Weighted block jackknife (Busing et al. 1999) for the mean of per-SNP values."""
    _, block_of = np.unique(blocks, return_inverse=True)
    m = np.bincount(block_of).astype(float)
    s = np.bincount(block_of, weights=values)
    n, total = m.sum(), values.sum()
    est = total / n
    loo = (total - s) / (n - m)                    # leave-one-block-out means
    h = n / m
    g = len(m)
    theta_j = g * est - np.sum((1 - m / n) * loo)
    pseudo = h * est - (h - 1) * loo
    var = np.sum((pseudo - theta_j) ** 2 / (h - 1)) / g
    se = float(np.sqrt(var))
    return Stat(float(est), se, float(est / se) if se > 0 else np.nan)


def _het(p, n):
    return p * (1 - p) / (n - 1)


def f2(data, a, b):
    """Unbiased f2(A, B) with block jackknife SE."""
    (pa, na), (pb, nb) = data[a], data[b]
    return jackknife((pa - pb) ** 2 - _het(pa, na) - _het(pb, nb), data.blocks)


def f3(data, c, a, b):
    """Unbiased f3(C; A, B). Negative with Z < -3 is evidence C is admixed."""
    (pc, nc), (pa, _), (pb, _) = data[c], data[a], data[b]
    return jackknife((pc - pa) * (pc - pb) - _het(pc, nc), data.blocks)


def f4(data, a, b, c, d):
    """f4(A, B; C, D) with block jackknife SE."""
    (pa, _), (pb, _), (pc, _), (pd_, _) = data[a], data[b], data[c], data[d]
    return jackknife((pa - pb) * (pc - pd_), data.blocks)
