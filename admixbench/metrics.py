"""Metrics comparing a true admixture graph with an inferred one.

Graphs are NetworkX DiGraphs. Leaves (out-degree 0) are sampled populations
and are matched by node name. Internal nodes are unlabelled and
interchangeable. An admixture node has in-degree 2, and its incoming edges
carry ``proportion``. Every edge may carry ``length`` (drift).
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from .fstats import expected_f2


# ----------------------------------------------------------- canonical form
def canonicalize_admixture(G):
    """Put admixture into one form so equivalent graphs compare equal.

    Two forms show up. One routes an admixture through an internal node with
    two admixture edges in and a drift edge down to the leaf. The other puts
    the two admixture edges straight into the leaf. They describe the same
    graph. This contracts every internal admixture node whose only child is a
    leaf into that leaf. It also removes pass-through nodes (one parent, one
    child). Edge type is reset from structure: an edge is an admixture edge if
    its target has in-degree 2.

    The result is for structural comparison. Drift lengths on contracted
    edges are dropped, so do not feed it to expected_f2.
    """
    H = G.copy()
    changed = True
    while changed:
        changed = False
        for n in list(H):
            if H.out_degree(n) != 1:
                continue
            (c,) = H.successors(n)
            if H.in_degree(n) == 2 and H.out_degree(c) == 0:
                for p in list(H.predecessors(n)):
                    H.add_edge(p, c, **H.edges[p, n])
                H.remove_node(n)
                changed = True
            elif H.in_degree(n) == 1:
                (p,) = H.predecessors(n)
                attrs = dict(H.edges[n, c])
                attrs["length"] = attrs.get("length", 0.0) + H.edges[p, n].get("length", 0.0)
                H.remove_node(n)
                H.add_edge(p, c, **attrs)
                changed = True
    for n in H:
        H.nodes[n]["leaf"] = H.out_degree(n) == 0
    for u, v in H.edges:
        H.edges[u, v]["admixture"] = H.in_degree(v) == 2
    return H


def _leaves(g):
    return {n for n in g if g.out_degree(n) == 0}


def _desc_leaves(g, n):
    return frozenset(d for d in nx.descendants(g, n) | {n} if g.out_degree(d) == 0)


# ------------------------------------------------------------------ metrics
def topology_equality(true, inferred):
    """True when the graphs are isomorphic after canonicalization, with leaves
    matched by name, internal nodes interchangeable, and edge types (drift or
    admixture) matching. Lengths and proportions are ignored. This is a strict
    binary test and is not the primary metric."""
    a, b = canonicalize_admixture(true), canonicalize_admixture(inferred)
    for H in (a, b):
        for n in H:
            H.nodes[n]["label"] = n if H.nodes[n]["leaf"] else None
    return bool(nx.is_isomorphic(
        a, b,
        node_match=lambda x, y: x["label"] == y["label"],
        edge_match=lambda x, y: x["admixture"] == y["admixture"]))


def set_distance(true, inferred):
    """Primary structural metric.

    For every internal node, take the set of leaves reachable from it by any
    directed path. Each graph gives a collection of such sets, deduplicated.
    The distance is the size of the symmetric difference between the two
    collections.

    On a tree each set is a clade and this is the Robinson-Foulds distance on
    rooted clades. That does not carry over directly to admixture graphs,
    because a leaf below an admixture node is reachable from both sources, so
    sets overlap without nesting. The choice made here: reachability only,
    with no weighting by admixture proportion. A source node's set therefore
    includes the admixed leaf. Set distance then measures branching structure
    alone, and covariance_distance measures the f2 that structure predicts.
    Keeping them separate is what lets identifiability_flag spot graphs that
    differ in structure but not in f2. Both graphs are canonicalized first, so
    an admixture node directly above a leaf adds no extra set.
    """
    sets = []
    for g in (true, inferred):
        H = canonicalize_admixture(g)
        sets.append({_desc_leaves(H, n) for n in H if H.out_degree(n) > 0})
    return len(sets[0] ^ sets[1])


def covariance_distance(true, inferred):
    """Frobenius norm between the expected f2 matrices of the two graphs.

    The inferred graph must carry fitted lengths on every edge (find_graphs
    returns them). Both graphs must have the same leaves.
    """
    if _leaves(true) != _leaves(inferred):
        raise ValueError(f"leaf sets differ: {sorted(_leaves(true))} vs {sorted(_leaves(inferred))}")
    order = sorted(_leaves(true))
    A = expected_f2(true).loc[order, order].values
    B = expected_f2(inferred).loc[order, order].values
    return float(np.linalg.norm(A - B))


def admixed_accuracy(true, inferred):
    """Which leaves are admixed (in-degree 2 after canonicalization).

    Returns a dict with the true and inferred admixed sets, ``exact_match``,
    per-leaf ``precision`` and ``recall`` (1.0 when the denominator is empty),
    and ``sources_correct``: for each leaf admixed in both graphs, whether its
    two sources sit above the same sister leaves. A source's sisters are the
    leaves below the source node, other than the admixed leaf itself.

    ``clades_exact`` also compares the leaf sets below every admixture node.
    It covers admixture into an internal lineage, which never makes a leaf
    in-degree 2 and so is invisible to the per-leaf sets.
    """
    out = {}
    H = {k: canonicalize_admixture(g) for k, g in (("true", true), ("inferred", inferred))}
    adm = {k: {n for n in h if h.nodes[n]["leaf"] and h.in_degree(n) == 2} for k, h in H.items()}
    t, i = adm["true"], adm["inferred"]
    tp = len(t & i)
    out.update(true_admixed=t, inferred_admixed=i, exact_match=t == i,
               precision=tp / len(i) if i else 1.0, recall=tp / len(t) if t else 1.0)

    def sisters(h, x):
        return {_desc_leaves(h, u) - {x} for u in h.predecessors(x)}

    out["sources_correct"] = {x: sisters(H["true"], x) == sisters(H["inferred"], x) for x in t & i}
    clades = {k: sorted(sorted(_desc_leaves(h, n)) for n in h if h.in_degree(n) == 2) for k, h in H.items()}
    out["clades_exact"] = clades["true"] == clades["inferred"]
    return out


def rank_of_truth(true, ranked):
    """1-based position of the first graph in ``ranked`` that is topology-equal
    to ``true``, or None if none is."""
    for r, g in enumerate(ranked, start=1):
        if topology_equality(true, g):
            return r
    return None


def identifiability_flag(true, inferred, min_set_distance=2, rel_tol=0.02):
    """True when the graphs differ in structure (set distance at least
    ``min_set_distance``) but predict nearly the same f2 (covariance distance
    at most ``rel_tol`` times the norm of the true f2 matrix). That pattern
    means the data cannot tell the two graphs apart. It is not a failure of
    the search method."""
    if set_distance(true, inferred) < min_set_distance:
        return False
    order = sorted(_leaves(true))
    scale = np.linalg.norm(expected_f2(true).loc[order, order].values)
    return covariance_distance(true, inferred) <= rel_tol * scale
