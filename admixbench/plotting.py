"""Draw admixture graphs with one convention for truth and inferred graphs.

Blue leaves, grey internal nodes, black solid drift edges, red dashed
admixture edges labelled with percentages.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx

from .metrics import canonicalize_admixture


def _is_admix(G, u, v):
    return G.in_degree(v) == 2


def layered_positions(G):
    """Rows by depth from the root (longest path), leaves on the bottom row.
    Leaves are spread left to right in depth-first order; internal nodes are
    pulled toward the mean of their neighbours."""
    root = next(n for n in G if G.in_degree(n) == 0)
    depth = {root: 0}
    for n in nx.topological_sort(G):
        for c in G.successors(n):
            depth[c] = max(depth.get(c, 0), depth[n] + 1)
    bottom = max(depth.values())

    # leaf order: depth-first over drift edges only; then each admixed subtree
    # is slotted in between its two sources
    tree = G.copy()
    tree.remove_edges_from([(u, v) for u, v in G.edges if _is_admix(G, u, v)])

    def leaves_below(n):
        return [d for d in nx.dfs_preorder_nodes(tree, n, sort_neighbors=lambda ns: sorted(ns, key=str))
                if tree.out_degree(d) == 0 and G.out_degree(d) == 0]

    x = {leaf: float(i) for i, leaf in enumerate(leaves_below(root))}
    for a in (n for n in nx.topological_sort(G) if G.in_degree(n) == 2):
        anchors = [x[d] for p in G.predecessors(a) for d in nx.descendants(tree, p) | {p} if d in x]
        block = leaves_below(a)
        centre = sum(anchors) / len(anchors) if anchors else len(x)
        for k, leaf in enumerate(block):
            x[leaf] = centre + 0.01 * (k + 1)
    x = {leaf: float(i) for i, leaf in enumerate(sorted(x, key=x.get))}
    internal = [n for n in G if G.out_degree(n) > 0]
    for n in internal:
        below = [d for d in nx.descendants(G, n) if G.out_degree(d) == 0]
        x[n] = sum(x[d] for d in below) / len(below)
    for _ in range(100):
        for n in internal:
            nb = [x[c] for c in G.successors(n)] + [x[p] for p in G.predecessors(n)]
            x[n] = 0.5 * x[n] + 0.5 * sum(nb) / len(nb)
    return {n: (x[n], -(bottom if G.out_degree(n) == 0 else depth[n])) for n in G}


def plot_graph(G, ax, positions=None, title=None):
    pos = positions or layered_positions(G)
    leaves = [n for n in G if G.out_degree(n) == 0]
    internal = [n for n in G if G.out_degree(n) > 0]
    drift = [(u, v) for u, v in G.edges if not _is_admix(G, u, v)]
    admix = [(u, v) for u, v in G.edges if _is_admix(G, u, v)]
    nx.draw_networkx_edges(G, pos, edgelist=drift, ax=ax, edge_color="black", width=1.2, arrows=True, arrowsize=8)
    nx.draw_networkx_edges(G, pos, edgelist=admix, ax=ax, edge_color="crimson", style="dashed",
                           width=1.8, arrows=True, arrowsize=10)
    labels = {e: f"{100 * G.edges[e]['proportion']:.0f}%" for e in admix if "proportion" in G.edges[e]}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax, font_color="crimson", font_size=8,
                                 bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    nx.draw_networkx_nodes(G, pos, nodelist=internal, ax=ax, node_color="lightgrey", node_size=60)
    nx.draw_networkx_nodes(G, pos, nodelist=leaves, ax=ax, node_color="#8ecae6", node_size=650)
    nx.draw_networkx_labels(G, pos, labels={n: n for n in leaves}, ax=ax, font_size=9, font_weight="bold")
    if title:
        ax.set_title(title)
    ax.margins(0.12)
    ax.axis("off")
    return ax


def plot_comparison(true, inferred, titles=("truth", "inferred")):
    """Truth and inferred side by side, both canonicalized so equivalent
    graphs look the same."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, g, t in zip(axes, (true, inferred), titles):
        plot_graph(canonicalize_admixture(g), ax, title=t)
    fig.tight_layout()
    return fig
