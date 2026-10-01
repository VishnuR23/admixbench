import networkx as nx
import numpy as np
import pytest

from admixbench.graphspec import easy_graph, random_graph, trivial_graph
from admixbench.metrics import (admixed_accuracy, canonicalize_admixture, covariance_distance,
                                identifiability_flag, rank_of_truth, set_distance, topology_equality)


def _graph(edges, admix=()):
    g = nx.DiGraph()
    for u, v, l in edges:
        g.add_edge(u, v, length=l, admixture=False)
    for u, v, w in admix:
        g.add_edge(u, v, length=0.0, proportion=w, admixture=True)
    return g


def leaf_form(g):
    """Rewrite an X_adm -> X admixture so both admixture edges go into X, the
    way the other representation does it."""
    h = g.copy()
    for n in [n for n in h if h.in_degree(n) == 2]:
        (x,) = h.successors(n)
        for p in list(h.predecessors(n)):
            h.add_edge(p, x, **h.edges[p, n])
        h.remove_node(n)
    return h


def relabel_internal(g, seed):
    rng = np.random.default_rng(seed)
    internal = [n for n in g if g.out_degree(n) > 0]
    new = [f"n{i}" for i in rng.permutation(len(internal))]
    return nx.relabel_nodes(g, dict(zip(internal, new)))


# ------------------------------------------------------------- identical graphs
@pytest.mark.parametrize("g", [trivial_graph().to_networkx(), easy_graph().to_networkx(),
                               random_graph(8, 4, 3).to_networkx()], ids=["trivial", "easy", "hard"])
def test_identical_graphs_score_zero(g):
    other = relabel_internal(g, 1)          # internal names must not matter
    assert topology_equality(g, other)
    assert set_distance(g, other) == 0
    assert covariance_distance(g, other) == pytest.approx(0, abs=1e-12)
    acc = admixed_accuracy(g, other)
    assert acc["exact_match"] and acc["precision"] == acc["recall"] == 1.0
    assert all(acc["sources_correct"].values()) and acc["clades_exact"]
    assert rank_of_truth(g, [other]) == 1
    assert not identifiability_flag(g, other, tol=1e-9)


def test_one_leaf_swap_gives_positive_set_distance():
    g = trivial_graph().to_networkx()                  # ((P1,P2),(P3,(P4,P5))),O
    swapped = nx.relabel_nodes(g, {"P1": "P4", "P4": "P1"})
    assert not topology_equality(g, swapped)
    assert set_distance(g, swapped) > 0
    assert covariance_distance(g, swapped) > 0
    assert rank_of_truth(g, [swapped]) is None


# --------------------------------------------------------- canonicalization
def test_spec_form_and_leaf_form_compare_equal():
    spec_form = easy_graph().to_networkx()            # P3_adm -> P3
    other = leaf_form(spec_form)                      # sources -> P3 directly
    assert "P3_adm" in spec_form and "P3_adm" not in other
    assert set(other.predecessors("P3")) == {"A_L2", "A_R2"}
    assert topology_equality(spec_form, other)
    assert topology_equality(other, spec_form)
    assert set_distance(spec_form, other) == 0
    assert rank_of_truth(spec_form, [trivial_graph().to_networkx(), other]) == 2
    assert admixed_accuracy(spec_form, other)["exact_match"]


def test_canonical_form_without_it_would_mismatch():
    # the comparison would fail on raw graphs: different node counts
    spec_form = easy_graph().to_networkx()
    other = leaf_form(spec_form)
    assert len(spec_form) != len(other)
    assert len(canonicalize_admixture(spec_form)) == len(canonicalize_admixture(other))


def test_wrong_source_is_not_equal():
    truth = easy_graph().to_networkx()
    bad = leaf_form(truth)
    bad.remove_edge("A_R2", "P3")
    bad.add_edge("A_L", "P3", proportion=0.65, admixture=True, length=0.0)   # P1-side source
    assert not topology_equality(truth, bad)
    acc = admixed_accuracy(truth, bad)
    assert acc["exact_match"]                         # right leaf is admixed...
    assert acc["sources_correct"] == {"P3": False}    # ...but from the wrong side


def test_admixed_accuracy_precision_recall():
    truth = easy_graph().to_networkx()
    tree = trivial_graph().to_networkx()
    acc = admixed_accuracy(truth, tree)
    assert acc["true_admixed"] == {"P3"} and acc["inferred_admixed"] == set()
    assert not acc["exact_match"] and acc["recall"] == 0.0 and acc["precision"] == 1.0
    acc = admixed_accuracy(tree, truth)
    assert acc["precision"] == 0.0 and acc["recall"] == 1.0


# ---------------------------------------------------------- identifiability
def test_root_placement_twins_are_flagged():
    # one unrooted tree, rooted in two places: same f2, different clades
    a = _graph([("R", "u", 0.07), ("u", "A", 0.20), ("u", "B", 0.20),
                ("R", "w", 0.08), ("w", "C", 0.25), ("w", "O", 0.40)])
    b = _graph([("R", "O", 0.20), ("R", "w", 0.20), ("w", "C", 0.25),
                ("w", "u", 0.15), ("u", "A", 0.20), ("u", "B", 0.20)])
    assert not topology_equality(a, b)
    assert set_distance(a, b) >= 2
    assert covariance_distance(a, b) == pytest.approx(0, abs=1e-12)
    assert identifiability_flag(a, b, tol=1e-9)
    # the same pair, but the truth fits much better: a search failure, not identifiability
    assert not identifiability_flag(a, b, tol=1e-9, score_gap=9.0)


def test_different_admixed_population_is_not_flagged():
    # structure differs and so does f2: a real error, not an identifiability case
    def admixed(x, others):
        s, t = others
        return _graph([("R", "O", 0.30), ("R", "M", 0.10), ("M", "n1", 0.13), ("M", "n2", 0.09),
                       ("n1", s, 0.20), ("n2", t, 0.22), ("adm", x, 0.18)],
                      admix=[("n1", "adm", 0.5), ("n2", "adm", 0.5)])
    a, c = admixed("A", ("C", "B")), admixed("C", ("A", "B"))
    assert set_distance(a, c) >= 2
    assert covariance_distance(a, c) > 1e-3
    assert not identifiability_flag(a, c, tol=1e-3)


def test_covariance_distance_rejects_mismatched_leaves():
    with pytest.raises(ValueError, match="leaf sets"):
        covariance_distance(trivial_graph().to_networkx(), _graph([("R", "A", 1), ("R", "B", 1)]))
