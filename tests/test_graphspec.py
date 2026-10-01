import networkx as nx
import pytest

from admixbench.graphspec import Admixture, GraphSpec, Population, Split, easy_graph, random_graph, trivial_graph


def three_pop_tree():
    # ((A,B),O): A and B split at 1000, their ancestor meets O at 3000.
    return GraphSpec(
        populations=[Population("A"), Population("B"), Population("O"),
                     Population("AB", leaf=False), Population("R", leaf=False)],
        splits=[Split(1000, ("A", "B"), "AB"), Split(3000, ("AB", "O"), "R")],
        admixtures=[],
        outgroup="O",
    )


def one_admix():
    # X is founded at t=100 by srcA (0.35) and srcB (0.65). srcA is a ghost
    # sister of A, srcB a ghost sister of B.
    return GraphSpec(
        populations=[Population(n) for n in ("A", "B", "X", "O")]
        + [Population(n, leaf=False) for n in ("srcA", "srcB", "LA", "LB", "AB", "R")],
        splits=[Split(1000, ("A", "srcA"), "LA"), Split(1200, ("B", "srcB"), "LB"),
                Split(2000, ("LA", "LB"), "AB"), Split(3000, ("AB", "O"), "R")],
        admixtures=[Admixture(100, "X", ("srcA", "srcB"), (0.35, 0.65))],
        outgroup="O",
    )


# ------------------------------------------------------------------ to_networkx
def test_tree_graph_structure_and_lengths():
    g = three_pop_tree().to_networkx()
    assert set(g.edges) == {("R", "AB"), ("R", "O"), ("AB", "A"), ("AB", "B")}
    assert {n for n in g if g.nodes[n]["leaf"]} == {"A", "B", "O"}
    Ne2 = 2 * 10_000
    assert g.edges["AB", "A"]["length"] == pytest.approx(1000 / Ne2)
    assert g.edges["R", "AB"]["length"] == pytest.approx(2000 / Ne2)
    assert g.edges["R", "O"]["length"] == pytest.approx(3000 / Ne2)
    assert not any(d["admixture"] for *_, d in g.edges(data=True))


def test_admixture_graph_structure():
    g = one_admix().to_networkx()
    # the ghost sources are pass-through lineages, so they are contracted into
    # the admixture edges, which start at the sister split nodes
    adm = [n for n in g if g.in_degree(n) == 2]
    assert adm == ["X_adm"]
    ins = {u: g.edges[u, "X_adm"] for u in g.predecessors("X_adm")}
    assert set(ins) == {"LA", "LB"}
    assert all(d["admixture"] for d in ins.values())
    assert ins["LA"]["proportion"] == 0.35 and ins["LB"]["proportion"] == 0.65
    assert ins["LA"]["length"] == pytest.approx((1000 - 100) / 20_000)
    assert g.edges["X_adm", "X"]["length"] == pytest.approx(100 / 20_000)
    assert not g.edges["X_adm", "X"]["admixture"]
    assert nx.is_directed_acyclic_graph(g)
    assert [n for n in g if g.in_degree(n) == 0] == ["R"]


def test_source_on_leaf_lineage_gets_its_own_node():
    # X draws directly from leaf lineages A and B: nodes appear on their branches
    spec = GraphSpec(
        populations=[Population(n) for n in ("A", "B", "C", "X", "O")]
        + [Population(n, leaf=False) for n in ("AC", "ABC", "R")],
        splits=[Split(1000, ("A", "C"), "AC"), Split(2000, ("AC", "B"), "ABC"),
                Split(3000, ("ABC", "O"), "R")],
        admixtures=[Admixture(100, "X", ("A", "B"), (0.3, 0.7))],
        outgroup="O",
    )
    g = spec.to_networkx()
    assert set(g.predecessors("X_adm")) == {"A@100", "B@100"}
    assert g.edges["A@100", "A"]["length"] == pytest.approx(100 / 20_000)
    assert g.edges["AC", "A@100"]["length"] == pytest.approx(900 / 20_000)
    assert g.edges["ABC", "B@100"]["length"] == pytest.approx(1900 / 20_000)


def test_leaves_listed_in_declared_order():
    assert one_admix().leaves == ("A", "B", "X", "O")


# ------------------------------------------------------------------- validation
def test_rejects_unsorted_events():
    with pytest.raises(ValueError, match="increasing"):
        GraphSpec(
            populations=three_pop_tree().populations,
            splits=[Split(3000, ("AB", "O"), "R"), Split(1000, ("A", "B"), "AB")],
            admixtures=[], outgroup="O")


def test_rejects_tied_times_across_event_lists():
    base = one_admix()
    with pytest.raises(ValueError, match="increasing"):
        GraphSpec(base.populations, base.splits,
                  [Admixture(1000, "X", ("srcA", "srcB"), (0.35, 0.65))], "O")


def test_rejects_undeclared_population():
    with pytest.raises(ValueError, match="undeclared"):
        GraphSpec(
            populations=[Population("A"), Population("B"), Population("O"), Population("R", leaf=False)],
            splits=[Split(1000, ("A", "B"), "AB"), Split(3000, ("AB", "O"), "R")],
            admixtures=[], outgroup="O")


def test_rejects_proportions_not_summing_to_one():
    base = one_admix()
    with pytest.raises(ValueError, match="sum to 1"):
        GraphSpec(base.populations, base.splits,
                  [Admixture(100, "X", ("srcA", "srcB"), (0.35, 0.6))], "O")


def test_rejects_sibling_sources():
    # A and B split directly from AB with nothing in between: siblings
    with pytest.raises(ValueError, match="siblings"):
        GraphSpec(
            populations=[Population(n) for n in ("A", "B", "X", "O")]
            + [Population("AB", leaf=False), Population("R", leaf=False)],
            splits=[Split(1000, ("A", "B"), "AB"), Split(3000, ("AB", "O"), "R")],
            admixtures=[Admixture(100, "X", ("A", "B"), (0.3, 0.7))],
            outgroup="O")


def test_rejects_source_not_alive():
    # A is already merged into AB at 1000, so it cannot be a source at 1500
    with pytest.raises(ValueError, match="not alive"):
        GraphSpec(
            populations=[Population(n) for n in ("A", "B", "C", "X", "O")]
            + [Population(n, leaf=False) for n in ("AB", "ABC", "R")],
            splits=[Split(1000, ("A", "B"), "AB"), Split(2000, ("AB", "C"), "ABC"),
                    Split(3000, ("ABC", "O"), "R")],
            admixtures=[Admixture(1500, "X", ("A", "C"), (0.3, 0.7))],
            outgroup="O")


def test_rejects_two_roots():
    with pytest.raises(ValueError, match="root"):
        GraphSpec(
            populations=[Population("A"), Population("B"), Population("O"), Population("AB", leaf=False)],
            splits=[Split(1000, ("A", "B"), "AB")], admixtures=[], outgroup="O")


# ------------------------------------------------------------------ fixed graphs
def test_fixed_graphs_are_valid_and_shaped_right():
    t = trivial_graph()
    assert len(t.leaves) == 6 and not t.admixtures and t.outgroup == "O"
    e = easy_graph()
    assert len(e.leaves) == 6 and len(e.admixtures) == 1
    a = e.admixtures[0]
    assert a.derived == "P3" and a.time == 100 and sorted(a.proportions) == [0.35, 0.65]


# ------------------------------------------------------------------ random_graph
@pytest.mark.parametrize("n_leaves,n_admix", [(5, 0), (5, 1), (8, 2), (8, 4)])
@pytest.mark.parametrize("seed", range(20))
def test_random_graph_is_valid(n_leaves, n_admix, seed):
    spec = random_graph(n_leaves, n_admix, seed)   # validation runs on construction
    g = spec.to_networkx()
    leaves = [n for n in g if g.nodes[n]["leaf"]]
    assert len(leaves) == n_leaves + 1
    assert sum(1 for n in g if g.in_degree(n) == 2) == n_admix
    # binary: every split node has two children
    assert all(g.out_degree(n) in (0, 1, 2) for n in g)
    # outgroup splits deepest: it hangs directly off the root
    root = next(n for n in g if g.in_degree(n) == 0)
    assert spec.outgroup in g.successors(root)
    # proportions from [0.1, 0.4] on the minor side
    for a in spec.admixtures:
        assert 0.1 <= min(a.proportions) <= 0.4
        assert spec.outgroup not in (a.derived, *a.sources)


def test_random_graph_is_deterministic_in_seed():
    assert random_graph(8, 2, 7) == random_graph(8, 2, 7)
    assert random_graph(8, 2, 7) != random_graph(8, 2, 8)


def test_random_graph_rejects_too_many_admixtures():
    with pytest.raises(ValueError):
        random_graph(5, 3, 0)
