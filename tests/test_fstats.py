import networkx as nx
import numpy as np
import pytest

from admixbench import fstats
from admixbench.export import write_eigenstrat
from admixbench.graphspec import GraphSpec, Population, Split, easy_graph, trivial_graph
from admixbench.simulate import simulate


def _graph(edges, admix=()):
    g = nx.DiGraph()
    for u, v, l in edges:
        g.add_edge(u, v, length=l)
    for u, v, w in admix:
        g.add_edge(u, v, length=0.0, proportion=w, admixture=True)
    return g


# ------------------------------------------------------------- forward model
def test_expected_f2_three_population_tree_by_hand():
    #        R
    #     2 / \ 3
    #      I   C        f2(A,B) = 4+5 = 9
    #   4 / \ 5         f2(A,C) = 4+2+3 = 9
    #    A   B          f2(B,C) = 5+2+3 = 10
    g = _graph([("R", "I", 2), ("R", "C", 3), ("I", "A", 4), ("I", "B", 5)])
    f = fstats.expected_f2(g)
    assert f.loc["A", "B"] == pytest.approx(9)
    assert f.loc["A", "C"] == pytest.approx(9)
    assert f.loc["B", "C"] == pytest.approx(10)
    assert np.allclose(f.values, f.values.T) and np.allclose(np.diag(f.values), 0)


def test_expected_f2_from_spec_lengths():
    # A,B split at 1000, meet O at 3000, Ne=10k: lengths 0.05, 0.05, 0.10, 0.15
    spec = GraphSpec(
        populations=[Population("A"), Population("B"), Population("O"),
                     Population("AB", leaf=False), Population("R", leaf=False)],
        splits=[Split(1000, ("A", "B"), "AB"), Split(3000, ("AB", "O"), "R")],
        admixtures=[], outgroup="O")
    f = fstats.expected_f2(spec.to_networkx())
    assert f.loc["A", "B"] == pytest.approx(0.10)
    assert f.loc["A", "O"] == pytest.approx(0.30)


def test_expected_f2_admixture_by_hand():
    # x = r+dx, y = r+dy, A = x+dA, B = y+dB, C = (x+y)/2 + dC, all variances 1
    # A - C = dx/2 - dy/2 + dA - dC  ->  f2(A,C) = 1/4 + 1/4 + 1 + 1 = 2.5
    g = _graph([("R", "x", 1), ("R", "y", 1), ("x", "A", 1), ("y", "B", 1), ("Cadm", "C", 1)],
               admix=[("x", "Cadm", 0.5), ("y", "Cadm", 0.5)])
    f = fstats.expected_f2(g)
    assert f.loc["A", "B"] == pytest.approx(4)
    assert f.loc["A", "C"] == pytest.approx(2.5)
    assert f.loc["B", "C"] == pytest.approx(2.5)


def test_expected_f2_counts_length_on_admixture_edges():
    g = _graph([("R", "x", 1), ("R", "y", 1), ("x", "A", 1), ("y", "B", 1), ("Cadm", "C", 1)],
               admix=[("x", "Cadm", 0.5), ("y", "Cadm", 0.5)])
    g.edges["x", "Cadm"]["length"] = 1.0       # adds 0.5**2 * 1 to f2(A,C)
    assert fstats.expected_f2(g).loc["A", "C"] == pytest.approx(2.75)


def test_expected_f3_of_easy_target_is_negative():
    f = fstats.expected_f2(easy_graph().to_networkx())
    f3 = (f.loc["P3", "P2"] + f.loc["P3", "P5"] - f.loc["P2", "P5"]) / 2
    assert f3 < 0


# --------------------------------------------------------------- jackknife
def test_jackknife_equal_blocks_matches_textbook_formula():
    rng = np.random.default_rng(0)
    vals = rng.normal(1.0, 2.0, 1000)
    blocks = np.repeat(np.arange(20), 50)
    s = fstats.jackknife(vals, blocks)
    loo = np.array([vals[blocks != b].mean() for b in range(20)])
    textbook = np.sqrt(19 / 20 * np.sum((loo - loo.mean()) ** 2))
    assert s.est == pytest.approx(vals.mean())
    assert s.se == pytest.approx(textbook)
    assert s.z == pytest.approx(s.est / s.se)


# ------------------------------------------------------- empirical vs model
@pytest.fixture(scope="module")
def sims(tmp_path_factory):
    out = {}
    for name, spec in (("trivial", trivial_graph()), ("easy", easy_graph())):
        ts = simulate(spec, samples_per_pop=10, sequence_length=2e7, seed=5)
        prefix = tmp_path_factory.mktemp(name) / "sim"
        write_eigenstrat(ts, prefix)
        out[name] = (spec, fstats.read_eigenstrat(prefix))
    return out


def test_read_eigenstrat(sims):
    spec, data = sims["easy"]
    assert data.pops == list(spec.leaves)
    assert set(data.n_hap) == {20}
    assert len(np.unique(data.blocks)) == 10        # 2e7 bp / 2 Mb


def test_empirical_f2_is_proportional_to_expected(sims):
    # with constant Ne, E[f2 per SNP] is a fixed multiple of the drift-unit f2
    spec, data = sims["trivial"]
    exp = fstats.expected_f2(spec.to_networkx())
    ratios = [fstats.f2(data, a, b).est / exp.loc[a, b]
              for i, a in enumerate(data.pops) for b in data.pops[i + 1:]]
    assert np.allclose(ratios, np.median(ratios), rtol=0.15), ratios


def test_empirical_f3_of_easy_target_is_negative(sims):
    _, data = sims["easy"]
    s = fstats.f3(data, "P3", "P2", "P5")
    assert s.est < 0 and s.se > 0


def test_f4_treeness_on_trivial_tree(sims):
    # ((P1,P2),(P4,P5)) is a true split: f4(P1,P2;P4,P5) is 0 within noise,
    # while f4(P1,P4;P2,P5) is clearly nonzero
    _, data = sims["trivial"]
    assert abs(fstats.f4(data, "P1", "P2", "P4", "P5").z) < 3
    assert abs(fstats.f4(data, "P1", "P4", "P2", "P5").z) > 3
