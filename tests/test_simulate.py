import msprime
import numpy as np
import pytest

from admixbench.graphspec import easy_graph, random_graph, trivial_graph
from admixbench.simulate import build_demography, simulate


def test_demography_declares_all_populations_and_orders_events():
    spec = easy_graph()
    d = build_demography(spec)
    assert [p.name for p in d.populations] == [p.name for p in spec.populations]
    times = [e.time for e in d.events]
    assert times == sorted(times) and times[0] == 100
    assert type(d.events[0]).__name__ == "Admixture"


@pytest.mark.parametrize("n_leaves,n_admix", [(5, 1), (8, 2), (8, 4)])
@pytest.mark.parametrize("seed", range(5))
def test_random_graph_demographies_run(n_leaves, n_admix, seed):
    spec = random_graph(n_leaves, n_admix, seed)
    ts = simulate(spec, samples_per_pop=2, sequence_length=1e4, seed=seed + 1)
    assert ts.num_samples == 2 * 2 * (n_leaves + 1)


def test_samples_land_in_leaf_populations():
    spec = easy_graph()
    ts = simulate(spec, samples_per_pop=3, sequence_length=1e5, seed=1)
    names = [ts.population(ts.node(u).population).metadata["name"] for u in ts.samples()]
    assert names == [leaf for leaf in spec.leaves for _ in range(6)]   # 3 diploids = 6 nodes


def test_diversity_matches_4_Ne_mu():
    # constant Ne everywhere, so pairwise coalescence time is 2Ne on any topology
    ts = simulate(trivial_graph(), samples_per_pop=5, sequence_length=2e7, seed=3)
    p1 = ts.samples(population=ts.populations()[0].id)
    pi = ts.diversity(sample_sets=[p1], mode="site")
    assert pi == pytest.approx(4 * 10_000 * 1.25e-8, rel=0.15)
    assert ts.num_sites > 0 and all(len(s.mutations) >= 1 for s in ts.sites())
