import numpy as np
import pytest

from admixbench.export import write_eigenstrat
from admixbench.graphspec import easy_graph
from admixbench.simulate import simulate

L = 1e7


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    spec = easy_graph()
    ts = simulate(spec, samples_per_pop=4, sequence_length=L, seed=11)
    prefix = tmp_path_factory.mktemp("eig") / "sim"
    info = write_eigenstrat(ts, prefix)
    return spec, ts, prefix, info


def test_geno_rows_have_one_char_per_individual(exported):
    spec, ts, prefix, info = exported
    lines = open(f"{prefix}.geno").read().splitlines()
    assert len(lines) == info["n_snps"] > 1000
    assert {len(l) for l in lines} == {info["n_ind"]} == {4 * len(spec.leaves)}
    assert set("".join(lines)) <= set("012")


def test_positions_are_real_and_span_the_sequence(exported):
    spec, ts, prefix, info = exported
    snp = [l.split("\t") for l in open(f"{prefix}.snp").read().splitlines()]
    pos = np.array([int(r[3]) for r in snp])
    assert len(snp) == info["n_snps"]
    assert np.all(np.diff(pos) > 0)
    assert pos[0] < 0.01 * L and pos[-1] > 0.99 * L
    # they are the tree sequence's own site positions, not a made-up grid
    assert set(pos) <= set(ts.sites_position.astype(int))
    assert {r[2] for r in snp} == {"0.0"}   # no genetic map, so 2 Mb blocks


def test_ind_labels_follow_msprime_sample_order(exported):
    spec, ts, prefix, info = exported
    labels = [l.split("\t")[2] for l in open(f"{prefix}.ind").read().splitlines()]
    expected = [ts.population(ts.node(ind.nodes[0]).population).metadata["name"]
                for ind in ts.individuals()]
    assert labels == expected == [leaf for leaf in spec.leaves for _ in range(4)]


def test_geno_values_are_reference_allele_counts(exported):
    spec, ts, prefix, info = exported
    # recompute every cell from the haplotypes, independently of the writer
    pos = {int(l.split("\t")[3]) for l in open(f"{prefix}.snp")}
    pairs = [ind.nodes for ind in ts.individuals()]
    expected = []
    for v in ts.variants():
        if int(v.site.position) in pos:
            expected.append("".join(str(2 - v.genotypes[a] - v.genotypes[b]) for a, b in pairs))
    assert open(f"{prefix}.geno").read().splitlines() == expected
