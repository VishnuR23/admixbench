"""Simulate genotypes under a GraphSpec with msprime."""
from __future__ import annotations

import msprime

from .graphspec import Split


def build_demography(spec):
    """msprime Demography for ``spec``.

    Every population, leaf and ancestral, is declared before any event is
    added. Events are added in increasing time order. msprime runs backwards
    in time, and an earlier event added after a later one fails with
    "derived population must be active".
    """
    d = msprime.Demography()
    for p in spec.populations:
        d.add_population(name=p.name, initial_size=p.Ne)
    for e in spec.events():
        if isinstance(e, Split):
            d.add_population_split(time=e.time, derived=list(e.derived), ancestral=e.ancestral)
        else:
            d.add_admixture(time=e.time, derived=e.derived, ancestral=list(e.sources),
                            proportions=list(e.proportions))
    return d


def simulate(spec, samples_per_pop=10, sequence_length=1e8, recombination_rate=1e-8,
             mutation_rate=1.25e-8, seed=1):
    """Tree sequence with ``samples_per_pop`` diploids from every leaf.

    The rates are measured human values and stay fixed. ``sequence_length``
    and ``samples_per_pop`` are the knobs; 1e8 bp gives roughly 500k SNPs.
    Mutations use a binary model, so every site has two alleles. msprime
    needs 1 <= seed < 2**32.
    """
    ts = msprime.sim_ancestry(
        samples={leaf: samples_per_pop for leaf in spec.leaves},
        demography=build_demography(spec), sequence_length=sequence_length,
        recombination_rate=recombination_rate, ploidy=2, random_seed=seed)
    return msprime.sim_mutations(ts, rate=mutation_rate, model=msprime.BinaryMutationModel(),
                                 random_seed=seed)
