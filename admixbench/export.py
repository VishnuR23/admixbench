"""Write a tree sequence as EIGENSTRAT text files for ADMIXTOOLS 2.

ADMIXTOOLS 2 reads EIGENSTRAT or binary PLINK. It does not read text PLINK
(.ped/.map), so that format is not written here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def write_eigenstrat(ts, prefix):
    """Write ``prefix``.geno, .snp and .ind and return {"n_snps", "n_ind"}.

    .geno has one row per SNP and one character per diploid individual: the
    number of reference (ancestral) alleles, 0/1/2. The two haplotype columns
    of each individual are added together.

    .snp uses the real site positions. ADMIXTOOLS 2 cuts the genome into
    jackknife blocks by position. The genetic position column is written as 0,
    so it falls back to 2 Mb blocks of physical position. Positions squeezed
    into a short range would give one block, a singular covariance matrix, and
    the find_graphs error ``all(!is.na(precomp$ppinv)) is not TRUE``.

    .ind lists individuals in msprime's sample order, labelled by population.
    Sites that are not polymorphic in the sample are dropped.
    """
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    samples = list(ts.samples())
    col = {u: i for i, u in enumerate(samples)}
    inds = [ind for ind in ts.individuals() if any(u in col for u in ind.nodes)]
    if any(len(ind.nodes) != 2 for ind in inds):
        raise ValueError("write_eigenstrat expects diploid individuals")

    G = ts.genotype_matrix()                                   # sites x haplotypes, 0/1
    if G.max(initial=0) > 1:
        raise ValueError("write_eigenstrat expects biallelic sites")
    dosage = np.stack([G[:, col[a]] + G[:, col[b]] for a, b in (ind.nodes for ind in inds)], axis=1)
    total = dosage.sum(axis=1)
    keep = (total > 0) & (total < 2 * len(inds))
    ref_count = (2 - dosage[keep]).astype(np.uint8)
    pos = ts.sites_position[keep].astype(np.int64)
    if np.any(np.diff(pos) <= 0):
        raise ValueError("site positions are not strictly increasing after casting to int")

    rows = np.empty((ref_count.shape[0], ref_count.shape[1] + 1), dtype=np.uint8)
    rows[:, :-1] = ref_count + ord("0")
    rows[:, -1] = ord("\n")
    Path(f"{prefix}.geno").write_bytes(rows.tobytes())

    with open(f"{prefix}.snp", "w") as fh:
        fh.writelines(f"rs{i + 1}\t1\t0.0\t{p}\tA\tG\n" for i, p in enumerate(pos))

    pop_name = {p.id: p.metadata["name"] for p in ts.populations()}
    count = {}
    with open(f"{prefix}.ind", "w") as fh:
        for ind in inds:
            pop = pop_name[ts.node(ind.nodes[0]).population]
            count[pop] = count.get(pop, 0) + 1
            fh.write(f"{pop}_{count[pop]}\tU\t{pop}\n")

    return {"n_snps": int(len(pos)), "n_ind": len(inds)}
