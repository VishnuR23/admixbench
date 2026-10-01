"""GraphSpec: the single source of truth for a simulated population history.

Times follow msprime: 0 is the present and larger times are older. A spec has
populations and two kinds of events.

- Split(time, derived, ancestral). Going back in time, every population in
  ``derived`` merges into ``ancestral``.
- Admixture(time, derived, sources, proportions). Going back in time, the
  lineages of ``derived`` move into the two sources with the given proportions.

Each population is one lineage. A leaf lives from time 0. An internal
population starts at the first event that uses it as an ancestral population
or a source. Every population ends at the one event where it is ``derived``,
except the root, which never ends.

The msprime demography (simulate.build_demography) and the NetworkX graph
(GraphSpec.to_networkx) are both derived from this object. Neither is ever
written by hand.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class Population:
    name: str
    Ne: float = 10_000.0
    leaf: bool = True


@dataclass(frozen=True)
class Split:
    time: float
    derived: tuple
    ancestral: str

    def __post_init__(self):
        object.__setattr__(self, "derived", tuple(self.derived))


@dataclass(frozen=True)
class Admixture:
    time: float
    derived: str
    sources: tuple
    proportions: tuple

    def __post_init__(self):
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "proportions", tuple(self.proportions))


@dataclass(frozen=True)
class GraphSpec:
    populations: tuple
    splits: tuple
    admixtures: tuple
    outgroup: str
    name: str = ""

    def __post_init__(self):
        for f in ("populations", "splits", "admixtures"):
            object.__setattr__(self, f, tuple(getattr(self, f)))
        self._validate()

    # ------------------------------------------------------------- accessors
    @property
    def leaves(self):
        return tuple(p.name for p in self.populations if p.leaf)

    def population(self, name):
        return next(p for p in self.populations if p.name == name)

    def events(self):
        """All events in increasing time order, the order msprime needs."""
        return sorted(self.splits + self.admixtures, key=lambda e: e.time)

    # ------------------------------------------------------------ validation
    def _validate(self):
        names = [p.name for p in self.populations]
        if len(set(names)) != len(names):
            raise ValueError("population names must be unique")
        declared = set(names)

        for kind, evs in (("splits", self.splits), ("admixtures", self.admixtures)):
            times = [e.time for e in evs]
            if any(b <= a for a, b in zip(times, times[1:])):
                raise ValueError(f"{kind} must be strictly increasing in time, got {times}")
        all_times = sorted(e.time for e in self.splits + self.admixtures)
        if any(b <= a for a, b in zip(all_times, all_times[1:])):
            raise ValueError(f"event times must be strictly increasing across all events, got {all_times}")
        if all_times and all_times[0] <= 0:
            raise ValueError("event times must be positive")

        for e in self.events():
            refs = (*e.derived, e.ancestral) if isinstance(e, Split) else (e.derived, *e.sources)
            missing = [r for r in refs if r not in declared]
            if missing:
                raise ValueError(f"event at t={e.time} references undeclared populations {missing}")

        for a in self.admixtures:
            if len(a.sources) != 2 or len(a.proportions) != 2:
                raise ValueError(f"admixture at t={a.time} needs exactly two sources and two proportions")
            if len(set(a.sources)) != 2 or a.derived in a.sources:
                raise ValueError(f"admixture at t={a.time}: sources must be two distinct other populations")
            if not all(0 < p < 1 for p in a.proportions):
                raise ValueError(f"admixture at t={a.time}: proportions must lie in (0, 1)")
            if not math.isclose(sum(a.proportions), 1.0, abs_tol=1e-9):
                raise ValueError(f"admixture at t={a.time}: proportions {a.proportions} do not sum to 1")

        if self.outgroup not in declared or not self.population(self.outgroup).leaf:
            raise ValueError(f"outgroup {self.outgroup!r} must be a declared leaf")

        # lifetimes: each population is derived at most once; exactly one root
        end = {}
        for e in self.events():
            for d in (e.derived if isinstance(e, Split) else (e.derived,)):
                if d in end:
                    raise ValueError(f"population {d!r} is derived in more than one event")
                end[d] = e.time
        roots = [n for n in names if n not in end]
        if len(roots) != 1:
            raise ValueError(f"need exactly one root (a population never derived), got {roots}")
        if self.population(roots[0]).leaf:
            raise ValueError(f"the root {roots[0]!r} must be an internal population")

        start = self._start_times()
        for p in self.populations:
            if p.name not in start:
                raise ValueError(f"internal population {p.name!r} is never used")
        for e in self.events():
            users = (e.ancestral,) if isinstance(e, Split) else e.sources
            derived = e.derived if isinstance(e, Split) else (e.derived,)
            for u in users:
                if end.get(u, math.inf) <= e.time:
                    raise ValueError(f"population {u!r} is not alive at t={e.time}")
            for d in derived:
                if start[d] >= e.time:
                    raise ValueError(f"population {d!r} has no lineage before t={e.time}")

        g = self._raw_graph()
        for a in self.admixtures:
            parents = [next(iter(g.predecessors(u)), None) for u in g.predecessors(f"{a.derived}_adm")]
            if parents[0] is not None and parents[0] == parents[1]:
                raise ValueError(f"admixture at t={a.time}: sources {a.sources} are siblings, "
                                 "so the admixture signal is not detectable")

    def _start_times(self):
        start = {p.name: 0.0 for p in self.populations if p.leaf}
        for e in self.events():
            for u in ((e.ancestral,) if isinstance(e, Split) else e.sources):
                start.setdefault(u, e.time)
        return start

    # ---------------------------------------------------------------- graphs
    def _raw_graph(self):
        """Graph with one node per (population, event time). Pass-through nodes
        (one parent, one child) are kept here so the sibling check can see them."""
        pops = {p.name: p for p in self.populations}
        start = self._start_times()
        chain = {n: {} for n in pops}            # population -> {time: node}
        for n, p in pops.items():
            if p.leaf:
                chain[n][0.0] = n
        for e in self.events():
            for u in ((e.ancestral,) if isinstance(e, Split) else e.sources):
                if e.time not in chain[u]:
                    chain[u][e.time] = u if e.time == start[u] and not pops[u].leaf else f"{u}@{e.time:g}"

        g = nx.DiGraph()
        for n, nodes in chain.items():
            for t, node in nodes.items():
                g.add_node(node, leaf=(pops[n].leaf and t == 0.0), time=t, pop=n)
            ts = sorted(nodes)
            for lo, hi in zip(ts, ts[1:]):
                g.add_edge(nodes[hi], nodes[lo], length=(hi - lo) / (2 * pops[n].Ne), admixture=False)

        for e in self.events():
            if isinstance(e, Split):
                for d in e.derived:
                    top = max(chain[d])
                    g.add_edge(chain[e.ancestral][e.time], chain[d][top],
                               length=(e.time - top) / (2 * pops[d].Ne), admixture=False)
            else:
                adm = f"{e.derived}_adm"
                g.add_node(adm, leaf=False, time=e.time, pop=e.derived)
                for s, w in zip(e.sources, e.proportions):
                    g.add_edge(chain[s][e.time], adm, length=0.0, admixture=True, proportion=w)
                top = max(chain[e.derived])
                g.add_edge(adm, chain[e.derived][top],
                           length=(e.time - top) / (2 * pops[e.derived].Ne), admixture=False)
        return g

    def to_networkx(self):
        """DiGraph with drift edges (``length``) and admixture edges
        (``admixture=True``, ``proportion``, ``length``). Nodes carry ``leaf``.

        Each admixed population X gets an internal node ``X_adm`` with two
        admixture edges in and one drift edge down to X. A source lineage that
        exists only to feed an admixture (a ghost) is contracted, so its drift
        sits on the admixture edge. Every other admixture edge has length 0.
        """
        g = self._raw_graph()
        for n in list(g):
            if g.in_degree(n) == 1 and g.out_degree(n) == 1 and not g.nodes[n]["leaf"]:
                (p,), (c,) = g.predecessors(n), g.successors(n)
                attrs = dict(g.edges[n, c])
                attrs["length"] += g.edges[p, n]["length"]
                g.remove_node(n)
                g.add_edge(p, c, **attrs)
        return g


# -------------------------------------------------------------- fixed graphs
def _tree_pops(leaves, internal, Ne):
    return [Population(n, Ne) for n in leaves] + [Population(n, Ne, leaf=False) for n in internal]


def trivial_graph(Ne=10_000.0):
    """Five ingroup leaves and an outgroup, no admixture: ((P1,P2),(P3,(P4,P5))),O."""
    return GraphSpec(
        populations=_tree_pops(["P1", "P2", "P3", "P4", "P5", "O"], ["A45", "A12", "A345", "A", "R"], Ne),
        splits=[Split(1500, ("P4", "P5"), "A45"), Split(2000, ("P1", "P2"), "A12"),
                Split(3000, ("P3", "A45"), "A345"), Split(4000, ("A12", "A345"), "A"),
                Split(6000, ("A", "O"), "R")],
        admixtures=[], outgroup="O", name="trivial")


def easy_graph(Ne=10_000.0):
    """The walkthrough case. Five ingroup leaves and an outgroup. P3 is founded
    at t=100 by srcA (0.35, a ghost sister of P2) and srcB (0.65, a ghost
    sister of P5). The recent date keeps f3(P3; P2, P5) clearly negative."""
    return GraphSpec(
        populations=_tree_pops(["P1", "P2", "P3", "P4", "P5", "O"],
                               ["srcA", "srcB", "A_L2", "A_R2", "A_L", "A_R", "A", "R"], Ne),
        splits=[Split(2000, ("P2", "srcA"), "A_L2"), Split(2200, ("P5", "srcB"), "A_R2"),
                Split(4000, ("P1", "A_L2"), "A_L"), Split(4200, ("P4", "A_R2"), "A_R"),
                Split(6000, ("A_L", "A_R"), "A"), Split(8000, ("A", "O"), "R")],
        admixtures=[Admixture(100, "P3", ("srcA", "srcB"), (0.35, 0.65))],
        outgroup="O", name="easy")


# ------------------------------------------------------------- random graphs
def random_graph(n_leaves, n_admix, seed, Ne=10_000.0, gap=(200, 1000)):
    """Random valid spec with ``n_leaves`` ingroup leaves P1..Pn plus outgroup O.

    Built backwards in time. Each step is either a binary split of two live
    lineages or an admixture, where one live lineage is founded by two other
    live lineages. Gaps between events are uniform integers in ``gap``
    generations, so times strictly increase toward the root. The outgroup
    joins last, at the root, and is never part of an admixture. Minor
    proportions are uniform in [0.1, 0.4]. Two lineages that were sources of
    the same admixture are never merged by the next split, which would make
    them siblings.

    An admixture needs four live lineages (the derived one, two sources, and
    one more so the sources need not become siblings), so n_admix <= n_leaves - 3.
    """
    if n_admix and n_admix > n_leaves - 3:
        raise ValueError(f"n_admix={n_admix} too large for n_leaves={n_leaves} (max {n_leaves - 3})")
    rng = np.random.default_rng(seed)
    leaves = [f"P{i}" for i in range(1, n_leaves + 1)]
    alive = list(leaves)
    last = {p: None for p in alive}         # most recent event each live lineage took part in
    splits, admixtures, internal = [], [], []
    splits_left, admix_left = n_leaves - 1 - n_admix, n_admix
    t = 0

    def feasible(k, a):
        return a == 0 or a <= k - 3

    while splits_left + admix_left:
        t += int(rng.integers(gap[0], gap[1] + 1))
        k = len(alive)
        moves = []
        if splits_left and feasible(k - 1, admix_left):
            moves.append(("split", splits_left))
        if admix_left and k >= 4 and feasible(k - 1, admix_left - 1):
            moves.append(("admix", admix_left))
        w = np.array([m[1] for m in moves], float)
        move = moves[rng.choice(len(moves), p=w / w.sum())][0]

        if move == "split":
            pairs = [(a, b) for i, a in enumerate(alive) for b in alive[i + 1:]
                     if last[a] is None or last[a] != last[b]]
            a, b = pairs[rng.integers(len(pairs))]
            anc = f"A{len(internal) + 1}"
            internal.append(anc)
            splits.append(Split(t, (a, b), anc))
            alive = [x for x in alive if x not in (a, b)] + [anc]
            last[anc] = ("split", t)
            splits_left -= 1
        else:
            d, s1, s2 = (alive[i] for i in rng.choice(k, size=3, replace=False))
            p = round(float(rng.uniform(0.1, 0.4)), 3)
            admixtures.append(Admixture(t, d, (s1, s2), (p, round(1 - p, 3))))
            alive.remove(d)
            last[s1] = last[s2] = ("admix", t)
            admix_left -= 1

    t += int(rng.integers(gap[0], gap[1] + 1))
    splits.append(Split(t, (alive[0], "O"), "R"))
    return GraphSpec(
        populations=_tree_pops(leaves + ["O"], internal + ["R"], Ne),
        splits=splits, admixtures=admixtures, outgroup="O",
        name=f"random_n{n_leaves}_a{n_admix}_s{seed}")
