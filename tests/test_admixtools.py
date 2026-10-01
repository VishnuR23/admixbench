import copy

import pytest

from admixbench import admixtools
from admixbench.admixtools import edges_to_networkx, parse_find_graphs
from admixbench.graphspec import easy_graph
from admixbench.metrics import topology_equality


def row(score=5.0, h="abc", edges=None):
    # the shape write_json produces for one find_graphs row (dataframe="columns")
    e = edges or {
        "from": ["R", "R", "Rr", "Rr", "Rrl", "Rrl", "Rrr", "admix", "Rrr2", "Rrr2"],
        "to": ["O", "Rr", "Rrl", "Rrr", "P1", "admix", "Rrr2", "P3", "P2", "admix"],
        "type": ["edge", "edge", "edge", "edge", "edge", "admix", "edge", "edge", "edge", "admix"],
        "weight": [.02, .02, .01, .01, .01, .4, .01, .007, .01, .6],
        "low": [None] * 5 + [.4] + [None] * 3 + [.6],
        "high": [None] * 5 + [.4] + [None] * 3 + [.6],
    }
    return {"score": score, "hash": h, "run": 1, "edges": e}


def test_parse_dedupes_by_hash_and_sorts_by_score():
    rows = parse_find_graphs([row(9.0, "a"), row(5.0, "b"), row(7.0, "a")])
    assert [(r["hash"], r["score"]) for r in rows] == [("b", 5.0), ("a", 7.0)]
    assert rows[0]["edges"][5] == {"from": "Rrl", "to": "admix", "type": "admix", "weight": .4}


@pytest.mark.parametrize("mutate,match", [
    (lambda r: r["edges"].pop("weight"), "edge columns"),
    (lambda r: r["edges"].__setitem__("wt", r["edges"].pop("weight")), "edge columns"),
    (lambda r: r["edges"].__setitem__("extra", [0] * 10), "edge columns"),
    (lambda r: r["edges"]["type"].__setitem__(0, "drift"), "edge types"),
    (lambda r: r["edges"]["to"].pop(), "different lengths"),
    (lambda r: r["edges"]["weight"].__setitem__(2, None), "numbers"),
    (lambda r: r.pop("score"), "missing"),
])
def test_parse_rejects_malformed_or_renamed_columns(mutate, match):
    r = copy.deepcopy(row())
    mutate(r)
    with pytest.raises(ValueError, match=match):
        parse_find_graphs([r])


def test_parse_rejects_empty():
    with pytest.raises(ValueError):
        parse_find_graphs([])


def test_edges_to_networkx():
    g = edges_to_networkx(parse_find_graphs([row()])[0]["edges"])
    assert g.edges["Rrl", "admix"]["admixture"] and g.edges["Rrl", "admix"]["proportion"] == .4
    assert not g.edges["admix", "P3"]["admixture"] and g.edges["admix", "P3"]["length"] == .007
    assert {n for n in g if g.nodes[n]["leaf"]} == {"O", "P1", "P2", "P3"}


def test_r_edges_rename_internal_nodes_and_keep_leaves():
    g = easy_graph().to_networkx()
    rows = admixtools._r_edges(g)
    names = {x for r in rows for x in r}
    assert {"P1", "P2", "P3", "P4", "P5", "O"} <= names
    assert not any("@" in n or "_adm" in n for n in names)
    back = edges_to_networkx([{"from": u, "to": v, "type": "edge", "weight": 1.0} for u, v in rows])
    assert topology_equality(g, back)
