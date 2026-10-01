import pytest

from admixbench import admixtools
from admixbench.benchmark import run_replicate

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not admixtools.r_available(), reason="Rscript with admixtools and jsonlite not available"),
]


def test_easy_end_to_end(tmp_path):
    row = run_replicate("easy", {"graph": "easy"}, seed=1, cache_root=tmp_path,
                        numstart=5, sim={"sequence_length": 2e7})
    print(row)
    assert row["n_blocks"] > 1
    assert row["gate_passed"], f"true graph does not fit its own data: worst |Z| {row['qpgraph_worst_residual']}"
    assert row["n_topologies"] > 1
    assert row["find_graphs_best_score"] <= row["qpgraph_true_score"] + 1e-6 or row["set_distance"] > 0
    if row["topology_equal"]:
        # same topology, same data: same score and the same predicted f2
        assert row["find_graphs_best_score"] == pytest.approx(row["qpgraph_true_score"], rel=1e-4)
        assert row["covariance_distance"] < 1e-3
    # a second call reuses every cached stage
    again = run_replicate("easy", {"graph": "easy"}, seed=1, cache_root=tmp_path,
                          numstart=5, sim={"sequence_length": 2e7})
    assert again["t_find_graphs"] < 5 and again["find_graphs_best_score"] == row["find_graphs_best_score"]
