import matplotlib

matplotlib.use("Agg")

from admixbench.graphspec import easy_graph, random_graph
from admixbench.plotting import layered_positions, plot_comparison


def test_layout_puts_leaves_on_the_bottom_row():
    g = random_graph(8, 4, 2).to_networkx()
    pos = layered_positions(g)
    leaf_y = {pos[n][1] for n in g if g.out_degree(n) == 0}
    assert len(leaf_y) == 1 and leaf_y.pop() == min(y for _, y in pos.values())


def test_plot_comparison_draws_both_panels():
    fig = plot_comparison(easy_graph().to_networkx(), random_graph(5, 1, 1).to_networkx())
    assert len(fig.axes) == 2
