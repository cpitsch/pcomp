from collections import Counter
import numpy as np
from pytest import fixture
from pcomp.emd.approximations.comparing_stars.comparing_stars import (
    DiGraph,
    GraphNode,
    Star,
    extract_star_representation,
    _normalize_graphs,
    graph_edit_distance_stars,
    star_graph_edit_distance,
)


@fixture
def graph_1() -> DiGraph:
    """Graph 1 from the paper running example (See Figure 2.)"""

    a = GraphNode("a")
    b = GraphNode("b")
    c = GraphNode("c")
    d = GraphNode("d")

    # "Undirected"
    edges = [(a, b), (b, a), (a, d), (d, a), (b, c), (c, b), (d, c), (c, d)]

    return DiGraph(nodes=[a, b, c, d], edges=edges)


@fixture
def graph_2() -> DiGraph:
    """Graph 2 from the paper running example (See Figure 2.)"""

    a = GraphNode("a")
    b = GraphNode("b")
    c = GraphNode("c")

    edges = [(a, b), (b, a), (a, c), (c, a), (b, c), (c, b)]

    return DiGraph(nodes=[a, b, c], edges=edges)


@fixture
def graph_fig_4a() -> DiGraph:
    """The graph in Figure 4a of the paper."""

    a = GraphNode("a")
    b = GraphNode("b")
    c = GraphNode("c")
    d = GraphNode("d")
    e = GraphNode("e")
    f = GraphNode("f")

    edges = [
        (a, b),
        (a, c),
        (b, a),
        (b, c),
        (c, a),
        (c, b),
        (c, d),
        (c, e),
        (c, f),
        (d, c),
        (e, c),
        (f, c),
    ]

    return DiGraph(nodes=[a, b, c, d, e, f], edges=edges)


@fixture
def graph_fig_4b() -> DiGraph:
    """The graph in Figure 4b of the paper."""
    a = GraphNode("a")
    b = GraphNode("b")
    c = GraphNode("c")
    i = GraphNode("i")

    edges = [
        (a, b),
        (a, c),
        (a, i),
        (b, a),
        (b, c),
        (b, i),
        (c, a),
        (c, b),
        (c, i),
        (i, a),
        (i, b),
        (i, c),
    ]

    return DiGraph(nodes=[a, b, c, i], edges=edges)


@fixture
def graph_fig_4c() -> DiGraph:
    """The graph in Figure 4c of the paper."""
    a = GraphNode("a")
    b = GraphNode("b")
    c = GraphNode("c")
    n = GraphNode("n")

    edges = [(a, b), (a, c), (b, a), (b, c), (c, a), (c, b), (c, n), (n, c)]

    return DiGraph(nodes=[a, b, c, n], edges=edges)


def test_star_extraction_graph_1(graph_1):
    """Test that the star extraction works correctly on graph 1."""
    nodes = graph_1.nodes

    node_a = nodes[0]
    node_b = nodes[1]
    node_c = nodes[2]
    node_d = nodes[3]

    stars = extract_star_representation(graph_1)

    assert stars == Counter(
        [
            Star(node_a, frozenset([node_b, node_d])),
            Star(node_b, frozenset([node_a, node_c])),
            Star(node_c, frozenset([node_b, node_d])),
            Star(node_d, frozenset([node_a, node_c])),
        ]
    )


def test_star_extraction_graph_2(graph_2):
    """Test that the star extraction works correctly on graph 2."""

    nodes = graph_2.nodes
    node_a = nodes[0]
    node_b = nodes[1]
    node_c = nodes[2]

    stars = extract_star_representation(graph_2)

    assert stars == Counter(
        [
            Star(node_a, frozenset([node_b, node_c])),
            Star(node_b, frozenset([node_a, node_c])),
            Star(node_c, frozenset([node_a, node_b])),
        ]
    )


def test_graph_edit_distance_stars_mapping_correct(graph_1, graph_2):
    """Check that the mapping computed by the implementation matches the one in the example in the paper."""

    graph_1, graph_2 = _normalize_graphs(graph_1, graph_2)
    node_a_1 = graph_1.nodes[0]
    node_b_1 = graph_1.nodes[1]
    node_c_1 = graph_1.nodes[2]
    node_d_1 = graph_1.nodes[3]

    node_a_2 = graph_2.nodes[0]
    node_b_2 = graph_2.nodes[1]
    node_c_2 = graph_2.nodes[2]
    node_epsilon_2 = graph_2.nodes[3]

    # Extract stars
    stars_1 = extract_star_representation(graph_1)
    stars_2 = extract_star_representation(graph_2)

    # Compute edit distance
    _, permutation_matrix = graph_edit_distance_stars(
        stars_1, stars_2, graph_1, graph_2
    )

    # Check that the permutation matrix is correct
    correct_permutation_matrix = np.zeros((len(graph_1.nodes), len(graph_2.nodes)))
    graph_1_index = graph_1.node_ordering()
    graph_2_index = graph_2.node_ordering()

    correct_permutation_matrix[  # a -> a
        graph_1_index.index(node_a_1), graph_2_index.index(node_a_2)
    ] = 1
    correct_permutation_matrix[  # b -> b
        graph_1_index.index(node_b_1), graph_2_index.index(node_b_2)
    ] = 1
    correct_permutation_matrix[  # c -> c
        graph_1_index.index(node_c_1), graph_2_index.index(node_c_2)
    ] = 1
    correct_permutation_matrix[  # d -> epsilon
        graph_1_index.index(node_d_1), graph_2_index.index(node_epsilon_2)
    ] = 1

    assert np.array_equal(permutation_matrix, correct_permutation_matrix)


def test_ged_bounds_4a_4c(graph_fig_4a, graph_fig_4c):
    """Check that the bounds computed by the implementation satisfy the example "real" GED's listed in the paper."""

    # Graphs Fig 4a and Fig 4c - Real distance is 5
    lower_bound_ac, upper_bound_ac = star_graph_edit_distance(
        graph_fig_4a, graph_fig_4c
    )
    assert lower_bound_ac <= 5
    # In the hundreds of runs, the following assertion has failed once - need to investigate
    assert 5 <= upper_bound_ac


def test_ged_bounds_4b_4c(graph_fig_4b, graph_fig_4c):
    """Check that the bounds computed by the implementation satisfy the example "real" GED's listed in the paper."""

    # Graphs Fig 4b and Fig 4c - Real distance is 3
    lower_bound_bc, upper_bound_bc = star_graph_edit_distance(
        graph_fig_4b, graph_fig_4c
    )

    assert lower_bound_bc <= 3
    # In the hundreds of runs, the following assertion has also failed once - need to investigate
    # Likely due to identically labeled nodes being considered the same one.
    assert 3 <= upper_bound_bc
