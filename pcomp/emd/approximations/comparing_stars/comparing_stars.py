from collections import Counter
from dataclasses import dataclass
from functools import cache
from scipy.optimize import linear_sum_assignment  # type: ignore
from itertools import product
import numpy as np
from typing import overload as typing_overload, Literal

EPSILON_NODE_LABEL = "pcomp:ged_approx:epsilon"


@dataclass(frozen=True)
class GraphNode:
    label: str
    duration: int = 0


@dataclass
class DiGraph:
    nodes: list[GraphNode]
    edges: list[tuple[GraphNode, GraphNode]]

    def degree(self, node: GraphNode) -> int:
        return len([e for e in self.edges if e[0] == node])

    def max_degree(self) -> int:
        return max(self.degree(v) for v in self.nodes)

    def node_ordering(self) -> list[GraphNode]:
        return sorted(self.nodes, key=id)  # Sort nodes by their identity

    def adjacency_matrix(self) -> np.ndarray:
        ordering = self.node_ordering()
        matrix = np.zeros((len(ordering), len(ordering)))

        for u, v in self.edges:
            matrix[ordering.index(u), ordering.index(v)] = 1

        return matrix


@dataclass(frozen=True)
class Star:
    root: GraphNode
    leaves: frozenset[GraphNode]


@typing_overload
def graph_edit_distance(
    graph_1: DiGraph, graph_2: DiGraph, return_permutations: Literal[True]
) -> tuple[tuple[float, float], tuple[np.ndarray, np.ndarray]]:
    ...


@typing_overload
def graph_edit_distance(
    graph_1: DiGraph, graph_2: DiGraph, return_permutations: Literal[False] = False
) -> tuple[float, float]:
    ...


def graph_edit_distance(
    graph_1: DiGraph, graph_2: DiGraph, return_permutations: bool = False
) -> tuple[float, float] | tuple[tuple[float, float], tuple[np.ndarray, np.ndarray]]:
    """Compute an upper and lower bound on the Graph edit distance of the two graphs
    This is based on the paper "Comparing Sttars: On Approximating Graph Edit Distance" by Zeng et al.
    This algorithm is not defined for self-loops, multi-edges or edge labels. If self-loops exist, an error will be thrown

    Args:
        graph_1 (DiGraph): The first graph
        graph_2 (DiGraph): The second graph
        return_permutations (bool, optional): Whether to also return the permutation matrices yielding the bounds. Defaults to False.

    Returns:
        tuple[float, float]: The computed bounds on the edit distance
        tuple[np.ndarray, np.ndarray]: Additionally, the permutation matrices yielding the bounds. If return_permutations is True.
    """

    if _has_self_loops(graph_1) or _has_self_loops(graph_2):
        raise ValueError("This algorithm is not defined for graphs with self-loops")

    # Make graphs have the same number of nodes
    _normalize_graphs(graph_1, graph_2)

    star_rep_1 = extract_star_representation(graph_1)
    star_rep_2 = extract_star_representation(graph_2)

    mapping_distance, permutation_matrix = graph_edit_distance_stars(
        star_rep_1, star_rep_2, graph_1, graph_2
    )
    coefficient = max(4, max(graph_1.max_degree(), graph_2.max_degree()) + 1)
    lower_bound = mapping_distance / coefficient

    star_edit_distance.cache_clear()

    upper_bound, upper_bound_permutation_matrix = refined_upper_bound(
        graph_1, graph_2, permutation_matrix
    )

    if return_permutations:
        return (lower_bound, upper_bound), (
            permutation_matrix,
            upper_bound_permutation_matrix,
        )
    else:
        return lower_bound, upper_bound


def _has_self_loops(graph: DiGraph) -> bool:
    """Check if the graph contains any self loops (loops of a node to itself)

    Args:
        graph (DiGraph): The graph

    Returns:
        bool: True if any self loops exist
    """
    return any(u == v for u, v in graph.edges)


def _normalize_graphs(graph_1: DiGraph, graph_2: DiGraph) -> None:
    """Normalize the graphs w.r.t. eachother. This is done by adding epsilon ("dummy") nodes to the smaller graph. All epsilon nodes have the same label

    Args:
        graph_1 (DiGraph): The first graph.
        graph_2 (DiGraph): The second graph.
    """
    difference = len(graph_1.nodes) - len(graph_2.nodes)
    if difference > 0:
        graph_2.nodes.extend([GraphNode(EPSILON_NODE_LABEL)] * difference)
    elif difference < 0:
        graph_1.nodes.extend([GraphNode(EPSILON_NODE_LABEL)] * -difference)


def extract_star_representation(graph: DiGraph) -> Counter[Star]:
    """Extract the star representation for a graph. This is a multiset (counter) of stars.
    This is done by extracting a star using each node in the graph as root once.

    Args:
        graph (DiGraph): The graph to extract the star representation from.

    Returns:
        Counter[Star]: A multiset of stars.
    """
    return Counter(
        [
            Star(root=r, leaves=frozenset(u for u, v in graph.edges if v == r))
            for r in graph.nodes
        ]
    )


@cache
def star_edit_distance(star_1: Star, star_2: Star) -> float:
    """Compute the edit distance between two stars.

    Args:
        star_1 (Star): The first star.
        star_2 (Star): The second star.

    Returns:
        float: The edit distance between the two stars.
    """
    t = 0 if star_1.root.label == star_2.root.label else 1
    labels_multiset_1 = Counter([l.label for l in star_1.leaves])
    labels_multiset_2 = Counter([l.label for l in star_2.leaves])

    m = (
        max(labels_multiset_1.total(), labels_multiset_2.total())
        - (labels_multiset_1 & labels_multiset_2).total()
    )

    d = abs(len(star_1.leaves) - len(star_2.leaves)) + m

    return t + d


def graph_edit_distance_stars(
    stars_1: Counter[Star], stars_2: Counter[Star], graph_1: DiGraph, graph_2: DiGraph
) -> tuple[float, np.ndarray]:
    """Compute a lower bound on the graph edit distance by computing a min-weight full matching between the stars found in them.

    Args:
        star_1 (Star): The first star.
        star_2 (Star): The second star.

    Returns:
        float: The edit distance between the two stars.
    """
    nodes_1 = list(stars_1.elements())
    nodes_2 = list(stars_2.elements())

    cost_matrix = np.empty((len(nodes_1), len(nodes_2)), dtype=float)
    for u in nodes_1:
        for v in nodes_2:
            cost_matrix[nodes_1.index(u), nodes_2.index(v)] = star_edit_distance(u, v)

    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=False)
    lower_bound = cost_matrix[row_ind, col_ind].sum()

    # Map star indices to the index of their root
    graph_1_index = graph_1.node_ordering()
    graph_2_index = graph_2.node_ordering()

    new_row_ind = np.array([graph_1_index.index(nodes_1[i].root) for i in row_ind])
    new_col_ind = np.array([graph_2_index.index(nodes_2[i].root) for i in col_ind])

    permutation_matrix = np.zeros((len(nodes_1), len(nodes_2)))
    permutation_matrix[new_row_ind, new_col_ind] = 1

    return lower_bound, permutation_matrix


def calculate_transformation_cost(
    graph_1: DiGraph, graph_2: DiGraph, permutation_matrix: np.ndarray
) -> float:
    """Calculate the cost of transforming graph_1 into graph_2 using the given permutation matrix for edit operations.

    Equation 1 from the paper (Section 3.1)

    Args:
        graph_1 (DiGraph): The first graph.
        graph_2 (DiGraph): The second graph.
        permutation_matrix (np.ndarray): The permutation matrix. A 2D matrix containing a 1 for each pair of nodes that are mapped to eachother.

    Returns:
        float: The transformation cost.
    """

    ordering_1 = graph_1.node_ordering()
    adj_1 = graph_1.adjacency_matrix()
    ordering_2 = graph_2.node_ordering()
    adj_2 = graph_2.adjacency_matrix()

    if len(ordering_1) != len(ordering_2):
        raise ValueError("The graphs must have the same number of nodes")

    return sum(
        (0 if ordering_1[i].label == ordering_1[j].label else 1)
        * permutation_matrix[i, j]
        for i in range(len(ordering_1))
        for j in range(len(ordering_1))
    ) + 0.5 * np.linalg.norm(
        adj_1 - (permutation_matrix * adj_2 * permutation_matrix.T), ord=1
    )


def refined_upper_bound(
    graph_1: DiGraph, graph_2: DiGraph, permutation_matrix: np.ndarray
) -> tuple[float, np.ndarray]:
    """Compute an upper bound on the graph edit distance between the two graphs. Guaranteed to be terminated in 2n + n^2 steps.

    Args:
        graph_1 (DiGraph): The first graph.
        graph_2 (DiGraph): The second graph.
        permutation_matrix (np.ndarray): The permutation matrix yielded for the lower bound.

    Returns:
        tuple[float, np.ndarray]: The upper bound on the graph edit distance, and the permutation matrix yielding it.
    """
    dist = calculate_transformation_cost(graph_1, graph_2, permutation_matrix)

    min_dist = dist
    min_permutation = permutation_matrix.copy()

    graph_1_index = graph_1.node_ordering()

    for u, v in product(graph_1.nodes, graph_1.nodes):
        # Swap their matching
        new_permutation = permutation_matrix.copy()
        index_u = graph_1_index.index(u)
        index_v = graph_1_index.index(v)

        old_u = permutation_matrix[index_u, :]
        new_permutation[index_u, :] = new_permutation[index_v, :]
        new_permutation[index_v, :] = old_u

        new_dist = calculate_transformation_cost(graph_1, graph_2, permutation_matrix)

        if new_dist < min_dist:
            min_dist = new_dist
            min_permutation = new_permutation

    if min_dist < dist:
        return refined_upper_bound(graph_1, graph_2, min_permutation)

    return min_dist, min_permutation


@cache
def timed_star_graph_edit_distance(
    graph_1: DiGraph, graph_2: DiGraph, time_scaling_factor: int = 1
) -> tuple[float, float]:
    """Calculate an approximation of the graph edit distance between two graphs.
    This is done by first computing the upper and lower bound using the "Comparing Stars" algorithm,
    then adding the time difference between nodes that are mapped to eachother to the lower and upper bound respectively.

    Args:
        graph_1 (DiGraph): The first graph.
        graph_2 (DiGraph): The second graph.
        time_scaling_factor (int, optional): The factor to use to scale time differences. Used to normalize time differences so that
            the impact of label differences are not diminished. For instance, if the time values go from 0 to 5, a time_scaling_factor
            of `1/5` would make a time difference cost at most 1. Defaults to 1.

    Returns:
        tuple[float, float]: The adjusted lower and upper bound on the graph edit distance.
    """
    (lower_bound, upper_bound), (perm_lower, perm_upper) = graph_edit_distance(
        graph_1, graph_2, True
    )

    lower_bound_time_cost = 0
    upper_bound_time_cost = 0

    graph_1_index = graph_1.node_ordering()
    graph_2_index = graph_2.node_ordering()

    for u in graph_1.nodes:
        for v in graph_2.nodes:
            if perm_lower[graph_1_index.index(u), graph_2_index.index(v)] == 1:
                # Since default is 0, we don't need to check if it's an epsilon node
                lower_bound_time_cost += (
                    abs(u.duration - v.duration) * time_scaling_factor
                )
            if perm_upper[graph_1_index.index(u), graph_2_index.index(v)] == 1:
                upper_bound_time_cost += (
                    abs(u.duration - v.duration) * time_scaling_factor
                )
    return lower_bound + lower_bound_time_cost, upper_bound + upper_bound_time_cost
