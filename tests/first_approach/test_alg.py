from pcomp.first_approach import (
    AnnotatedGraph,
    Event,
    calculate_behavior_graph,
    compare_processes,
    discretize_populations,
    extract_representation,
    NodeAnnotation,
    EdgeAnnotation,
)
from pcomp.utils import constants
from pcomp.utils.utils import split_log_cases
from tests.utils.fixtures import simple_event_log, event_log
import pandas as pd
import pytest


def test_behavior_graph_simple(simple_event_log):
    case1 = simple_event_log[simple_event_log[constants.DEFAULT_TRACEID_KEY] == "case1"]

    graph = calculate_behavior_graph(case1)
    vertices = [
        Event(activity="a", index=0),
        Event(activity="b", index=1),
        Event(activity="c", index=2),
        Event(activity="d", index=3),
    ]

    # Expected graph:
    #
    #         ┌───┐
    #   ┌────►│ b ├─────┐
    #   │     └───┘     ▼
    # ┌─┴─┐           ┌───┐
    # │ a │           │ d │
    # └─┬─┘           └───┘
    #   │     ┌───┐     ▲
    #   └────►│ c ├─────┘
    #         └───┘
    expected = AnnotatedGraph(
        V=set(vertices),
        E={
            (vertices[0], vertices[1]),  # (a, b)
            (vertices[0], vertices[2]),  # (a, c)
            (vertices[1], vertices[3]),  # (b, d)
            (vertices[2], vertices[3]),  # (c, d)
        },
    )
    assert graph == expected


def test_behavior_graph_complex_lifecycles(event_log):
    case1 = event_log[event_log[constants.DEFAULT_TRACEID_KEY] == "case1"]

    graph = calculate_behavior_graph(case1)
    vertices = [
        Event(activity="a", index=0),
        Event(activity="b", index=2),  # B starts before C, but completes after
        Event(activity="c", index=1),
        Event(activity="d", index=3),
    ]

    # Expected graph:
    # ┌───┐      ┌───┐
    # │ a ├─────►│ c ├─────┐
    # └───┘      └───┘     │
    #                      │
    #                      ▼
    # ┌───┐              ┌───┐
    # │ b ├─────────────►│ d │
    # └───┘              └───┘
    expected = AnnotatedGraph(
        V=set(vertices),
        E={
            (vertices[0], vertices[2]),  # (a, c)
            (vertices[1], vertices[3]),  # (b, d)
            (vertices[2], vertices[3]),  # (c, d)
        },
    )
    assert graph == expected


def test_extract_representation(simple_event_log: pd.DataFrame):
    rep: list[AnnotatedGraph] = extract_representation(simple_event_log)
    assert len(rep) == 1

    # Structure of the graph already tested in test_behavior_graph_simple
    graph = rep[0]

    vertices = [  # For simplicity of the assertions below
        Event(activity="a", index=0),
        Event(activity="b", index=1),
        Event(activity="c", index=2),
        Event(activity="d", index=3),
    ]
    assert graph.V == set(vertices)

    # Check node annotations
    assert graph.get_node_annotation(vertices[0]) == NodeAnnotation(
        activity="a", service_time=pd.Timedelta(days=0).total_seconds()
    )
    assert graph.get_node_annotation(vertices[1]) == NodeAnnotation(
        activity="b", service_time=pd.Timedelta(days=1).total_seconds()
    )
    assert graph.get_node_annotation(vertices[2]) == NodeAnnotation(
        activity="c", service_time=pd.Timedelta(days=1).total_seconds()
    )
    assert graph.get_node_annotation(vertices[3]) == NodeAnnotation(
        activity="d", service_time=pd.Timedelta(days=0).total_seconds()
    )

    # Check edge annotations (waiting times)
    assert graph.get_edge_annotation((vertices[0], vertices[1])) == EdgeAnnotation(
        waiting_time=pd.Timedelta(days=1).total_seconds()
    )  # (a, b)
    assert graph.get_edge_annotation((vertices[0], vertices[2])) == EdgeAnnotation(
        waiting_time=pd.Timedelta(days=1).total_seconds()
    )  # (a, c)
    assert graph.get_edge_annotation((vertices[1], vertices[3])) == EdgeAnnotation(
        waiting_time=pd.Timedelta(days=1).total_seconds()
    )  # (b, d)
    assert graph.get_edge_annotation((vertices[2], vertices[3])) == EdgeAnnotation(
        waiting_time=pd.Timedelta(days=1).total_seconds()
    )  # (c, d)


def test_discretize_populations(event_log):
    """Test that the populations are correctly discretized"""

    # Each population will hold exactly one trace
    log1 = event_log[event_log[constants.DEFAULT_TRACEID_KEY] == "case1"]
    log2 = event_log[event_log[constants.DEFAULT_TRACEID_KEY] == "case2"]

    pop1 = extract_representation(log1)
    pop2 = extract_representation(log2)

    with pytest.warns(
        UserWarning,
        match=r"Feature \d is constant and will be replaced with \d\.",
    ):
        transformed_pop1, transformed_pop2 = discretize_populations(pop1, pop2, 2)

    # Problem: Discretizing doesnt really do much here because we have only two distinct values...
    # Also: Is the binning deterministic?


def test_identical_event_log(event_log):
    with pytest.warns(
        UserWarning,
        match=r"Feature \d is constant and will be replaced with \d\.",
    ):
        p = compare_processes(event_log, event_log)
        # The same process should have the exact same representations -> 1.0 p-value
        assert p == 1.0

    # Problem: the created graphs could be too different due to many edges, etc.
    # --> Many different times, unlikely to have identical graphs in two populations
    # Will need to see how this behaves on real life event logs


def test_split_event_log(event_log):
    log1, log2 = split_log_cases(event_log, 0.5)
    with pytest.warns(
        UserWarning,
        match=r"Feature \d is constant and will be replaced with \d\.",
    ):
        assert compare_processes(log1, log2) == 1.0
