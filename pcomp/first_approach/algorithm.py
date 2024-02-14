from collections import Counter
from typing import TypeVar

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy.stats import chi2_contingency  # type: ignore
from pcomp.utils import constants, ensure_start_timestamp_column
from pcomp.utils.data import Binner, create_binner
from pcomp.utils.typing import Numpy1DArray

T = TypeVar("T")


@dataclass(frozen=True)
class Event:
    index: int
    activity: str

    def __repr__(self):
        return f"Event({self.activity}, {self.index})"


@dataclass
class BehaviorGraphNode:
    """Used internally in the computation of the behavior graph."""

    timestamp: pd.Timestamp
    event: Event
    is_atomic: bool
    is_start: bool


@dataclass
class NodeAnnotation:
    service_time: float
    activity: str


@dataclass
class EdgeAnnotation:
    waiting_time: float


@dataclass
class AnnotatedGraph:
    V: set[Event]
    E: set[tuple[Event, Event]]

    node_annotations: dict[Event, NodeAnnotation] = field(default_factory=dict)
    edge_annotations: dict[tuple[Event, Event], EdgeAnnotation] = field(
        default_factory=dict
    )

    def annotate_node(self, node: Event, annotation: NodeAnnotation):
        self.node_annotations[node] = annotation

    def annotate_edge(self, edge: tuple[Event, Event], annotation: EdgeAnnotation):
        self.edge_annotations[edge] = annotation

    def get_node_annotation(self, node: Event) -> NodeAnnotation | None:
        return self.node_annotations.get(node, None)

    def get_edge_annotation(self, edge: tuple[Event, Event]) -> EdgeAnnotation | None:
        return self.edge_annotations.get(edge, None)


def calculate_behavior_graph(
    trace: pd.DataFrame,
    activity_key: str = constants.DEFAULT_NAME_KEY,
    timestamp_key: str = constants.DEFAULT_TIMESTAMP_KEY,
    start_timestamp_key: str = constants.DEFAULT_START_TIMESTAMP_KEY,
    lifecycle_key: str = constants.DEFAULT_LIFECYCLE_KEY,
) -> AnnotatedGraph:
    """Extract a behavior graph for a given trace.

    Args:
        trace (pd.DataFrame): The trace to extract the behavior graph from.
        activity_key (str, optional): The name of the column containing the activity label. Defaults to "concept:name".
        timestamp_key (str, optional): The name of the column containing the end timestamp of the events. Defaults to "time:timestamp.
        start_timestamp_key (str, optional): The name of the column containing the start timestamp of an event.
            Defaults to "start_timestamp". If the column is not present, it will be created using either lifecycle information
            if `lifecycle_key` is an existing column, or by assuming the start timestamp is equal to the end timestamp.
        lifecycle_key (str, optional): The column containing lifecycle information. Defaults to "lifecycle:transition".

    Returns:
        AnnotatedGraph: The extracted behavior graph.
    """
    trace = ensure_start_timestamp_column(trace, start_timestamp_key, lifecycle_key)

    V: set[Event] = {
        Event(activity=activity, index=idx)
        for idx, activity in enumerate(trace[activity_key])
    }
    E: set[tuple[Event, Event]] = set()

    L: list[BehaviorGraphNode] = []
    for idx, (activity, start_timestamp, timestamp) in enumerate(
        zip(trace[activity_key], trace[start_timestamp_key], trace[timestamp_key])
    ):
        is_atomic = start_timestamp == timestamp

        end_event_node = BehaviorGraphNode(
            timestamp=timestamp,
            event=Event(activity=activity, index=idx),
            is_atomic=is_atomic,
            is_start=False,
        )

        if is_atomic:
            L.append(end_event_node)
        else:  # Also add a node for the start event
            L += [
                end_event_node,
                BehaviorGraphNode(
                    timestamp=start_timestamp,
                    event=end_event_node.event,
                    is_atomic=is_atomic,
                    is_start=True,
                ),
            ]

    L.sort(key=lambda x: x.timestamp)

    for i, node1 in enumerate(L):
        if node1.is_atomic or not node1.is_start:
            for node2 in L[i + 1 :]:
                if node2.is_start and not node2.is_atomic:
                    E.add((node1.event, node2.event))
                    continue
                if node2.is_atomic:
                    E.add((node1.event, node2.event))
                    break
                if (not node2.is_atomic) and (not node2.is_start):
                    if (node1.event, node2.event) not in E:
                        continue
                    else:
                        break
    return AnnotatedGraph(V=V, E=E)


def get_event_timeframe(
    index_in_trace: int,
    trace: pd.DataFrame,
    timestamp_key: str = constants.DEFAULT_TIMESTAMP_KEY,
    start_timestamp_key: str = constants.DEFAULT_START_TIMESTAMP_KEY,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    evt = trace.iloc[index_in_trace]
    return evt[start_timestamp_key], evt[timestamp_key]


def extract_representation(
    log: pd.DataFrame,
    traceid_key: str = constants.DEFAULT_TRACEID_KEY,
    activity_key: str = constants.DEFAULT_NAME_KEY,
    timestamp_key: str = constants.DEFAULT_TIMESTAMP_KEY,
    start_timestamp_key: str = constants.DEFAULT_START_TIMESTAMP_KEY,
    lifecycle_key: str = constants.DEFAULT_LIFECYCLE_KEY,
) -> list[AnnotatedGraph]:
    # If necessary, convert the log to the internal format
    log = ensure_start_timestamp_column(log, start_timestamp_key, lifecycle_key)

    representations = []
    for _, trace in log.groupby(traceid_key):
        trace_representation = calculate_behavior_graph(
            trace, activity_key, timestamp_key, start_timestamp_key, lifecycle_key
        )
        for event in trace_representation.V:
            start, end = get_event_timeframe(
                event.index, trace, timestamp_key, start_timestamp_key
            )
            trace_representation.annotate_node(
                event,
                NodeAnnotation(
                    service_time=(end - start).total_seconds(),
                    activity=event.activity,
                ),
            )

            for event2 in [
                e
                for e in trace_representation.V
                if (event, e) in trace_representation.E
            ]:
                start2, _ = get_event_timeframe(
                    event2.index, trace, timestamp_key, start_timestamp_key
                )
                trace_representation.annotate_edge(
                    (event, event2),
                    EdgeAnnotation(waiting_time=(start2 - end).total_seconds()),
                )
        representations.append(trace_representation)
    return representations


def apply_binners(
    graph: AnnotatedGraph, node_binners: dict[str, Binner], edge_binner: Binner
) -> AnnotatedGraph:
    # Copy old Graph
    new_graph = AnnotatedGraph(
        graph.V, graph.E, graph.node_annotations, graph.edge_annotations
    )

    for node in new_graph.V:
        old_node_annotation = graph.get_node_annotation(node)
        if old_node_annotation is not None:
            new_value = node_binners[old_node_annotation.activity].transform_one(
                old_node_annotation.service_time
            )
            new_graph.node_annotations[node].service_time = new_value

    for edge in new_graph.E:
        old_edge_annotation = graph.get_edge_annotation(edge)
        if old_edge_annotation is not None:
            old_value = old_edge_annotation.waiting_time
            new_value = edge_binner.transform_one(old_value)
            new_graph.edge_annotations[edge].waiting_time = new_value

    return new_graph


def discretize_populations(
    pop1: list[AnnotatedGraph], pop2: list[AnnotatedGraph], num_bins: int = 10
) -> tuple[list[AnnotatedGraph], list[AnnotatedGraph]]:
    both = pop1 + pop2
    all_activities = {e.activity for p in both for e in p.V}
    all_edges = {(u.activity, v.activity) for p in both for (u, v) in p.E}

    activity_service_times = {
        activity: [
            annotation.service_time
            for graph in both
            for evt in graph.V
            if (annotation := graph.get_node_annotation(evt)) is not None
            if evt.activity == activity
        ]
        for activity in all_activities
    }

    edge_waiting_times = {
        edge: [
            ann.waiting_time
            for graph in both
            for e in graph.E
            if (ann := graph.get_edge_annotation(e)) is not None
            if e[0].activity == edge[0] and e[1].activity == edge[1]
        ]
        for edge in all_edges
    }

    service_time_binners = {
        activity: create_binner(times, num_bins)
        for activity, times in activity_service_times.items()
    }
    waiting_time_binner = create_binner(
        [time for times in edge_waiting_times.values() for time in times],
        num_bins,
    )

    transformed_pop1 = [
        apply_binners(p, service_time_binners, waiting_time_binner) for p in pop1
    ]
    transformed_pop2 = [
        apply_binners(p, service_time_binners, waiting_time_binner) for p in pop2
    ]

    return transformed_pop1, transformed_pop2


def old_chi_square(dist1: list[T], dist2: list[T]) -> float:
    """A helper function to compute Chi-Square Test for two populations by computing the contingency table and then using the `chi2_contingency` function from scipy

    Args:
        dist1 (list[T]): Distribution 1.
        dist2 (list[T]): Distribution 2.

    Returns:
        float: The computed p-value
    """
    pop1 = Counter(dist1)
    pop2 = Counter(dist2)

    # Perform Chi^2 Test
    keys = list(set(pop2).union(set(pop1)))

    def population_to_contingency_matrix(
        pop: Counter[T], keys: list[T]
    ) -> Numpy1DArray[np.int64]:
        matrix = np.zeros(len(keys), dtype=np.int64)
        # Set the count of each point in the matrix
        for point, count in pop.items():
            matrix[keys.index(point)] = count
        return matrix

    mat1 = population_to_contingency_matrix(pop1, keys)
    mat2 = population_to_contingency_matrix(pop2, keys)

    # Contingency Table
    contingency = np.array([mat1, mat2])

    # Apply Chi^2 Test on contingency matrix
    # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html
    # chi2, p, dof, expected <- these are the return values of chi2_contingency; We are only interested in the p-value
    _, p, _, _ = chi2_contingency(contingency)
    return p


def chi_square(dist1: list[T], dist2: list[T]) -> float:
    """A helper function to compute Chi-Square Test for two populations by computing the contingency table and then using the `chi2_contingency` function from scipy

    Args:
        dist1 (list[T]): Distribution 1.
        dist2 (list[T]): Distribution 2.

    Returns:
        float: The computed p-value
    """

    # Build manually to remove duplicates without requiring T hashable
    keys: list[T] = []
    for item in dist1 + dist2:
        if item not in keys:
            keys.append(item)

    pop1 = Counter([keys.index(item) for item in dist1])
    pop2 = Counter([keys.index(item) for item in dist2])

    # Perform Chi^2 Test
    def population_to_contingency_matrix(
        pop: Counter[int], keys: list[T]
    ) -> Numpy1DArray[np.int64]:
        matrix = np.zeros(len(keys), dtype=int)
        # Set the count of each point in the matrix
        for point, count in pop.items():
            matrix[point] = count
        return matrix

    mat1 = population_to_contingency_matrix(pop1, keys)
    mat2 = population_to_contingency_matrix(pop2, keys)

    print(mat1)
    print(mat2)

    # Contingency Table
    contingency = np.array([mat1, mat2])

    # Apply Chi^2 Test on contingency matrix
    # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html
    # chi2, p, dof, expected <- these are the return values of chi2_contingency; We are only interested in the p-value
    _, p, _, _ = chi2_contingency(contingency)
    return p


def new_chi_square(dist1: list[T], dist2: list[T]) -> float:
    keys: list[T] = []
    for item in dist1 + dist2:
        if item not in keys:
            keys.append(item)

    def population_to_observation_counts(
        pop: list[T], keys: list[T]
    ) -> Numpy1DArray[np.int32]:
        counts = np.zeros(len(keys), dtype=np.int32)
        for item in pop:
            counts[keys.index(item)] += 1
        return counts

    pop1 = population_to_observation_counts(dist1, keys)
    pop2 = population_to_observation_counts(dist2, keys)

    print(pop1)
    print(pop2)

    from scipy.stats import chisquare

    return chisquare(pop1, pop2).pvalue


def compare_pops(pop1: list[AnnotatedGraph], pop2: list[AnnotatedGraph]) -> float:
    return chi_square(pop1, pop2)


def compare_processes(
    log1: pd.DataFrame,
    log2: pd.DataFrame,
    num_bins: int = 5,
    traceid_key: str = constants.DEFAULT_TRACEID_KEY,
    activity_key: str = constants.DEFAULT_NAME_KEY,
    timestamp_key: str = constants.DEFAULT_TIMESTAMP_KEY,
    start_timestamp_key: str = constants.DEFAULT_START_TIMESTAMP_KEY,
    lifecycle_key: str = constants.DEFAULT_LIFECYCLE_KEY,
) -> float:
    params: dict[str, str] = {
        "traceid_key": traceid_key,
        "activity_key": activity_key,
        "timestamp_key": timestamp_key,
        "start_timestamp_key": start_timestamp_key,
        "lifecycle_key": lifecycle_key,
    }

    # Convert to internal format
    log1 = ensure_start_timestamp_column(log1, start_timestamp_key, lifecycle_key)
    log2 = ensure_start_timestamp_column(log2, start_timestamp_key, lifecycle_key)

    pop1 = extract_representation(log1, **params)
    pop2 = extract_representation(log2, **params)

    transformed_pop1, transformed_pop2 = discretize_populations(pop1, pop2, num_bins)

    return compare_pops(transformed_pop1, transformed_pop2)
