from collections import Counter
from typing import TypeVar

import numpy as np
import pandas as pd
import pydantic
from scipy.stats import chi2_contingency
from pcomp.utils import constants
from pcomp.utils.data import Binner, create_binner

T = TypeVar("T")
LIFECYCLES = ["start", "complete", "ate_abort"]


class Event(pydantic.BaseModel):
    # Make immutable --> Hashable
    model_config = pydantic.ConfigDict(frozen=True)

    instance_id: str
    activity: str

    def __repr__(self):
        return f"({self.activity}, {self.instance_id})"


class BehaviorGraphNode(pydantic.BaseModel, arbitrary_types_allowed=True):
    """Used internally in the computation of the behavior graph."""

    timestamp: pd.Timestamp
    event: Event
    is_atomic: bool
    is_start: bool


class NodeAnnotation(pydantic.BaseModel):
    service_time: float
    activity: str


class EdgeAnnotation(pydantic.BaseModel):
    waiting_time: float


class AnnotatedGraph(pydantic.BaseModel):
    V: set[Event]
    E: set[tuple[Event, Event]]

    node_annotations: dict[Event, NodeAnnotation]
    edge_annotations: dict[tuple[Event, Event], EdgeAnnotation]

    def __init__(self, V, E):
        super().__init__(V=V, E=E, node_annotations=dict(), edge_annotations=dict())

    def annotate_node(self, node: Event, annotation: NodeAnnotation):
        self.node_annotations[node] = annotation

    def annotate_edge(self, edge: tuple[Event, Event], annotation: EdgeAnnotation):
        self.edge_annotations[edge] = annotation

    def get_node_annotation(self, node: Event) -> NodeAnnotation | None:
        return self.node_annotations.get(node, None)

    def get_edge_annotation(self, edge: tuple[Event, Event]) -> EdgeAnnotation | None:
        return self.edge_annotations.get(edge, None)


def extract_representation(
    log: pd.DataFrame,
    traceid_key: str = constants.DEFAULT_TRACEID_KEY,
    activity_key: str = constants.DEFAULT_NAME_KEY,
    timestamp_key: str = constants.DEFAULT_TIMESTAMP_KEY,
    lifecycle_key: str = constants.DEFAULT_LIFECYCLE_KEY,
    instance_key: str = constants.DEFAULT_INSTANCE_KEY,
) -> list[AnnotatedGraph]:
    representations = []
    for _, trace in log.groupby(traceid_key):
        trace_representation = calculate_behavior_graph(
            trace, activity_key, timestamp_key, lifecycle_key, instance_key
        )
        for event in trace_representation.V:
            start, end = get_event_timeframe(
                event, trace, timestamp_key, lifecycle_key, instance_key
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
                    event2, trace, timestamp_key, lifecycle_key, instance_key
                )
                trace_representation.annotate_edge(
                    (event, event2),
                    EdgeAnnotation(waiting_time=(start2 - end).total_seconds()),
                )
        representations.append(trace_representation)
    return representations


def get_event_timeframe(
    event: Event,
    trace: pd.DataFrame,
    timestamp_key: str = constants.DEFAULT_TIMESTAMP_KEY,
    lifecycle_key: str = constants.DEFAULT_LIFECYCLE_KEY,
    instance_key: str = constants.DEFAULT_INSTANCE_KEY,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Get the start and end of the execution of an activity.

    Args:
        event (Event): The event whose instance id to extract the timeframe for
        trace (pd.DataFrame): The trace to extract the timeframe from

    Returns:
        tuple[pd.Timestamp, pd.Timestamp]: The start and end of the execution of the activity
    """
    filtered_trace = trace[
        (trace[lifecycle_key].isin(LIFECYCLES))
        & (trace[instance_key] == event.instance_id)
    ].sort_values(by=timestamp_key)
    return (
        filtered_trace.iloc[0][timestamp_key],
        filtered_trace.iloc[-1][timestamp_key],
    )


def compare_pops(pop1: list[AnnotatedGraph], pop2: list[AnnotatedGraph]) -> float:
    return chi_square(pop1, pop2)


def summarize_dicts(dicts: list[dict[str, T]]) -> dict[str, list[T]]:
    """Summarize a list of dicts.

    Args:
        dicts (list[dict[str, T]]): The dicts to summarize

    Returns:
        dict[str, list[T]]: The summarized dict
    """
    keys = {key for d in dicts for key in d.keys()}
    return {key: [d[key] for d in dicts if key in d] for key in keys}


def apply_binners(graph: AnnotatedGraph, node_binners: dict[str, Binner], edge_binner):
    for node in graph.V:
        node_annotation = graph.get_node_annotation(node)
        if node_annotation is not None:
            new_value = node_binners[node_annotation.activity].transform_one(
                node_annotation.service_time
            )
            graph.node_annotations[node].service_time = new_value

    for edge in graph.E:
        edge_annotation = graph.get_edge_annotation(edge)
        if edge_annotation is not None:
            old_value = edge_annotation.waiting_time
            new_value = edge_binner.transform_one(old_value)
            graph.edge_annotations[edge].waiting_time = new_value


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


def compare_processes(
    log1: pd.DataFrame,
    log2: pd.DataFrame,
    num_bins: int = 5,
    traceid_key: str = constants.DEFAULT_TRACEID_KEY,
    activity_key: str = constants.DEFAULT_NAME_KEY,
    timestamp_key: str = constants.DEFAULT_TIMESTAMP_KEY,
    lifecycle_key: str = constants.DEFAULT_LIFECYCLE_KEY,
    instance_key: str = constants.DEFAULT_INSTANCE_KEY,
) -> float:
    params: dict[str, str] = {
        "traceid_key": traceid_key,
        "activity_key": activity_key,
        "timestamp_key": timestamp_key,
        "lifecycle_key": lifecycle_key,
        "instance_key": instance_key,
    }
    pop1 = extract_representation(log1, **params)
    pop2 = extract_representation(log2, **params)

    transformed_pop1, transformed_pop2 = discretize_populations(pop1, pop2, num_bins)

    return compare_pops(transformed_pop1, transformed_pop2)


def calculate_behavior_graph(
    trace: pd.DataFrame,
    activity_key: str = constants.DEFAULT_NAME_KEY,
    timestamp_key: str = constants.DEFAULT_TIMESTAMP_KEY,
    lifecycle_key: str = constants.DEFAULT_LIFECYCLE_KEY,
    instance_key: str = constants.DEFAULT_INSTANCE_KEY,
) -> AnnotatedGraph:
    V: set[Event] = {
        Event(instance_id=e[instance_key], activity=e[activity_key])
        for (_, e) in trace.iterrows()
        if e[lifecycle_key] in ["complete", "ate_abort"]
    }
    E: set[tuple[Event, Event]] = set()

    non_duplicated_instances = trace[instance_key].drop_duplicates(keep=False).tolist()

    L: list[BehaviorGraphNode] = [
        BehaviorGraphNode(
            timestamp=timestamp,
            event=Event(
                instance_id=instance_id,
                activity=activity,
            ),
            is_atomic=instance_id in non_duplicated_instances,
            is_start=lifecycle == "start",
        )
        for (timestamp, instance_id, activity, lifecycle) in zip(
            trace[timestamp_key],
            trace[instance_key],
            trace[activity_key],
            trace[lifecycle_key],
        )
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


def chi_square(dist1: list[T], dist2: list[T]) -> float:
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
    ) -> np.ndarray[T]:
        matrix = np.zeros(len(keys))
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
