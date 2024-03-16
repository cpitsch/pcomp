from typing import Any, Literal

import pandas as pd

from pcomp.binning import BinnerFactory, BinnerManager
from pcomp.binning.KMeans_Binner import KMeans_Binner
from pcomp.emd.approximations.comparing_stars import DiGraph, GraphNode
from pcomp.emd.approximations.comparing_stars.comparing_stars import (
    timed_star_graph_edit_distance,
)
from pcomp.emd.core import BootstrappingStyle, EMD_ProcessComparator, EMDBackend
from pcomp.emd.emd import (
    extract_service_time_traces,
    extract_traces_activity_service_times,
)


class Timed_ApproxTraceGED_EMD_Comparator(EMD_ProcessComparator[DiGraph]):
    """An implementation of the EMD_ProcessComparator for comparing event logs
    w.r.t. the timed-control-flow, converting BinnedServiceTimeTraces to graphs
    and applying the GED approximation from the paper "Comparing Stars: On
    Approximating Graph Edit Distance" by Zeng et al.
    """

    binner_manager: BinnerManager

    def __init__(
        self,
        log_1: pd.DataFrame,
        log_2: pd.DataFrame,
        bootstrapping_dist_size: int = 10000,
        resample_size: int | float | None = None,
        verbose: bool = True,
        cleanup_on_del: bool = True,
        bootstrapping_style: BootstrappingStyle = "replacement sublogs",
        emd_backend: EMDBackend = "wasserstein",
        seed: int | None = None,
        binner_factory: BinnerFactory | None = None,
        binner_args: dict[str, Any] | None = None,
    ):
        super().__init__(
            log_1,
            log_2,
            bootstrapping_dist_size,
            resample_size,
            verbose,
            cleanup_on_del,
            bootstrapping_style,
            emd_backend,
            seed,
        )

        # Default to KMeans_Binner with 3 bins
        self.binner_factory = binner_factory or KMeans_Binner
        self.binner_args = binner_args or (
            {
                "k": 3,
            }
            if self.binner_factory == KMeans_Binner
            else {}
        )

    def extract_representations(
        self, log_1: pd.DataFrame, log_2: pd.DataFrame
    ) -> tuple[list[DiGraph], list[DiGraph]]:
        traces_1 = extract_service_time_traces(log_1)

        self.binner_manager = BinnerManager(
            [evt for trace in traces_1 for evt in trace],
            self.binner_factory,
            seed=self.seed,
            **self.binner_args
        )

        binned_traces_1 = extract_traces_activity_service_times(
            log_1, self.binner_manager
        )
        binned_traces_2 = extract_traces_activity_service_times(
            log_2, self.binner_manager
        )

        # Create the Graphs
        graphs_1 = [
            DiGraph(
                nodes=tuple(GraphNode(label, duration) for label, duration in trace),
                edges=frozenset((i, i + 1) for i in range(len(trace) - 1)),
            )
            for trace in binned_traces_1
        ]
        graphs_2 = [
            DiGraph(
                nodes=tuple(GraphNode(label, duration) for label, duration in trace),
                edges=frozenset((i, i + 1) for i in range(len(trace) - 1)),
            )
            for trace in binned_traces_2
        ]

        return (graphs_1, graphs_2)

    def cost_fn(
        self,
        item1: DiGraph,
        item2: DiGraph,
        bound: Literal["lower", "upper"] = "lower",
    ) -> float:
        # Scale time differences by the largest possible time difference, num_bins - 1
        lower_bound, upper_bound = timed_star_graph_edit_distance(
            item1, item2, self.binner_manager.num_bins - 1
        )

        # For now, for testing, we only use the lower bound because `compare` doesn't pass the bound in.
        return lower_bound if bound == "lower" else upper_bound

    def cleanup(self) -> None:
        timed_star_graph_edit_distance.cache_clear()
