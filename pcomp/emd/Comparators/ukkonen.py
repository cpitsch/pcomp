from functools import cache

import pandas as pd

from pcomp.binning import Binner, IQR_Binner
from pcomp.emd.approximations.string_edit_distance import ukkonen_distance
from pcomp.emd.core import (
    BootstrappingStyle,
    EMD_ProcessComparator,
    EMDBackend,
    compute_time_distance_component,
)
from pcomp.emd.emd import (
    BinnedServiceTimeTrace,
    ServiceTimeTrace,
    extract_service_time_traces,
)


@cache
def _cached_ukkonen_distance(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    return ukkonen_distance(a, b)


@cache
def _cached_time_distance(
    a: BinnedServiceTimeTrace, b: BinnedServiceTimeTrace
) -> float:
    return compute_time_distance_component(a, b)


class Ukkonen_Distance_EMD_Comparator(EMD_ProcessComparator[BinnedServiceTimeTrace]):
    def __init__(
        self,
        log_1: pd.DataFrame,
        log_2: pd.DataFrame,
        bootstrapping_dist_size: int = 1_000,
        resample_size: int | None = None,
        verbose: bool = True,
        cleanup_on_del: bool = True,
        bootstrapping_style: BootstrappingStyle = "replacement sublogs",
        emd_backend: EMDBackend = "wasserstein",
    ):
        super().__init__(
            log_1,
            log_2,
            bootstrapping_dist_size,
            resample_size,
            verbose,
            cleanup_on_del,
            bootstrapping_style,
        )

    def extract_representations(
        self, log_1: pd.DataFrame, log_2: pd.DataFrame
    ) -> tuple[list[BinnedServiceTimeTrace], list[BinnedServiceTimeTrace]]:
        unbinned_traces_1: list[ServiceTimeTrace] = extract_service_time_traces(log_1)
        unbinned_traces_2: list[ServiceTimeTrace] = extract_service_time_traces(log_2)

        groupby_activity: dict[str, list[float]] = dict()
        for trace in unbinned_traces_1 + unbinned_traces_2:
            for act, dur in trace:
                if act not in groupby_activity:
                    groupby_activity[act] = []
                groupby_activity[act].append(dur)

        self.binners: dict[str, Binner] = {
            act: IQR_Binner(durations) for act, durations in groupby_activity.items()
        }

        binned_traces_1: list[BinnedServiceTimeTrace] = [
            tuple((act, self.binners[act].bin(dur)) for act, dur in trace)
            for trace in unbinned_traces_1
        ]
        binned_traces_2: list[BinnedServiceTimeTrace] = [
            tuple((act, self.binners[act].bin(dur)) for act, dur in trace)
            for trace in unbinned_traces_2
        ]

        return binned_traces_1, binned_traces_2

    def cost_fn(self, a: BinnedServiceTimeTrace, b: BinnedServiceTimeTrace) -> float:
        control_flow_dist = _cached_ukkonen_distance(
            tuple(act for act, _ in a), tuple(act for act, _ in b)
        )
        time_dist = _cached_time_distance(a, b)

        return (control_flow_dist + time_dist) / max(
            len(a), len(b)
        )  # Post-normalization

    def cleanup(self) -> None:
        _cached_ukkonen_distance.cache_clear()
