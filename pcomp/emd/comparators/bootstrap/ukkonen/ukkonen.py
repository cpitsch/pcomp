from functools import cache
from typing import Any

import pandas as pd

from pcomp.binning import BinnerFactory, BinnerManager, KMeans_Binner
from pcomp.emd.approximations.string_edit_distance import ukkonen_distance
from pcomp.emd.comparators.bootstrap import BootstrapComparator, BootstrappingStyle
from pcomp.emd.core import EMDBackend, compute_time_distance_component
from pcomp.emd.emd import (
    BinnedServiceTimeTrace,
    extract_service_time_traces,
    extract_traces_activity_service_times,
)


@cache
def _cached_ukkonen_distance(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    return ukkonen_distance(a, b)


@cache
def _cached_time_distance(
    a: BinnedServiceTimeTrace, b: BinnedServiceTimeTrace
) -> float:
    return compute_time_distance_component(a, b)


class Timed_Ukkonen_BootstrapComparator(BootstrapComparator[BinnedServiceTimeTrace]):
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

        # Default to KMenas_Binner with 3 bins
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
    ) -> tuple[list[BinnedServiceTimeTrace], list[BinnedServiceTimeTrace]]:
        self.binner_manager = BinnerManager(
            [
                evt
                for trace in (
                    extract_service_time_traces(log_1)
                    + extract_service_time_traces(log_2)
                )
                for evt in trace
            ],
            self.binner_factory,
            seed=self.seed,
            show_training_progress_bar=self.verbose,
            **self.binner_args,
        )

        binned_traces_1 = extract_traces_activity_service_times(
            log_1, self.binner_manager
        )
        binned_traces_2 = extract_traces_activity_service_times(
            log_2, self.binner_manager
        )

        return binned_traces_1, binned_traces_2

    def cost_fn(
        self, item1: BinnedServiceTimeTrace, item2: BinnedServiceTimeTrace
    ) -> float:
        control_flow_dist = _cached_ukkonen_distance(
            tuple(act for act, _ in item1), tuple(act for act, _ in item2)
        )
        time_dist = _cached_time_distance(item1, item2)

        return (control_flow_dist + time_dist) / max(
            len(item1), len(item2)
        )  # Post-normalization

    def cleanup(self) -> None:
        _cached_ukkonen_distance.cache_clear()
