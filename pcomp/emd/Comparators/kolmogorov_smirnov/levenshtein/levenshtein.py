from typing import Any

import pandas as pd

from pcomp.binning import BinnerFactory, BinnerManager, KMeans_Binner
from pcomp.emd.Comparators.kolmogorov_smirnov import (
    EMD_KS_ProcessComparator,
    Self_Bootstrapping_Style,
)
from pcomp.emd.core import EMDBackend
from pcomp.emd.emd import (
    BinnedServiceTimeTrace,
    custom_postnormalized_levenshtein_distance,
    extract_service_time_traces,
    extract_traces_activity_service_times,
    post_normalized_weighted_levenshtein_distance,
)


class LevenshteinKSComparator(EMD_KS_ProcessComparator[BinnedServiceTimeTrace]):
    """
    A class to compare two processes by comparing distributions of calculated EMDs. For more information, see the documentation of the abstract class `EMD_KS_ProcessComparator`.
    Represents the processes through the extracted sequences of (activity, duration) pairs for each case.
    Uses the Levenshtein distance as a cost function between items.
    """

    binner_manager: BinnerManager

    def __init__(
        self,
        log_1: pd.DataFrame,
        log_2: pd.DataFrame,
        bootstrapping_dist_size: int = 10000,
        verbose: bool = True,
        cleanup_on_del: bool = True,
        self_emds_bootstrapping_style: Self_Bootstrapping_Style = "replacement",
        emd_backend: EMDBackend = "wasserstein",
        seed: int | None = None,
        weighted_time_cost: bool = False,
        binner_factory: BinnerFactory | None = None,
        binner_args: dict[str, Any] | None = None,
    ):
        super().__init__(
            log_1,
            log_2,
            bootstrapping_dist_size,
            verbose,
            cleanup_on_del,
            self_emds_bootstrapping_style,
            emd_backend,
            seed,
        )
        self.weighted_time_cost = weighted_time_cost

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

        return (
            extract_traces_activity_service_times(log_1, self.binner_manager),
            extract_traces_activity_service_times(log_2, self.binner_manager),
        )

    def cost_fn(
        self, item1: BinnedServiceTimeTrace, item2: BinnedServiceTimeTrace
    ) -> float:
        """
        If `weighted_time_cost` is True, the time costs are all weighted by the maximum possible time difference, `num_bins - 1`.
        As such, the maximum cost that can be incurred due to time differences in each event is 1.
        """
        if self.weighted_time_cost:
            return post_normalized_weighted_levenshtein_distance(
                item1,
                item2,
                rename_cost=lambda *_: 1,
                insertion_deletion_cost=lambda _: 1,
                cost_time_match_rename=lambda x, y: abs(x - y)
                / max(self.binner_manager.num_bins - 1, 1),
                cost_time_insert_delete=lambda x: x
                / max(self.binner_manager.num_bins - 1, 1),
            )
        else:
            return custom_postnormalized_levenshtein_distance(item1, item2)

    def cleanup(self) -> None:
        pass
