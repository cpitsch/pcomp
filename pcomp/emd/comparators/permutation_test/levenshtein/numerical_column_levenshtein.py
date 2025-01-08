from typing import Any

import pandas as pd

from pcomp.binning import BinnerFactory, BinnerManager, KMeans_Binner
from pcomp.emd.comparators.permutation_test import Permutation_Test_Comparator
from pcomp.emd.core import EMDBackend
from pcomp.emd.distances.levenshtein import (
    custom_postnormalized_levenshtein_distance,
    post_normalized_weighted_levenshtein_distance,
)
from pcomp.emd.extraction import (
    BinnedServiceTimeTrace,
    extract_binned_trace_with_numerical_attribute,
)
from pcomp.utils.constants import DEFAULT_NAME_KEY


class NumericalColumn_Levenshtein_PermutationComparator(
    Permutation_Test_Comparator[BinnedServiceTimeTrace]
):
    binner_manager: BinnerManager
    binner_factory: BinnerFactory
    binner_args: dict[str, Any]
    column: str

    def __init__(
        self,
        log_1: pd.DataFrame,
        log_2: pd.DataFrame,
        column: str,
        distribution_size: int = 10_000,
        verbose: bool = True,
        cleanup_on_del: bool = True,
        emd_backend: EMDBackend = "wasserstein",
        seed: int | None = None,
        multiprocess_cores: int = 0,
        weighted_time_cost: bool = False,
        binner_factory: BinnerFactory | None = None,
        binner_args: dict[str, Any] | None = None,
    ):
        super().__init__(
            log_1,
            log_2,
            distribution_size,
            verbose,
            cleanup_on_del,
            emd_backend,
            seed,
            multiprocess_cores,
        )

        self.column = column
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
        """
        Extract the service time traces from the event logs and bin their activity
        service times.
        """
        activity_value_pairs = list(
            pd.concat([log_1, log_2])[[DEFAULT_NAME_KEY, self.column]].itertuples(
                index=False, name=None
            )
        )
        self.binner_manager = BinnerManager(
            activity_value_pairs,
            self.binner_factory,
            seed=self.seed,
            show_training_progress_bar=self.verbose,
            **self.binner_args,
        )

        return (
            extract_binned_trace_with_numerical_attribute(
                log_1, self.binner_manager, self.column
            ),
            extract_binned_trace_with_numerical_attribute(
                log_2, self.binner_manager, self.column
            ),
        )

    def cost_fn(
        self, item1: BinnedServiceTimeTrace, item2: BinnedServiceTimeTrace
    ) -> float:
        """
        If `weighted_time_cost` is True, the time costs are all weighted by the maximum
        possible time difference, `num_bins - 1`.
        As such, the maximum cost that can be incurred due to time differences in each
        event is 1.
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
            ) / (2 if self.binner_manager.num_bins > 0 else 1)
        else:
            return custom_postnormalized_levenshtein_distance(item1, item2)

    def cleanup(self) -> None:
        pass
