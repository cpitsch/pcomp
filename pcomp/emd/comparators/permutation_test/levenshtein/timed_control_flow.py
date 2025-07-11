from typing import Any

import pandas as pd

from pcomp.binning.Binner import BinnerFactory, BinnerManager
from pcomp.binning.KMeans_Binner import KMeans_Binner
from pcomp.emd.comparators.permutation_test import Permutation_Test_Comparator
from pcomp.emd.core import EMDBackend
from pcomp.emd.distances.levenshtein import (
    custom_postnormalized_levenshtein_distance,
    post_normalized_weighted_levenshtein_distance,
)
from pcomp.emd.extraction import (
    BinnedServiceTimeTrace,
    extract_binned_service_time_traces,
)
from pcomp.utils import (
    add_duration_column_to_log,
    constants,
    ensure_start_timestamp_column,
)


class Timed_Levenshtein_PermutationComparator(
    Permutation_Test_Comparator[BinnedServiceTimeTrace]
):
    """
    An implementation of the Permutation_Test_Comparator for comparisons w.r.t the timed control flow, i.e., activities and service times
    """

    binner_manager: BinnerManager

    def __init__(
        self,
        log_1: pd.DataFrame,
        log_2: pd.DataFrame,
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
        """Create an instance.

        Args:
            log_1 (pd.DataFrame): The first event log in the comparison.
            log_2 (pd.DataFrame): The second event log in the comparison.
            distribution_size (int, optional): The number of permutations to compute the
                EMD for. Defaults to 10_000.
            verbose (bool, optional): If True, show progress bars. Defaults to True.
            cleanup_on_del (bool, optional): If True, call `cleanup` upon destruction,
                e.g., when the object goes out of scope. Defaults to True.
            emd_backend (EMDBackend, optional): The backend to use for EMD computation.
                Defaults to "wasserstein" (use the "wasserstein" module).
                    Alternatively, "ot" or "pot" will use the "Python Optimal Transport"
                    package.
            seed (int, optional): The seed to use for sampling in the permutation test
                phase.
            multiprocess_cores (int, optional): The number of cores to use for
                multiprocessing. Defaults to 0 (no multiprocessing used).
            weighted_time_cost (bool, optional): In the trace distance computation, divide
                the time cost by the maximal number of bins, ensuring an equal cost contribution
                of both dimensions (control flow and time). If False, the "normalized" distance
                can still exceed `1`. Setting to True is strongly recommended, and will be
                the default in future versions. Defaults to False.
            binner_factory (BinnerFactory | None, optional): How to create the binners for the
                time dimension. If None, KMeans++ binning will be used. Defaults to None.
            binner_args (dict[str, Any] | None, optional): The arguments to pass to the
                binner factory. If None, k=3 will be used (sets the number of clusters for
                KMeans++). Defaults to None.
        """
        super().__init__(
            ensure_start_timestamp_column(log_1),
            ensure_start_timestamp_column(log_2),
            distribution_size,
            verbose,
            cleanup_on_del,
            emd_backend,
            seed,
            multiprocess_cores,
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
        """Extract the service time traces from the event logs and bin their activity
        service times.

        Args:
            log_1 (pd.DataFrame): The first event log.
            log_2 (pd.DataFrame): The second event log.

        Returns:
            tuple[list[BinnedServiceTimeTrace], list[BinnedServiceTimeTrace]]: The extracted
                BinnedServiceTimeTraces from each event log.
        """
        log_1 = add_duration_column_to_log(log_1, duration_key="@pcomp:duration")
        log_2 = add_duration_column_to_log(log_2, duration_key="@pcomp:duration")
        activity_duration_pairs = list(
            pd.concat([log_1, log_2])[
                [constants.DEFAULT_NAME_KEY, "@pcomp:duration"]
            ].itertuples(index=False, name=None)
        )
        self.binner_manager = BinnerManager(
            activity_duration_pairs,
            self.binner_factory,
            seed=self.seed,
            show_training_progress_bar=self.verbose,
            **self.binner_args,
        )

        return (
            extract_binned_service_time_traces(log_1, self.binner_manager),
            extract_binned_service_time_traces(log_2, self.binner_manager),
        )

    def cost_fn(
        self, item1: BinnedServiceTimeTrace, item2: BinnedServiceTimeTrace
    ) -> float:
        """Compute the dissimilarity between two binned service time traces. If
        `self.weighted_time_cost` is `True`, the time costs are all weighted by the maximum
        possible time difference, `num_bins - 1`. As such, the maximum cost that can be
        incurred due to time differences in each event is 1, and the postnormalized distance
        is in [0,1].


        Args:
            item1 (BinnedServiceTimeTrace): The first binned service time trace.
            item2 (BinnedServiceTimeTrace): The second binned service time trace.

        Returns:
            float: The postnormalized weighted levenshtein distance between the two traces.
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
