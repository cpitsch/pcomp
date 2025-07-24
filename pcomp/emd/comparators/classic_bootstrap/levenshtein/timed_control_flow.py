from typing import Any

import pandas as pd

from pcomp.binning import BinnerFactory, BinnerManager, KMeans_Binner
from pcomp.emd.comparators.classic_bootstrap import ClassicBootstrap_Comparator
from pcomp.emd.core import EMDBackend
from pcomp.emd.distances.levenshtein import (
    custom_postnormalized_levenshtein_distance,
    post_normalized_weighted_levenshtein_distance,
)
from pcomp.emd.extraction import (
    BinnedServiceTimeTrace,
    extract_binned_service_time_traces,
    extract_service_time_traces,
)
from pcomp.utils import ensure_start_timestamp_column


class Timed_Levenshtein_ClassicBootstrapComparator(
    ClassicBootstrap_Comparator[BinnedServiceTimeTrace]
):
    """
    An implementation of the ClassicBootstrap_Comparator for comparing event logs
    w.r.t. the timed-control-flow using a weighted post-normalized levenshtein distance
    as the cost function.
    """

    binner_manager: BinnerManager

    def __init__(
        self,
        log_1: pd.DataFrame,
        log_2: pd.DataFrame,
        bootstrapping_dist_size: int = 10000,
        verbose: bool = True,
        cleanup_on_del: bool = True,
        emd_backend: EMDBackend = "wasserstein",
        seed: int | None = None,
        weighted_time_cost: bool = False,
        binner_factory: BinnerFactory | None = None,
        binner_args: dict[str, Any] | None = None,
    ):
        """Create an instance. The classic bootstrap comparator performs a "classic" two-sample
        bootstrap test. This is done by pooling both event logs together and then computing the
        EMD between samples (with replacement) and the pooled observations.

        Args:
            log_1 (pd.DataFrame): The first event log in the comparison.
            log_2 (pd.DataFrame): The second event log in the comparison.
            bootstrapping_dist_size (int, optional): The number of samples to compute
                the Self-EMD for. Defaults to 10_000.
            verbose (bool, optional): If True, show progress bars. Defaults to True.
            cleanup_on_del (bool, optional): If True, call `cleanup` upon destruction,
                e.g., when the object goes out of scope. Defaults to True.
            emd_backend (EMDBackend, optional): The backend to use for EMD computation.
                Defaults to "wasserstein" (use the "wasserstein" module). Alternatively,
                "ot" or "pot" will use the "Python Optimal Transport" package.
            seed (int, optional): The seed to use for sampling in the bootstrapping
                phase.
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
            bootstrapping_dist_size,
            verbose,
            cleanup_on_del,
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
        """
        Extract the service time traces from the event logs and bin their activity
        service times.
        """
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
            extract_binned_service_time_traces(log_1, self.binner_manager),
            extract_binned_service_time_traces(log_2, self.binner_manager),
        )

    def cost_fn(
        self, item1: BinnedServiceTimeTrace, item2: BinnedServiceTimeTrace
    ) -> float:
        """
        If `weighted_time_cost` is True, the time costs are all weighted by the maximum
        possible time difference, `num_bins - 1`. As such, the maximum cost that can be
        incurred due to time differences in each event is 1.
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
