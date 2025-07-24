"""
An implementation of the PermutationTestComparator which only compares w.r.t. the control flow
"""

import pandas as pd

from pcomp.emd.comparators.permutation_test import Permutation_Test_Comparator
from pcomp.emd.distances.levenshtein import postnormalized_levenshtein_distance
from pcomp.emd.extraction import Trace, extract_traces


class ControlFlowPermutationComparator(Permutation_Test_Comparator[Trace]):
    """
    An implementation of the Permutation_Test_Comparator which compares the control flow
    """

    def extract_representations(
        self, log_1: pd.DataFrame, log_2: pd.DataFrame
    ) -> tuple[list[Trace], list[Trace]]:
        return (extract_traces(log_1), extract_traces(log_2))

    def cost_fn(self, item1: Trace, item2: Trace) -> float:
        return postnormalized_levenshtein_distance(item1, item2)

    def cleanup(self) -> None:
        pass
