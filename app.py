import pickle
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Generic, Literal, Protocol, TypeVar

import pandas as pd
import streamlit as st
from matplotlib.figure import Figure
from pm4py import read_xes  # type: ignore

from pcomp.binning import BinnerFactory, KMeans_Binner, OuterPercentileBinner
from pcomp.emd import Timed_Levenshtein_EMD_Comparator
from pcomp.emd.Comparators.double_bootstrap import (
    DoubleBootstrapStyle,
    LevenshteinDoubleBootstrapComparator,
)
from pcomp.emd.Comparators.kolmogorov_smirnov import (
    LevenshteinKSComparator,
    Self_Bootstrapping_Style,
)
from pcomp.emd.core import BootstrappingStyle
from pcomp.emd.emd import BinnedServiceTimeTrace
from pcomp.utils import enable_logging

enable_logging()

# Types and formatting helpers
ComparisonTechnique = Literal["emd_bootstrap", "ks_bootstrap", "double_bootstrap"]
funcs_to_name = {
    "emd_bootstrap": "Standard EMD Bootstrapping",
    "ks_bootstrap": "Kolmogorov-Smirnov Distribution Comparison",
    "double_bootstrap": "Double Bootstrap EMD",
}

BinnerSetting = Literal["kmeans_1", "kmeans_3", "outer_10"]
binner_setting_to_name = {
    "kmeans_1": "KMeans++ 1 Bin (No Time)",
    "kmeans_3": "KMeans++ 3 Bins",
    "outer_10": "Outer Percentile (10%)",
}


def binner_setting_to_args(
    binner_setting: BinnerSetting,
) -> tuple[BinnerFactory, dict[str, Any]]:
    binner, config = binner_setting.split("_")
    if binner == "kmeans":
        return (
            KMeans_Binner,
            {"k": int(config)},
        )
    elif binner == "outer":
        return OuterPercentileBinner, {}
    else:
        raise ValueError(f"Unknown binner setting: {binner_setting}")


# Default values
drifts: bool = False
weighted_time_cost: bool = False
binning_setting: BinnerSetting = "kmeans_1"
comparison_technique: ComparisonTechnique = "emd_bootstrap"

# EMD Bootstrap
bootstrapping_style: BootstrappingStyle = "replacement sublogs"
resample_percentage: float = 1.0

# EMD KS
self_bootstrapping_style: Self_Bootstrapping_Style = "split"

# Double Bootstrap
double_bootstrap_style: DoubleBootstrapStyle = "sample_smaller_log_size"


class ComparableAndPlottable(Protocol):
    def plot_result(self) -> Figure:
        pass

    def compare(self) -> float:
        pass


T = TypeVar("T", bound=ComparableAndPlottable)


@dataclass
class Instance(ABC, Generic[T]):
    drifts: bool
    weighted_time_cost: bool
    binner_setting: BinnerSetting

    @property
    def path(self) -> Path:
        return Path(
            "all_outputs",
            self.technique_name,
            "drift" if self.drifts else "no_drift",
            self.binner_setting,
            "weighted_time" if self.weighted_time_cost else "normal_time",
        )

    @property
    @abstractmethod
    def technique_name(self) -> str:
        raise NotImplementedError("Subclasses must implement `technique_name`")

    @abstractmethod
    def get_comparator(self, log_1: pd.DataFrame, log_2: pd.DataFrame) -> T:
        raise NotImplementedError("Subclasses must implement `get_comparator`")

    def run_and_save_results(self, log_1: pd.DataFrame, log_2: pd.DataFrame):
        comparator = self.get_comparator(log_1, log_2)
        _ = comparator.compare()

        # Save results
        pickle_path = self.path / "result.pkl"
        png_path = self.path / "result.png"
        self.path.mkdir(parents=True, exist_ok=True)
        with open(pickle_path, "wb") as f:
            pickle.dump(comparator, f)
        comparator.plot_result().savefig(png_path.as_posix(), bbox_inches="tight")


@dataclass
class StandardEmdInstance(Instance[Timed_Levenshtein_EMD_Comparator]):
    bootstrapping_style: BootstrappingStyle
    resample_percentage: float

    @property
    def path(self) -> Path:
        return Path(
            super().path,
            self.bootstrapping_style.replace(" ", "_"),
            f"resample_{self.resample_percentage}".replace(".", "-"),
        )

    @property
    def technique_name(self) -> str:
        return "emd_bootstrap"

    def get_comparator(
        self, log_1: pd.DataFrame, log_2: pd.DataFrame
    ) -> Timed_Levenshtein_EMD_Comparator:
        binner_factory, binner_args = binner_setting_to_args(self.binner_setting)
        return Timed_Levenshtein_EMD_Comparator(
            log_1,
            log_2,
            resample_size=self.resample_percentage,
            bootstrapping_style=self.bootstrapping_style,
            weighted_time_cost=self.weighted_time_cost,
            binner_factory=binner_factory,
            binner_args=binner_args,
        )


@dataclass
class KsEmdInstance(Instance[LevenshteinKSComparator]):
    self_bootstrapping_style: Self_Bootstrapping_Style

    @property
    def path(self) -> Path:
        return Path(super().path, self.self_bootstrapping_style.replace(" ", "_"))

    @property
    def technique_name(self) -> str:
        return "ks_bootstrap"

    def get_comparator(
        self, log_1: pd.DataFrame, log_2: pd.DataFrame
    ) -> LevenshteinKSComparator:
        binner_factory, binner_args = binner_setting_to_args(self.binner_setting)
        return LevenshteinKSComparator(
            log_1,
            log_2,
            self_emds_bootstrapping_style=self.self_bootstrapping_style,
            weighted_time_cost=self.weighted_time_cost,
            binner_factory=binner_factory,
            binner_args=binner_args,
        )


@dataclass
class DoubleBootstrapInstance(Instance[LevenshteinDoubleBootstrapComparator]):
    bootstrapping_style: DoubleBootstrapStyle

    @property
    def path(self) -> Path:
        return Path(super().path, self.bootstrapping_style.replace(" ", "_"))

    @property
    def technique_name(self) -> str:
        return "double_bootstrap"

    def get_comparator(
        self, log_1: pd.DataFrame, log_2: pd.DataFrame
    ) -> LevenshteinDoubleBootstrapComparator:
        binner_factory, binner_args = binner_setting_to_args(self.binner_setting)
        return LevenshteinDoubleBootstrapComparator(
            log_1,
            log_2,
            bootstrapping_style=self.bootstrapping_style,
            weighted_time_cost=self.weighted_time_cost,
            binner_factory=binner_factory,
            binner_args=binner_args,
        )


logs_base_path = Path("Testing Logs", "Run All")


# Drifts at [20000,4000,6000,8000, 10000, 12000]
## Assume that the preprocessed logs already exist
def get_logs_no_drift() -> tuple[pd.DataFrame, pd.DataFrame]:
    log_1 = read_xes(
        Path(logs_base_path, "no_drift_log_1.xes.gz").as_posix(), variant="rustxes"
    )
    log_2 = read_xes(
        Path(logs_base_path, "no_drift_log_2.xes.gz").as_posix(), variant="rustxes"
    )
    return log_1, log_2


def get_logs_with_drift() -> tuple[pd.DataFrame, pd.DataFrame]:
    log_1 = read_xes(
        Path(logs_base_path, "drift_log_1.xes.gz").as_posix(), variant="rustxes"
    )
    log_2 = read_xes(
        Path(logs_base_path, "drift_log_2.xes.gz").as_posix(), variant="rustxes"
    )
    return log_1, log_2


st.title("EMD Process Comparison Exploration")

with st.expander("🔧 Parameters", expanded=True):
    drifts = st.checkbox("Drift between logs")
    binning_setting = st.selectbox(  # type: ignore
        "Time Binning",
        options=["kmeans_1", "kmeans_3", "outer_10"],
        index=0,
        format_func=lambda x: binner_setting_to_name[x],
    )
    weighted_time_cost = st.checkbox("Weighted Time Cost")

    comparison_technique = st.selectbox(  # type: ignore
        "Comparison Technique",
        options=["emd_bootstrap", "ks_bootstrap", "double_bootstrap"],
        index=0,
        format_func=lambda x: funcs_to_name[x],
    )

    if comparison_technique == "emd_bootstrap":
        bootstrapping_style = st.selectbox(  # type: ignore
            "Bootstrapping Style",
            ["replacement sublogs", "resample split", "split sampling"],
            index=0,
            format_func=lambda x: x.replace("_", " ").title(),
        )

        resample_percentage = st.select_slider(  # type: ignore
            "Resample Percentage",
            [0.25, 0.5, 1.0],
            value=1.0,
            help="Percentage of the size of the reference event log to use as resample size",
        )

    elif comparison_technique == "ks_bootstrap":
        bootstrapping_styles_formatted = {
            "replacement": "Sample 2 Halves with Replacement",
            "split": "Split the Population into 2 Distinct Halves",
        }
        self_bootstrapping_style = st.selectbox(  # type: ignore
            "Bootstrapping Distribution Sampling Technique",
            ["split", "replacement"],
            index=0,
            format_func=lambda x: bootstrapping_styles_formatted[x],
        )

    elif comparison_technique == "double_bootstrap":
        double_bootstrap_styles_formatted = {
            "sample_smaller_log_size": "Sample with smaller Log Size",
            "splitted_resampling": "Resample from disjunct halves of Log 1",
        }
        double_bootstrap_style = st.selectbox(  # type: ignore
            "Bootstrapping Style",
            ["sample_smaller_log_size", "splitted_resampling"],
            index=0,
            format_func=lambda x: double_bootstrap_styles_formatted[x],
        )

instance: Instance
if comparison_technique == "emd_bootstrap":
    instance = StandardEmdInstance(
        drifts=drifts,
        weighted_time_cost=weighted_time_cost,
        binner_setting=binning_setting,
        bootstrapping_style=bootstrapping_style,
        resample_percentage=resample_percentage,
    )
elif comparison_technique == "ks_bootstrap":
    instance = KsEmdInstance(
        drifts=drifts,
        weighted_time_cost=weighted_time_cost,
        binner_setting=binning_setting,
        self_bootstrapping_style=self_bootstrapping_style,
    )
elif comparison_technique == "double_bootstrap":
    instance = DoubleBootstrapInstance(
        drifts=drifts,
        weighted_time_cost=weighted_time_cost,
        binner_setting=binning_setting,
        bootstrapping_style=double_bootstrap_style,
    )
else:
    raise ValueError(f"Unknown comparison technique: {comparison_technique}")


pickle_path = instance.path / "result.pkl"
png_path = instance.path / "result.png"


if not pickle_path.exists():
    none_found_warning = st.warning("No comparison results found. Run comparison?")
    if st.button("Go!"):
        warning = st.warning("Running comparison, please wait...")
        msg = st.toast("Running comparison, please wait... (~10min)", icon="⌛")

        logs = get_logs_with_drift() if drifts else get_logs_no_drift()

        instance.run_and_save_results(*logs)
        msg.toast("Comparison Complete!", icon="✅")

        # Clear the warnings
        none_found_warning.empty()
        warning.empty()

if pickle_path.exists():
    with open(pickle_path, "rb") as pickle_file:
        comparator = pickle.load(pickle_file)

    st.subheader(f"P-Value: {comparator.pval:.2f}")
    st.pyplot(comparator.plot_result())

    with st.expander("🔍 Timing Statistics"):

        def get_bin_class_counter_per_act(behavior: list[BinnedServiceTimeTrace]):
            events = [evt for trace in behavior for evt in trace]

            activities = set(act for act, _ in events)

            datapoints_per_activity = {
                act: [binned_dur for activity, binned_dur in events if activity == act]
                for act in activities
            }

            counters_per_activity = {
                act: Counter(binned_durs)
                for act, binned_durs in datapoints_per_activity.items()
            }

            return counters_per_activity

        counters_1 = get_bin_class_counter_per_act(comparator.behavior_1)
        counters_2 = get_bin_class_counter_per_act(comparator.behavior_2)

        all_keys = set(counters_1.keys()).union(counters_2.keys())

        # Create a dataframe
        df = pd.DataFrame(
            {
                "Activity": list(all_keys),
                "Class 0": [
                    (
                        counters_1.get(act, Counter()).get(0),
                        counters_2.get(act, Counter()).get(0),
                    )
                    for act in all_keys
                ],
                "Class 1": [
                    (
                        counters_1.get(act, Counter()).get(1),
                        counters_2.get(act, Counter()).get(1),
                    )
                    for act in all_keys
                ],
                "Class 2": [
                    (
                        counters_1.get(act, Counter()).get(2),
                        counters_2.get(act, Counter()).get(2),
                    )
                    for act in all_keys
                ],
                "Average Class in Log 1": [
                    mean(counters_1.get(act, Counter()).elements()) for act in all_keys
                ],
                "Average Class in Log 2": [
                    mean(counters_2.get(act, Counter()).elements()) for act in all_keys
                ],
            }
        )

        st.dataframe(df, use_container_width=True)
