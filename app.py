import logging
import pickle
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Generic, Literal, Protocol, TypeVar, get_args

import pandas as pd
from matplotlib.figure import Figure
from pm4py import read_xes, write_xes  # type: ignore

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
from pcomp.utils import enable_logging, import_log, split_log_cases


def prepare_logs():
    """
    Create the logs and save them to disk if they don't exist. Makes sure that the
    logs are always the same and the results are not influenced by random chance.
    """
    logs_base_path = Path("Testing Logs", "Run All")

    def extract_and_save_logs(
        log_path: Path, path: Path, drift_point: int, no_drift_point: int
    ):
        """Extract the logs for the different comparison contexts: Two logs with a drift in between,
        and two logs with no drift in between.

        Then, save the logs to disk in the supplied directory

        Args:
            log_path (Path): The path to the log to extract and save.
            path (Path): The directory to save the logs to.
            drift_point (int): The point at which a drift occurs. Will take 1000 cases before and after
                this point for the drift logs.
            no_drift_point (int): A point around which no drift occurs. Will take 1000 cases before and after
                this point for the no drift logs.
        """
        # Read log, split around/between (depending on drift/no drift, respectively) drifts and save to disk
        logging.getLogger("@pcomp").info(f"Preparing {path.name} for comparison.")
        log = import_log(log_path.as_posix(), show_progress_bar=False)
        path.mkdir(parents=True, exist_ok=True)
        log["caseid_as_int"] = log["case:concept:name"].astype(int)

        drift_range_1 = (drift_point - 1_000, drift_point)
        drift_range_2 = (drift_point, drift_point + 1_000)

        logging.getLogger("@pcomp").info(
            f"{log_path.name}: Extracting drift logs for ranges {drift_range_1}, {drift_range_2}."
        )

        drift_log_1 = log[
            log["caseid_as_int"].between(*drift_range_1, inclusive="left")
        ]
        drift_log_2 = log[
            log["caseid_as_int"].between(*drift_range_2, inclusive="left")
        ]

        no_drift_range = (no_drift_point - 1_000, no_drift_point + 1_000)

        logging.getLogger("@pcomp").info(
            f"{log_path.name}: Extracting no drift logs from range {no_drift_range}."
        )

        no_drift_log_1, no_drift_log_2 = split_log_cases(
            log[log["caseid_as_int"].between(*no_drift_range, inclusive="left")],
            frac=0.5,
        )

        write_xes(drift_log_1, Path(path / "drift_log_1.xes.gz").as_posix())
        write_xes(drift_log_2, Path(path / "drift_log_2.xes.gz").as_posix())
        write_xes(no_drift_log_1, Path(path / "no_drift_log_1.xes.gz").as_posix())
        write_xes(no_drift_log_2, Path(path / "no_drift_log_2.xes.gz").as_posix())

    # Assume that if the base path exists, the logs also already exist
    ## log_cft.xes.gz
    log_cft_path = logs_base_path / "log_cft"
    if not log_cft_path.exists():
        # Drift logs: split into (7000-8000) and (8000-9000) (right side exclusive)
        # No drift logs: Randomly split range (0, 2000) in half
        extract_and_save_logs(
            Path("Testing Logs", "log_cft.xes.gz"), log_cft_path, 8000, 1000
        )

    log_long_term_dep_path = logs_base_path / "log_long_term_dep"
    if not log_long_term_dep_path.exists():
        # Drift at 2000
        # Drif Logs: split into (1000-2000) and (2000-3000) (right side exclusive)
        # No drift logs: Randomly split range (0, 2000) in half
        extract_and_save_logs(
            Path("Testing Logs", "log_long_term_dep.xes.gz"),
            log_long_term_dep_path,
            2000,
            1000,
        )

    log_soj_drift_path = logs_base_path / "log_soj_drift"
    if not log_soj_drift_path.exists():
        # Drift every 2000 cases (like log_cft)
        # Drift logs: split into (7000-8000) and (8000-9000) (right side exclusive)
        # No drift logs: Randomly split range (0, 2000) in half
        extract_and_save_logs(
            Path("Testing Logs", "log_soj_drift.xes.gz"), log_soj_drift_path, 8000, 1000
        )


## Types and formatting helpers ##
EventLog = Literal["log_cft", "log_long_term_dep", "log_soj_drift"]
event_logs: list[EventLog] = list(get_args(EventLog))
event_logs_formatter: dict[EventLog, str] = {
    "log_cft": "Bose CFT Drift",
    "log_long_term_dep": "Bose Long Term Dependence Time Drift",
    "log_soj_drift": "Bose Sojourn Time Drift",
}

ComparisonTechnique = Literal["emd_bootstrap", "ks_bootstrap", "double_bootstrap"]
comparison_techniques: list[ComparisonTechnique] = list(get_args(ComparisonTechnique))
funcs_to_name: dict[ComparisonTechnique, str] = {
    "emd_bootstrap": "Standard EMD Bootstrapping",
    "ks_bootstrap": "Kolmogorov-Smirnov Distribution Comparison",
    "double_bootstrap": "Double Bootstrap EMD",
}

BinnerSetting = Literal["kmeans_1", "kmeans_3", "outer_10"]
binner_settings: list[BinnerSetting] = list(get_args(BinnerSetting))
binner_setting_to_name: dict[BinnerSetting, str] = {
    "kmeans_1": "KMeans++ 1 Bin (No Time)",
    "kmeans_3": "KMeans++ 3 Bins",
    "outer_10": "Outer Percentile (10%)",
}

### Standard EMD Bootstrap
bootstrapping_styles: list[BootstrappingStyle] = list(get_args(BootstrappingStyle))
resample_percentages: list[float] = [0.25, 0.5, 1.0]

### EMD KS
ks_self_bootstrapping_styles: list[Self_Bootstrapping_Style] = list(
    get_args(Self_Bootstrapping_Style)
)
ks_bootstrapping_styles_formatter: dict[Self_Bootstrapping_Style, str] = {
    "replacement": "Sample 2 Halves with Replacement",
    "split": "Split the Population into 2 Distinct Halves",
}

### Double Bootstrap
double_bootstrap_styles: list[DoubleBootstrapStyle] = list(
    get_args(DoubleBootstrapStyle)
)
double_bootstrap_styles_formatter: dict[DoubleBootstrapStyle, str] = {
    "sample_smaller_log_size": "Sample with smaller Log Size",
    "splitted_resampling": "Resample from disjunct halves of Log 1",
}


def binner_setting_to_args(
    binner_setting: BinnerSetting,
) -> tuple[BinnerFactory, dict[str, Any]]:
    binner, config = binner_setting.split("_")
    if binner == "kmeans":
        return (
            KMeans_Binner,
            {
                "k": int(config),
            },
        )
    elif binner == "outer":
        return OuterPercentileBinner, {}
    else:
        raise ValueError(f"Unknown binner setting: {binner_setting}")


class ComparableAndPlottable(Protocol):
    def plot_result(self) -> Figure:
        pass

    def compare(self) -> float:
        pass


T = TypeVar("T", bound=ComparableAndPlottable)


@dataclass
class Instance(ABC, Generic[T]):
    log: EventLog
    drifts: bool
    weighted_time_cost: bool
    binner_setting: BinnerSetting

    @property
    def path(self) -> Path:
        return Path(
            "all_outputs",
            self.log,
            self.technique_name,
            "drift" if self.drifts else "no_drift",
            self.binner_setting,
            "weighted_time" if self.weighted_time_cost else "normal_time",
        )

    @property
    @abstractmethod
    def technique_name(self) -> str:
        raise NotImplementedError("Subclasses must implement `technique_name`")

    def get_logs(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        logs_base_path = Path("Testing Logs", "Run All", self.log)
        if self.drifts:
            return (
                import_log(
                    Path(logs_base_path, "drift_log_1.xes.gz").as_posix(),
                    show_progress_bar=False,
                ),
                import_log(
                    Path(logs_base_path, "drift_log_2.xes.gz").as_posix(),
                    show_progress_bar=False,
                ),
            )
        else:
            return (
                import_log(
                    Path(logs_base_path, "no_drift_log_1.xes.gz").as_posix(),
                    show_progress_bar=False,
                ),
                import_log(
                    Path(logs_base_path, "no_drift_log_2.xes.gz").as_posix(),
                    show_progress_bar=False,
                ),
            )

    @abstractmethod
    def get_comparator(self, verbose: bool = True) -> T:
        raise NotImplementedError("Subclasses must implement `get_comparator`")

    def run_and_save_results(self, verbose: bool = True):
        comparator = self.get_comparator(verbose=verbose)
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

    def get_comparator(self, verbose: bool = True) -> Timed_Levenshtein_EMD_Comparator:
        binner_factory, binner_args = binner_setting_to_args(self.binner_setting)
        log_1, log_2 = self.get_logs()
        return Timed_Levenshtein_EMD_Comparator(
            log_1,
            log_2,
            resample_size=self.resample_percentage,
            bootstrapping_style=self.bootstrapping_style,
            weighted_time_cost=self.weighted_time_cost,
            binner_factory=binner_factory,
            binner_args=binner_args,
            verbose=verbose,
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

    def get_comparator(self, verbose: bool = True) -> LevenshteinKSComparator:
        binner_factory, binner_args = binner_setting_to_args(self.binner_setting)
        log_1, log_2 = self.get_logs()
        return LevenshteinKSComparator(
            log_1,
            log_2,
            self_emds_bootstrapping_style=self.self_bootstrapping_style,
            weighted_time_cost=self.weighted_time_cost,
            binner_factory=binner_factory,
            binner_args=binner_args,
            verbose=verbose,
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
        self, verbose: bool = True
    ) -> LevenshteinDoubleBootstrapComparator:
        binner_factory, binner_args = binner_setting_to_args(self.binner_setting)
        log_1, log_2 = self.get_logs()
        return LevenshteinDoubleBootstrapComparator(
            log_1,
            log_2,
            bootstrapping_style=self.bootstrapping_style,
            weighted_time_cost=self.weighted_time_cost,
            binner_factory=binner_factory,
            binner_args=binner_args,
            verbose=verbose,
        )


logs_base_path = Path("Testing Logs", "Run All")


# Drifts at [20000,4000,6000,8000, 10000, 12000]
## Assume that the preprocessed logs already exist
def get_logs_no_drift() -> tuple[pd.DataFrame, pd.DataFrame]:
    log_1 = import_log(
        Path(logs_base_path, "no_drift_log_1.xes.gz").as_posix(),
        show_progress_bar=False,
    )
    log_2 = import_log(
        Path(logs_base_path, "no_drift_log_2.xes.gz").as_posix(),
        show_progress_bar=False,
    )
    return log_1, log_2


def get_logs_with_drift() -> tuple[pd.DataFrame, pd.DataFrame]:
    log_1 = read_xes(
        Path(logs_base_path, "drift_log_1.xes.gz").as_posix(), show_progress_bar=False
    )
    log_2 = read_xes(
        Path(logs_base_path, "drift_log_2.xes.gz").as_posix(), show_progress_bar=False
    )
    return log_1, log_2


def streamlit_main_loop() -> None:
    import streamlit as st

    log: EventLog
    drifts: bool
    weighted_time_cost: bool
    binning_setting: BinnerSetting
    comparison_technique: ComparisonTechnique

    # EMD Bootstrap
    bootstrapping_style: BootstrappingStyle
    resample_percentage: float

    # EMD KS
    self_bootstrapping_style: Self_Bootstrapping_Style

    # Double Bootstrap
    double_bootstrap_style: DoubleBootstrapStyle

    st.title("EMD Process Comparison Exploration")

    with st.expander("🔧 Parameters", expanded=True):
        log = st.selectbox(  # type: ignore
            "Event Log",
            options=event_logs,
            index=0,
            format_func=lambda x: event_logs_formatter[x],
        )

        drifts = st.checkbox("Drift between logs")
        binning_setting = st.selectbox(  # type: ignore
            "Time Binning",
            options=binner_settings,
            index=0,
            format_func=lambda x: binner_setting_to_name[x],
        )
        weighted_time_cost = st.checkbox("Weighted Time Cost")

        comparison_technique = st.selectbox(  # type: ignore
            "Comparison Technique",
            options=comparison_techniques,
            index=0,
            format_func=lambda x: funcs_to_name[x],
        )

        if comparison_technique == "emd_bootstrap":
            bootstrapping_style = st.selectbox(  # type: ignore
                "Bootstrapping Style",
                bootstrapping_styles,
                index=0,
                format_func=lambda x: x.replace("_", " ").title(),
            )

            resample_percentage = st.select_slider(  # type: ignore
                "Resample Percentage",
                resample_percentages,
                value=1.0,
                help="Percentage of the size of the reference event log to use as resample size",
            )

        elif comparison_technique == "ks_bootstrap":
            self_bootstrapping_style = st.selectbox(  # type: ignore
                "Bootstrapping Distribution Sampling Technique",
                ks_self_bootstrapping_styles,
                index=0,
                format_func=lambda x: ks_bootstrapping_styles_formatter.get(x, x),
            )

        elif comparison_technique == "double_bootstrap":
            double_bootstrap_style = st.selectbox(  # type: ignore
                "Bootstrapping Style",
                double_bootstrap_styles,
                index=0,
                format_func=lambda x: double_bootstrap_styles_formatter[x],
            )

    instance: Instance
    if comparison_technique == "emd_bootstrap":
        instance = StandardEmdInstance(
            log=log,
            drifts=drifts,
            weighted_time_cost=weighted_time_cost,
            binner_setting=binning_setting,
            bootstrapping_style=bootstrapping_style,
            resample_percentage=resample_percentage,
        )
    elif comparison_technique == "ks_bootstrap":
        instance = KsEmdInstance(
            log=log,
            drifts=drifts,
            weighted_time_cost=weighted_time_cost,
            binner_setting=binning_setting,
            self_bootstrapping_style=self_bootstrapping_style,
        )
    elif comparison_technique == "double_bootstrap":
        instance = DoubleBootstrapInstance(
            log=log,
            drifts=drifts,
            weighted_time_cost=weighted_time_cost,
            binner_setting=binning_setting,
            bootstrapping_style=double_bootstrap_style,
        )
    else:
        raise ValueError(f"Unknown comparison technique: {comparison_technique}")

    pickle_path = instance.path / "result.pkl"
    if not pickle_path.exists():
        none_found_warning = st.warning("No comparison results found. Run comparison?")
        if st.button("Go!"):
            warning = st.warning("Running comparison, please wait...")
            msg = st.toast("Running comparison, please wait... (~10min)", icon="⌛")

            instance.run_and_save_results()
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
                    act: [
                        binned_dur for activity, binned_dur in events if activity == act
                    ]
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
                        mean(counters_1.get(act, Counter()).elements())
                        for act in all_keys
                    ],
                    "Average Class in Log 2": [
                        mean(counters_2.get(act, Counter()).elements())
                        for act in all_keys
                    ],
                }
            )

            st.dataframe(df, use_container_width=True)


def get_standard_emd_instances() -> list[Instance]:
    return [
        StandardEmdInstance(
            log=log,
            drifts=drifts,
            binner_setting=binner_setting,
            weighted_time_cost=weighted_time_cost,
            bootstrapping_style=bootstrapping_style,
            resample_percentage=resample_percentage,
        )
        for log in event_logs
        for drifts in [True, False]
        for binner_setting in binner_settings
        for weighted_time_cost in [True, False]
        for bootstrapping_style in bootstrapping_styles
        for resample_percentage in resample_percentages
    ]


def get_ks_emd_instances() -> list[Instance]:
    return [
        KsEmdInstance(
            log=log,
            drifts=drifts,
            binner_setting=binner_setting,
            weighted_time_cost=weighted_time_cost,
            self_bootstrapping_style=self_bootstrapping_style,
        )
        for log in event_logs
        for drifts in [True, False]
        for binner_setting in binner_settings
        for weighted_time_cost in [True, False]
        for self_bootstrapping_style in ks_self_bootstrapping_styles
    ]


def get_double_bootstrap_instances() -> list[Instance]:
    return [
        DoubleBootstrapInstance(
            log=log,
            drifts=drifts,
            binner_setting=binner_setting,
            weighted_time_cost=weighted_time_cost,
            bootstrapping_style=bootstrapping_style,
        )
        for log in event_logs
        for drifts in [True, False]
        for binner_setting in binner_settings
        for weighted_time_cost in [True, False]
        for bootstrapping_style in double_bootstrap_styles
    ]


def get_all_instances() -> list[Instance]:
    standard_instances: list[Instance] = get_standard_emd_instances()
    ks_instances = get_ks_emd_instances()
    double_bootstrap_instances = get_double_bootstrap_instances()

    return standard_instances + ks_instances + double_bootstrap_instances


def run_instance(instance: Instance):
    instance.run_and_save_results(verbose=False)


def main() -> None:
    from mpire import WorkerPool, cpu_count

    SAVE_CORES = 0

    filtered_args = [
        instance
        for instance in get_all_instances()
        if not (instance.path / "result.pkl").exists()
    ]

    prepare_logs()

    print("Remaining instances:", len(filtered_args))

    with WorkerPool(cpu_count() - SAVE_CORES) as p:
        p.map(run_instance, filtered_args, progress_bar=True)


if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx():
            is_streamlit_run = True
        else:
            is_streamlit_run = False
    except ImportError:
        is_streamlit_run = False

    if is_streamlit_run:
        enable_logging()
        streamlit_main_loop()
    else:
        main()
