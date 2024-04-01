import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Literal, Protocol, TypeVar, get_args

import pandas as pd
from matplotlib.figure import Figure
from pm4py import write_xes  # type: ignore

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
from pcomp.emd.Comparators.permutation_test.levenshtein.levenshtein import (
    Timed_Levenshtein_PermutationComparator,
)
from pcomp.emd.core import BootstrappingStyle
from pcomp.utils import enable_logging, import_log, split_log_cases


def prepare_logs() -> None:
    logs_base_path = Path("Testing Logs")
    logs_save_base_path = Path("EvaluationLogs")
    logs_drift_points: dict[EventLog, list[int]] = {
        "log_cft": [4000, 6000, 8000, 10000, 12000],
        "log_long_term_dep": [2000],
        # Case arrival rate goes from 45 to: 60, 45, 30, 45, 25, 45 every 2000 cases
        # 45 can easily be handled by the resources, so 45->60 and 60->45 have no measurable difference for us
        "log_soj_drift": [6000, 8000, 10000, 12000],
        "log_classic_bose": [
            1201,
            2401,
            3601,
            4801,
        ],  # Drift point = first caseid in new log
        # Ceravolo Logs; Lowest caseid is 0
        "ceravolo_noise0_re": [500],
        "ceravolo_noise0_rp": [500],
        # Ostovar Logs; Lowest caseid is 1
        # Calling drift point the first caseid that is from the new log
        "ostovar_noise0_sre": [1001, 2001],
        "ostovar_noise0_cm": [1001, 2001],
        "ostovar_noise0_rp": [1001, 2001],
        "ostovar_noise0_cb": [1001, 2001],
    }

    def get_drift_log_ranges(
        log: EventLog, radius: int
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        drift_points = logs_drift_points[log]

        ret = [
            (
                (drift_point - radius, drift_point),
                (drift_point, drift_point + radius),
            )
            for drift_point in drift_points
        ]
        return ret

    def get_non_drift_log_ranges(log: EventLog, width: int) -> list[tuple[int, int]]:
        drift_points = logs_drift_points[log]

        ret = [
            (  # Log part *before* the first drift
                drift_points[0] - (2 * width),
                drift_points[0],
            )
        ] + [
            (  # Log ranges *after* each drift
                drift_point,
                drift_point + (2 * width),
            )
            for drift_point in drift_points
        ]
        return ret

    for logname in event_logs:
        log_path = logs_base_path / f"{logname}.xes.gz"
        save_path = logs_save_base_path / logname

        DRIFTS_RADIUS = 1000
        if logname.startswith("ceravolo"):
            DRIFTS_RADIUS = 500
        elif logname.startswith("ostovar"):
            DRIFTS_RADIUS = 999

        NO_DRIFTS_RADIUS = 1000
        if logname.startswith("ceravolo"):
            # Inter-trace-distance is 500, so can only do 250-length non-drift-logs
            NO_DRIFTS_RADIUS = 250
        elif logname.startswith("ostovar"):
            # Inter-trace-distance is 1000, so can only do 500-length non-drift-logs
            NO_DRIFTS_RADIUS = 499
        elif logname.startswith("log_classic_bose"):
            # Inter-trace-distance is 1200, so can only do 600-length non-drift-logs
            NO_DRIFTS_RADIUS = 600

        if not ((save_path / "drift").exists() and (save_path / "no_drift").exists()):
            logger = logging.getLogger("@pcomp")
            logger.info(f"Preparing {save_path.name} for comparison.")
            save_path.mkdir(parents=True, exist_ok=True)
            log = import_log(log_path.as_posix(), show_progress_bar=False)
            log["caseid_as_int"] = log["case:concept:name"].astype(int)

            max_caseid = log["caseid_as_int"].max()
            min_caseid = log["caseid_as_int"].min()

            if not (save_path / "drift").exists():
                # Drift logs are saved in a path such as:
                # EvaluationLogs/log_cft/drift/3000_5000/log_1.xes.gz
                (save_path / "drift").mkdir(parents=True, exist_ok=True)

                for drift_range_1, drift_range_2 in get_drift_log_ranges(
                    logname, DRIFTS_RADIUS
                ):
                    # Skip drift ranges that would go over the edge of the log
                    if (
                        drift_range_1[0] < min_caseid
                        or drift_range_2[1] > max_caseid + 1
                    ):
                        logger.warning(
                            f"Skipping creating drift log pair for {logname} range {(drift_range_1, drift_range_2)}; Drift range out of bounds",
                        )
                        continue

                    drift_log_1 = log[
                        log["caseid_as_int"].between(*drift_range_1, inclusive="left")
                    ]
                    drift_log_2 = log[
                        log["caseid_as_int"].between(*drift_range_2, inclusive="left")
                    ]

                    this_pair_base_path = (
                        save_path / "drift" / f"{drift_range_1[0]}_{drift_range_2[1]}"
                    )

                    this_pair_base_path.mkdir(parents=True, exist_ok=True)

                    logger.info(
                        f"Creating drift log pair {this_pair_base_path.as_posix()}. Log lengths: {(drift_log_1['case:concept:name'].nunique(),drift_log_2['case:concept:name'].nunique())}"
                    )

                    write_xes(
                        drift_log_1,
                        Path(this_pair_base_path, "log_1.xes.gz").as_posix(),
                    )
                    write_xes(
                        drift_log_2,
                        Path(this_pair_base_path, "log_2.xes.gz").as_posix(),
                    )

            if not (save_path / "no_drift").exists():
                # Non-drift logs are saved in a path such as:
                # EvaluationLogs/log_cft/no_drift/3000_5000/log_1.xes.gz
                (save_path / "no_drift").mkdir(parents=True, exist_ok=True)

                # Skip drift ranges that would go over the edge of the log
                for range_start, range_end in get_non_drift_log_ranges(
                    logname, NO_DRIFTS_RADIUS
                ):
                    if range_start < min_caseid or range_end > max_caseid + 1:
                        logger.warning(
                            f"Skipping creating non-drift log pair for {logname} range {(range_start, range_end)}; Drift range out of bounds",
                        )
                        continue

                    no_drift_log_1, no_drift_log_2 = split_log_cases(
                        log[
                            log["caseid_as_int"].between(
                                range_start, range_end, inclusive="left"
                            )
                        ],
                        frac=0.5,
                        seed=1337,
                    )

                    this_pair_base_path = (
                        save_path / "no_drift" / f"{range_start}_{range_end}"
                    )

                    this_pair_base_path.mkdir(parents=True, exist_ok=True)

                    logger.info(
                        f"Creating non-drift log pair {this_pair_base_path.as_posix()}. Log lengths: {(no_drift_log_1['case:concept:name'].nunique(),no_drift_log_2['case:concept:name'].nunique())}"
                    )

                    write_xes(
                        no_drift_log_1,
                        Path(
                            this_pair_base_path,
                            "log_1.xes.gz",
                        ).as_posix(),
                    )
                    write_xes(
                        no_drift_log_2,
                        Path(
                            this_pair_base_path,
                            "log_2.xes.gz",
                        ).as_posix(),
                    )


## Types and formatting helpers ##
EventLog = Literal[
    "log_cft",
    "log_long_term_dep",
    "log_soj_drift",
    "log_classic_bose",
    # Ceravolo Logs
    "ceravolo_noise0_re",
    "ceravolo_noise0_rp",
    # Ostovar Logs
    "ostovar_noise0_sre",
    "ostovar_noise0_cm",
    "ostovar_noise0_rp",
    "ostovar_noise0_cb",
]
event_logs: list[EventLog] = list(get_args(EventLog))
event_logs_formatter: dict[EventLog, str] = {
    "log_cft": "Bose CFT Drift",
    "log_long_term_dep": "Bose Long Term Dependence Time Drift",
    "log_soj_drift": "Bose Sojourn Time Drift",
    "log_classic_bose": "Classic Bose Log (Control-Flow Drift)",
    "ceravolo_noise0_re": "Ceravolo RE (Activity Removal)",
    "ceravolo_noise0_rp": "Ceravolo RP (Substitute)",
    "ostovar_noise0_sre": "Ostovar SRE (Serial Removal)",
    "ostovar_noise0_cm": "Ostovar CM (Conditional Move)",
    "ostovar_noise0_rp": "Ostovar RP (Substitute)",
    "ostovar_noise0_cb": "Ostovar CB (Skip)",
}


@dataclass
class EventLogSetting:
    log: EventLog
    drifts: bool
    log_identifier: str  # "log range", e.g., "3000_5000"

    @property
    def as_path(self) -> Path:
        return Path(
            "EvaluationLogs",
            self.log,
            "drift" if self.drifts else "no_drift",
            self.log_identifier,
        )

    def get_log_paths(self) -> tuple[Path, Path]:
        base_path = self.as_path
        return base_path / "log_1.xes.gz", base_path / "log_2.xes.gz"


ComparisonTechnique = Literal[
    "emd_bootstrap", "ks_bootstrap", "double_bootstrap", "permutation_test"
]
comparison_techniques: list[ComparisonTechnique] = list(get_args(ComparisonTechnique))
funcs_to_name: dict[ComparisonTechnique, str] = {
    "emd_bootstrap": "Standard EMD Bootstrapping",
    "ks_bootstrap": "Kolmogorov-Smirnov Distribution Comparison",
    "double_bootstrap": "Double Bootstrap EMD",
    "permutation_test": "Permutation Test",
}

BinnerSetting = Literal[
    # "kmeans_1",
    # "kmeans_3",
    "outer_10",
]
binner_settings: list[BinnerSetting] = list(get_args(BinnerSetting))
binner_setting_to_name: dict[BinnerSetting, str] = {
    # "kmeans_1": "KMeans++ 1 Bin (No Time)",
    # "kmeans_3": "KMeans++ 3 Bins",
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
    log_setting: EventLogSetting
    weighted_time_cost: bool
    binner_setting: BinnerSetting

    @property
    def path(self) -> Path:
        return Path(
            "EvaluationOutputs",
            self.log,
            self.technique_name,
            "drift" if self.drifts else "no_drift",
            self.binner_setting,
            "weighted_time" if self.weighted_time_cost else "normal_time",
        )

    @property
    def log(self) -> EventLog:
        return self.log_setting.log

    @property
    def drifts(self) -> bool:
        return self.log_setting.drifts

    @property
    def pickle_path(self) -> Path:
        """The path to the saved pickle file."""
        return self.path / f"result_{self.log_setting.log_identifier}.pkl"

    @property
    @abstractmethod
    def technique_name(self) -> str:
        raise NotImplementedError("Subclasses must implement `technique_name`")

    def get_logs(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        log_1_path, log_2_path = self.log_setting.get_log_paths()
        return (
            import_log(
                log_1_path.as_posix(),
                show_progress_bar=False,
            ),
            import_log(
                log_2_path.as_posix(),
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
        self.path.mkdir(parents=True, exist_ok=True)
        with open(self.pickle_path, "wb") as f:
            pickle.dump(comparator, f)


@dataclass
class StandardEmdInstance(Instance[Timed_Levenshtein_EMD_Comparator]):
    bootstrapping_style: BootstrappingStyle
    resample_percentage: float | None  # Not needed for "split sampling"

    @property
    def path(self) -> Path:
        path = Path(
            super().path,
            self.bootstrapping_style.replace(" ", "_"),
        )
        if self.bootstrapping_style == "split sampling":
            return path
        else:
            return Path(
                path,
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
            seed=1337,
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
            seed=1337,
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
            seed=1337,
            verbose=verbose,
        )


@dataclass
class PermutationTestInstance(Instance[Timed_Levenshtein_PermutationComparator]):
    @property
    def path(self) -> Path:
        return super().path

    @property
    def technique_name(self) -> str:
        return "permutation_test"

    def get_comparator(
        self, verbose: bool = True
    ) -> Timed_Levenshtein_PermutationComparator:
        binner_factory, binner_args = binner_setting_to_args(self.binner_setting)
        log_1, log_2 = self.get_logs()
        return Timed_Levenshtein_PermutationComparator(
            log_1,
            log_2,
            weighted_time_cost=self.weighted_time_cost,
            binner_factory=binner_factory,
            binner_args=binner_args,
            seed=1337,
            verbose=verbose,
        )


logs_base_path = Path("Testing Logs", "Run All")


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

    instances: list[Instance]
    if comparison_technique == "emd_bootstrap":
        instances = [
            StandardEmdInstance(
                log_setting=EventLogSetting(
                    log,
                    drifts,
                    log_range.name,
                ),
                weighted_time_cost=weighted_time_cost,
                binner_setting=binning_setting,
                bootstrapping_style=bootstrapping_style,
                resample_percentage=resample_percentage,
            )
            for log_range in Path(
                "EvaluationLogs", log, "drift" if drifts else "no_drift"
            ).iterdir()
        ]
    elif comparison_technique == "ks_bootstrap":
        instances = [
            KsEmdInstance(
                log_setting=EventLogSetting(
                    log,
                    drifts,
                    log_range.name,
                ),
                weighted_time_cost=weighted_time_cost,
                binner_setting=binning_setting,
                self_bootstrapping_style=self_bootstrapping_style,
            )
            for log_range in Path(
                "EvaluationLogs", log, "drift" if drifts else "no_drift"
            ).iterdir()
        ]
    elif comparison_technique == "double_bootstrap":
        instances = [
            DoubleBootstrapInstance(
                log_setting=EventLogSetting(
                    log,
                    drifts,
                    log_range.name,
                ),
                weighted_time_cost=weighted_time_cost,
                binner_setting=binning_setting,
                bootstrapping_style=double_bootstrap_style,
            )
            for log_range in Path(
                "EvaluationLogs", log, "drift" if drifts else "no_drift"
            ).iterdir()
        ]
    elif comparison_technique == "permutation_test":
        instances = [
            PermutationTestInstance(
                log_setting=EventLogSetting(log, drifts, log_range.name),
                weighted_time_cost=weighted_time_cost,
                binner_setting=binning_setting,
            )
            for log_range in Path(
                "EvaluationLogs", log, "drift" if drifts else "no_drift"
            ).iterdir()
        ]
    else:
        raise ValueError(f"Unknown comparison technique: {comparison_technique}")

    instances = sorted(
        instances, key=lambda x: int(x.log_setting.log_identifier.split("_")[0])
    )
    for instance in instances:
        with st.expander(
            "Log Range: " + instance.log_setting.log_identifier.replace("_", ", "),
            expanded=True,
        ):
            pickle_path = instance.pickle_path
            if not pickle_path.exists():
                none_found_warning = st.warning(
                    "No comparison results found. Run comparison?"
                )
                if st.button(
                    "Go!", key=f"go_key_{instance.log_setting.log_identifier}"
                ):
                    warning = st.warning("Running comparison, please wait...")
                    msg = st.toast(
                        "Running comparison, please wait... (~10min)", icon="⌛"
                    )

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


def get_standard_emd_instances() -> list[Instance]:
    return [
        StandardEmdInstance(
            log_setting=EventLogSetting(log, drifts, log_ident.name),
            binner_setting=binner_setting,
            weighted_time_cost=weighted_time_cost,
            bootstrapping_style=bootstrapping_style,
            resample_percentage=resample_percentage,
        )
        for log in event_logs
        for drifts in [True, False]
        for log_ident in Path(
            "EvaluationLogs", log, "drift" if drifts else "no_drift"
        ).iterdir()
        for binner_setting in binner_settings
        for weighted_time_cost in [True, False]
        for bootstrapping_style in bootstrapping_styles
        for resample_percentage in (
            # No resample percentage needed for "split sampling"
            resample_percentages
            if bootstrapping_style != "split sampling"
            else [None]  # type: ignore
        )
    ]


def get_ks_emd_instances() -> list[Instance]:
    return [
        KsEmdInstance(
            log_setting=EventLogSetting(log, drifts, log_ident.name),
            binner_setting=binner_setting,
            weighted_time_cost=weighted_time_cost,
            self_bootstrapping_style=self_bootstrapping_style,
        )
        for log in event_logs
        for drifts in [True, False]
        for log_ident in Path(
            "EvaluationLogs", log, "drift" if drifts else "no_drift"
        ).iterdir()
        for binner_setting in binner_settings
        for weighted_time_cost in [True, False]
        for self_bootstrapping_style in ks_self_bootstrapping_styles
    ]


def get_double_bootstrap_instances() -> list[Instance]:
    return [
        DoubleBootstrapInstance(
            log_setting=EventLogSetting(log, drifts, log_ident.name),
            binner_setting=binner_setting,
            weighted_time_cost=weighted_time_cost,
            bootstrapping_style=bootstrapping_style,
        )
        for log in event_logs
        for drifts in [True, False]
        for log_ident in Path(
            "EvaluationLogs", log, "drift" if drifts else "no_drift"
        ).iterdir()
        for binner_setting in binner_settings
        for weighted_time_cost in [True, False]
        for bootstrapping_style in double_bootstrap_styles
    ]


def get_permutation_test_instances() -> list[Instance]:
    return [
        PermutationTestInstance(
            log_setting=EventLogSetting(log, drifts, log_ident.name),
            binner_setting=binner_setting,
            weighted_time_cost=weighted_time_cost,
        )
        for log in event_logs
        for drifts in [True, False]
        for log_ident in Path(
            "EvaluationLogs", log, "drift" if drifts else "no_drift"
        ).iterdir()
        for binner_setting in binner_settings
        for weighted_time_cost in [True, False]
    ]


def get_all_instances() -> list[Instance]:
    standard_instances: list[Instance] = get_standard_emd_instances()
    ks_instances = get_ks_emd_instances()
    double_bootstrap_instances = get_double_bootstrap_instances()
    permutation_test_instances = get_permutation_test_instances()

    return (
        standard_instances
        + ks_instances
        + double_bootstrap_instances
        + permutation_test_instances
    )


def run_instance(instance: Instance):
    instance.run_and_save_results(verbose=False)


def main() -> None:
    from argparse import ArgumentParser

    from mpire import WorkerPool, cpu_count

    parser = ArgumentParser()
    parser.add_argument(
        "--prepare", action="store_true", help="Only prepare the logs and exit."
    )
    args = parser.parse_args()

    prepare_logs()

    if not args.prepare:
        SAVE_CORES = 0

        all_instances = get_all_instances()
        filtered_args = [
            instance for instance in all_instances if not instance.pickle_path.exists()
        ]

        print("Remaining instances:", len(filtered_args), "of", len(all_instances))

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
        enable_logging(logging.WARNING)
        main()
