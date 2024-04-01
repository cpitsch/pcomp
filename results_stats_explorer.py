from pathlib import Path

import pandas as pd
import streamlit as st

from app import (
    binner_setting_to_name,
    binner_settings,
    bootstrapping_styles,
    comparison_techniques,
    double_bootstrap_styles,
    double_bootstrap_styles_formatter,
    event_logs,
    event_logs_formatter,
    funcs_to_name,
    ks_bootstrapping_styles_formatter,
    ks_self_bootstrapping_styles,
    resample_percentages,
)

st.title("Results Explorer")


@st.cache_data
def load_df() -> pd.DataFrame | None:
    """Load the summary csv if it exists. Else, return None

    Returns:
        pd.DataFrame | None: The imported dataframe.
    """
    path = Path("EvaluationOutputs", "summary.csv")
    if path.exists():
        return pd.read_csv(path)
    else:
        return None


df = load_df()


def write_classification_classes_summary(df):
    counts = {k: v for k, v in df["classification_class"].value_counts().items()}

    print(counts)
    accuracy = (counts.get("TP", 0) + counts.get("TN", 0)) / sum(counts.values())
    print(accuracy)
    precision = counts.get("TP", 0) / (counts.get("TP", 0) + counts.get("FP", 0))
    print(precision)
    recall = counts.get("TP", 0) / (counts.get("TP", 0) + counts.get("FN", 0))
    print(recall)
    f1 = 2 * (precision * recall) / (precision + recall)

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "TP": counts.get("TP", 0),
                    "TN": counts.get("TN", 0),
                    "FP": counts.get("FP", 0),
                    "FN": counts.get("FN", 0),
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1-Score": f1,
                }
            ]
        ),
        hide_index=True,
    )


if df is not None:
    st.header("Global Results")
    # Get the Classification Class per techique
    for _technique, group_df in df.groupby(by="technique"):
        st.subheader(str(_technique).replace("_", " ").title())
        write_classification_classes_summary(group_df)

    st.header("Specific Analysis")

    with st.expander("🍾 Drill-Down"):
        technique = st.selectbox(
            "Technique", comparison_techniques, index=0, format_func=funcs_to_name.get
        )

        selected_logs = st.multiselect(
            "Event Log",
            event_logs,
            default=event_logs,
            format_func=event_logs_formatter.get,
        )

        st.markdown("##### Drifts")

        selected_drifts = ([True] if st.checkbox("Drift", value=True) else []) + (
            [False] if st.checkbox("No Drift", value=True) else []
        )

        st.markdown("##### Time Cost Function")
        selected_time_costs = (
            [True] if st.checkbox("Weighted Time Cost", value=True) else []
        ) + ([False] if st.checkbox("Normal Time Cost", value=True) else [])

        selected_binner_settings = st.multiselect(
            "Binner Settings",
            binner_settings,
            default=binner_settings,
            format_func=binner_setting_to_name.get,
        )

        filtered_df = df[df["technique"] == technique]
        filtered_df = filtered_df[filtered_df["log"].isin(selected_logs)]
        filtered_df = filtered_df[filtered_df["drifts"].isin(selected_drifts)]
        filtered_df = filtered_df[
            filtered_df["weighted_time_cost"].isin(selected_time_costs)
        ]
        filtered_df = filtered_df[
            filtered_df["binner_setting"].isin(selected_binner_settings)
        ]

        st.subheader("Technique Parameters")

        if technique == "emd_bootstrap":
            selected_bootstrapping_styles = st.multiselect(
                "Bootstrapping Styles",
                bootstrapping_styles,
                default=bootstrapping_styles,
                format_func=lambda x: x.replace("_", " ").title(),
            )

            selected_resample_percentages = st.multiselect(
                "Resample Percentages",
                resample_percentages,
                default=resample_percentages,
            )

            filtered_df = filtered_df[
                filtered_df["bootstrapping_style"].isin(selected_bootstrapping_styles)
            ]
            if "split sampling" in selected_bootstrapping_styles:
                # Also allow resample percentage to be nan
                filtered_df = filtered_df[
                    filtered_df["resample_percentage"].isin(
                        selected_resample_percentages
                    )
                    | filtered_df[
                        "resample_percentage"
                    ].isna()  # "na" is the resample percentage for split sampling as it is not applicable there
                ]
            else:
                filtered_df = filtered_df[
                    filtered_df["resample_percentage"].isin(
                        selected_resample_percentages
                    )
                ]

        elif technique == "ks_bootstrap":
            selected_ks_bootstrapping_styles = st.multiselect(
                "Bootstrapping Styles",
                ks_self_bootstrapping_styles,
                default=ks_self_bootstrapping_styles,
                format_func=ks_bootstrapping_styles_formatter.get,
            )

            filtered_df = filtered_df[
                filtered_df["self_bootstrapping_style"].isin(
                    selected_ks_bootstrapping_styles
                )
            ]
        elif technique == "double_bootstrap":
            selected_double_bootstrapping_styles = st.multiselect(
                "Bootstrapping Styles",
                double_bootstrap_styles,
                default=double_bootstrap_styles,
                format_func=double_bootstrap_styles_formatter.get,
            )

            filtered_df = filtered_df[
                filtered_df["bootstrapping_style"].isin(
                    selected_double_bootstrapping_styles
                )
            ]

    with st.expander("📅 Filtered Results Dataframe"):
        st.dataframe(filtered_df)

    st.subheader("Results")
    write_classification_classes_summary(filtered_df)
else:
    st.error(
        "The summary csv does not exist. Make sure it exists at EvaluationOutputs/summary.csv"
    )
