import streamlit as st
import pandas as pd
import numpy as np

from intrusion.paths import DATA_PATH, ensure_dirs
from intrusion.data import load_df
from intrusion.rules import build_rules
from intrusion.evaluation import evaluate, incremental_coverage, mask_for_label
from intrusion.theme import apply_dark_theme, apply_light_theme
from intrusion.plots import (
    make_top_precision_figure,
    make_top_lift_figure,
    make_cumulative_recall_figure,
    make_incremental_precision_figure,
    make_precision_by_severity_figure,
    make_distributions_grid_figure,
    make_correlation_heatmap_figure,
)

st.set_page_config(page_title="Intrusion Analysis", layout="centered")
ensure_dirs()

st.sidebar.header("Controls")

uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
min_support = st.sidebar.slider(
    "Min support (Hits)", min_value=5, max_value=100, value=20, step=5
)

PAIR_IDS = ("R1", "R2", "R5", "R7")
TRIPLE_IDS = ("R1", "R3", "R5", "R6", "R7")


@st.cache_data
def get_df(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return load_df(DATA_PATH)


df = get_df(uploaded)
target = df["attack_detected"] == 1


@st.cache_data
def get_rules(df: pd.DataFrame):
    return build_rules(df)


rules = get_rules(df)

df_eval, baseline, total_attacks = evaluate(
    rules,
    target=(df["attack_detected"] == 1),
    min_support=min_support,
    pair_ids=PAIR_IDS,
    triple_ids=TRIPLE_IDS,
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Rules & Metrics", "Incremental Coverage", "Alerts", "Visuals"]
)

with tab1:
    st.subheader("Overview")
    total_rows = int(df.shape[0])
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{total_rows:,}")
    col1.metric("Baseline attack rate", f"{baseline:.3f}")
    col3.metric("Total attacks", f"{total_attacks}")
    # st.caption("Baseline = share of rows with attack_detected=1. Lift > 1 = better than baseline.")
    st.caption(
        "Baseline = share of sessions that are attacks in the whole dataset. "
        "Everything else compares against this baseline to judge lift/precision."
    )
    st.subheader("Distribution")
    with st.sidebar:
        dark_mode = st.checkbox("Use dark plot theme", value=True)

    if dark_mode:
        apply_dark_theme()
    else:
        apply_light_theme()

    t1, t2 = st.tabs(["Distribution", "Correlation"])
    with t1:
        st.caption(
            "Each panel shows the distribution of a numeric feature (histogram, optional KDE). "
            "Look for skew, outliers, or multimodal shapes that might explain why some rules work. "
            "Tall bars at the extremes often signal useful thresholds (e.g., long sessions)."
        )
        with st.expander("Plot Settings"):
            bins = st.slider("Bins", 5, 100, 30)
            cols_per_row = st.slider("Columns per row", 1, 4, 2)
            show_kde = st.checkbox("Overlay KDE (density curve)", True)
            density = st.checkbox("Normalize to density", True)
            dropna = st.checkbox("Drop NaN values", True)
            height_per_row = st.slider("Row height (inches)", 3.0, 6.0, 4.5, step=0.5)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.warning("No numeric columns found.")
        else:
            st.write(f"Found **{len(numeric_cols)}** numeric columns.")
            fig = make_distributions_grid_figure(
                df,
                bins=bins,
                cols_per_row=cols_per_row,
                show_kde=show_kde,
                density=density,
                dropna=dropna,
                height_per_row=height_per_row,
                include=numeric_cols,
            )
            st.pyplot(fig, clear_figure=True)

    with t2:
        with st.expander("Correlation Settings"):
            method = st.selectbox("Method", ["pearson", "spearman", "kendall"], index=0)
            mask_upper = st.checkbox("Show only lower triangle", True)
            annotate = st.checkbox("Annotate cells", True)

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.warning("No numeric columns found.")
        else:
            fig = make_correlation_heatmap_figure(
                df,
                include=numeric_cols,
                method=method,
                mask_upper=mask_upper,
                annotate=annotate,
            )
            st.pyplot(fig, clear_figure=True)
        st.caption(
            "Correlation (−1 to +1) between numeric features "
            f"(method = {method}). Darker = stronger relationship. "
            "Strong correlations can indicate redundancy or interactions to exploit in rules. "
            "The upper triangle is hidden to reduce clutter."
        )
        st.markdown("### Top correlations with `attack_detected`")

        if "attack_detected" in df.columns and pd.api.types.is_numeric_dtype(
            df["attack_detected"]
        ):
            _cols = list(dict.fromkeys(numeric_cols + ["attack_detected"]))
            corr = df[_cols].corr(method=method)  # pyright: ignore[]

            if "attack_detected" in corr.columns:
                top_n = st.slider(
                    "Show top N (numeric only)",
                    1,
                    min(30, len(_cols) - 1),
                    6,
                    key="topN_numeric",
                )
                top_corr = (
                    corr["attack_detected"]
                    .drop(
                        labels=["attack_detected"], errors="ignore"
                    )  # exclude self-corr
                    .rename("corr")  # pyright: ignore[]
                    .reset_index()
                    .rename(columns={"index": "feature"})
                )
                top_corr["abs_corr"] = top_corr["corr"].abs()
                top_corr = top_corr.sort_values("abs_corr", ascending=False).head(top_n)

                st.dataframe(top_corr, use_container_width=True)
                st.caption(
                    "Top absolute correlations with attack_detected (excluding self-correlation). "
                    "These features move most with the target—good places to focus thresholds and rule combinations."
                )
                st.download_button(
                    "Download top correlations (numeric)",
                    top_corr.to_csv(index=False),
                    "top_correlations_numeric.csv",
                    "text/csv",
                )
            else:
                st.info(
                    "Could not compute correlation against `attack_detected` with the selected method."
                )
        else:
            st.info(
                "`attack_detected` is missing or not numeric, so correlation to the target can’t be computed."
            )

with tab2:
    st.subheader("Rule Metrics")
    st.caption("This table uses a fixed set of pair/triple combos for consistency.")
    sort_by = st.selectbox("Sort by", ["Lift", "Precision", "Hits"], index=0)
    st.dataframe(
        df_eval.sort_values(sort_by, ascending=False), use_container_width=True
    )

    st.download_button(
        "Download rule_metrics.csv",
        df_eval.to_csv(index=False),
        "rule_metrics.csv",
        "text/csv",
    )

with tab3:
    st.subheader("Incremental Coverage")
    default_priority = [
        "R5",
        "R7",
        "R4 ^ R6",
        "R1 ^ R8",
        "R3 ^ R8",
        "R8",
        "R6",
        "R3",
        "R1",
        "R2",
    ]
    priority_text = st.text_area(
        "Priority (one per line)", "\n".join(default_priority), height=200
    )
    priority = [p.strip() for p in priority_text.splitlines() if p.strip()]

    @st.cache_data
    def compute_incremental(df, rules, priority):
        target = df["attack_detected"] == 1
        return incremental_coverage(rules, target, priority)

    incr_cov = compute_incremental(df, rules, tuple(priority))
    st.dataframe(incr_cov, use_container_width=True)
    st.download_button(
        "Download incremental_coverage.csv",
        incr_cov.to_csv(index=False),
        "incremental_coverage.csv",
        "text/csv",
    )

    covered = pd.Series(False, index=target.index)
    rows = []
    for label in priority:
        m = mask_for_label(label, rules)
        new_hits = int((m & ~covered).sum())
        new_tp = int(((m & target) & ~covered).sum())
        inc_prec = (new_tp / new_hits) if new_hits else 0.0
        rows.append(
            {
                "Rule": label,
                "New Hits": new_hits,
                "New Attacks": new_tp,
                "Incremental Precision": inc_prec,
            }
        )
        covered |= m
    inc_detail = pd.DataFrame(rows)
    st.markdown("**Incremental detail**")
    st.dataframe(inc_detail, use_container_width=True)
    st.download_button(
        "Download incremental_detail.csv",
        inc_detail.to_csv(index=False),
        "incremental_detail.csv",
        "text/csv",
    )

with tab4:
    st.subheader("Alerts (priority policy)")
    rule_of_record = pd.Series(pd.NA, index=df.index, dtype="string")
    for label in priority:
        m = mask_for_label(label, rules)
        unassigned = rule_of_record.isna()
        rule_of_record.loc[unassigned & m] = label

    HIGH_SET = {"R5", "R7", "R4 ^ R6", "R1 ^ R8", "R3 ^ R8", "R8"}
    severity = pd.Series("Low", index=df.index, dtype="string")
    severity.loc[rule_of_record.isin(HIGH_SET)] = "High"

    cols = [
        "session_id",
        "failed_logins",
        "login_attempts",
        "ip_reputation_score",
        "protocol_type",
        "encryption_used",
        "browser_type",
        "session_duration",
    ]
    alerts = df.loc[rule_of_record.notna(), cols].copy()
    alerts["rule_of_record"] = rule_of_record.loc[alerts.index].values
    alerts["severity"] = severity.loc[alerts.index].values
    alerts["severity_rank"] = alerts["severity"].map({"High": 3, "Medium": 2, "Low": 1})

    st.dataframe(alerts.head(1000), use_container_width=True)

    merged = alerts.merge(
        df[["session_id", "attack_detected"]], on="session_id", how="left"
    )
    by_sev = (
        merged.groupby("severity")["attack_detected"]
        .agg(rows="count", precision="mean")
        .sort_values("rows", ascending=False)
    )
    st.dataframe(by_sev.reset_index(), use_container_width=True)
    st.markdown("""
        **What this shows**

        - Each row gets a single **rule_of_record** based on priority (first matching rule wins).
        - Severity is mapped from the winning rule (e.g., high-precision rules → **High**).

        **Why it matters:** avoids double counting, makes triage clear, and ties severity to a primary reason.
    """)

with tab5:
    st.subheader("Visuals (interactive)")

    default_priority = [
        "R5",
        "R7",
        "R4 ^ R6",
        "R1 ^ R8",
        "R3 ^ R8",
        "R8",
        "R6",
        "R3",
        "R1",
        "R2",
    ]
    priority_text_viz = st.text_area(
        "Priority order for charts (one per line)",
        "\n".join(default_priority),
        height=160,
        key="viz_priority_text",
    )
    priority_viz = [p.strip() for p in priority_text_viz.splitlines() if p.strip()]

    @st.cache_data
    def compute_inc_detail(
        df: pd.DataFrame, rules: dict, priority: tuple[str, ...]
    ) -> pd.DataFrame:
        target = df["attack_detected"] == 1
        covered = pd.Series(False, index=target.index)
        rows = []
        for label in priority:
            m = mask_for_label(label, rules)
            new_hits = int((m & ~covered).sum())
            new_tp = int(((m & target) & ~covered).sum())
            inc_prec = (new_tp / new_hits) if new_hits else 0.0
            rows.append(
                {
                    "Rule": label,
                    "New Hits (uncovered)": new_hits,
                    "New Attacks (uncovered)": new_tp,
                    "Incremental Precision": inc_prec,
                }
            )
            covered |= m
        return pd.DataFrame(rows)

    @st.cache_data
    def compute_cum_table(
        df: pd.DataFrame, rules: dict, priority: tuple[str, ...]
    ) -> pd.DataFrame:
        target = df["attack_detected"] == 1
        covered = pd.Series(False, index=target.index)
        rows, cum_tp, cum_hits = [], 0, 0
        total_tp = int(target.sum())
        for label in priority:
            m = mask_for_label(label, rules)
            new_hits = int((m & ~covered).sum())
            new_tp = int(((m & target) & ~covered).sum())
            cum_hits += new_hits
            cum_tp += new_tp
            inc_prec = (new_tp / new_hits) if new_hits else 0.0
            cum_recall = (cum_tp / total_tp) if total_tp else 0.0
            rows.append(
                {
                    "Rule": label,
                    "New Hits": new_hits,
                    "New Attacks": new_tp,
                    "Incremental Precision": inc_prec,
                    "Cumulative Hits": cum_hits,
                    "Cumulative Attacks": cum_tp,
                    "Cumulative Recall": cum_recall,
                }
            )
            covered |= m
        return pd.DataFrame(rows)

    inc_detail_viz = compute_inc_detail(df, rules, tuple(priority_viz))
    cum_table_viz = compute_cum_table(df, rules, tuple(priority_viz))

    rule_of_record_viz = pd.Series(pd.NA, index=df.index, dtype="string")
    for label in priority_viz:
        m = mask_for_label(label, rules)
        unassigned = rule_of_record_viz.isna()
        rule_of_record_viz.loc[unassigned & m] = label

    HIGH_SET = {"R5", "R7", "R4 ^ R6", "R1 ^ R8", "R3 ^ R8", "R8"}
    severity_viz = pd.Series("Low", index=df.index, dtype="string")
    severity_viz.loc[rule_of_record_viz.isin(HIGH_SET)] = "High"

    alerts_viz = df[["session_id"]].copy()
    alerts_viz["severity"] = severity_viz
    by_sev_viz = (  # pyright: ignore[]
        alerts_viz.merge(
            df[["session_id", "attack_detected"]], on="session_id", how="left"
        )
        .groupby("severity")["attack_detected"]
        .agg(rows="count", precision="mean")
        .sort_values("rows", ascending=False)
    )
    v1, v2, v3, v4, v5 = st.tabs(
        [
            "Top Precision",
            "Cumulative Recall",
            "Incremental Precision",
            "Top Lift",
            "Precision by Severity",
        ]
    )

    with v1:
        top_n = st.slider("Top N", 5, 30, 15, key="tp_topn")
        short = st.checkbox("Short rule labels", True, key="tp_short")
        rotate = st.slider("Y-label rotation", 0, 90, 0, step=5, key="tp_rot")
        fig = make_top_precision_figure(
            df_eval, baseline, top_n=top_n, short_labels=short, rotate=rotate
        )
        st.pyplot(fig, clear_figure=True)
        st.caption(
            "Rules with highest precision (dashed line = baseline). "
            "Great candidates for high-severity alerts because they minimize false positives."
        )

    with v2:
        short = st.checkbox("Short rule labels", True, key="cum_short")
        rotate = st.slider("X-label rotation", 0, 90, 45, step=5, key="cum_rot")
        fig = make_cumulative_recall_figure(
            cum_table_viz, short_labels=short, rotate=rotate
        )
        st.pyplot(fig, clear_figure=True)
        st.dataframe(cum_table_viz, use_container_width=True)
        st.caption(
            "How total recall grows as adding rules in this order. "
            "Plateaus mean a rule added little new coverage; jumps mean the rule found many previously missed attacks."
        )

    with v3:
        short = st.checkbox("Short rule labels", True, key="inc_short")
        rotate = st.slider("X-label rotation", 0, 90, 45, step=5, key="inc_rot")
        fig = make_incremental_precision_figure(
            inc_detail_viz, short_labels=short, rotate=rotate
        )
        st.pyplot(fig, clear_figure=True)

        st.dataframe(inc_detail_viz, use_container_width=True)
        st.markdown("""
            **What this shows (new coverage only)**
            - **New Hits** — rows newly captured by this rule (excluding anything earlier rules already caught).
            - **New Attacks** — true positives among those new hits.
            - **Incremental Precision** — `New Attacks / New Hits` (quality of this step’s *marginal* catch).
            **How to use it:** put rules with **high incremental precision** early; push noisy, broad rules later.
            """)

    with v4:
        top_n = st.slider("Top N", 5, 30, 15, key="lift_topn")
        short = st.checkbox("Short rule labels", True, key="lift_short")
        rotate = st.slider("Y-label rotation", 0, 90, 0, step=5, key="lift_rot")
        fig = make_top_lift_figure(
            df_eval, top_n=top_n, short_labels=short, rotate=rotate
        )
        st.pyplot(fig, clear_figure=True)
        st.caption(
            "Rules with highest lift (precision relative to baseline). "
            "Lift > 1 means better than guessing based on base rate; higher is stronger."
        )
    with v5:
        fig = make_precision_by_severity_figure(by_sev_viz, baseline)
        st.pyplot(fig, clear_figure=True)
        st.dataframe(by_sev_viz.reset_index(), use_container_width=True)
        st.caption(
            "Observed precision for each severity bucket (dashed = baseline). "
            "‘High’ should sit well above baseline."
        )
