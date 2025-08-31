# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
# ---

# %% [imports]
import pandas as pd
from intrusion.paths import DATA_PATH, OUTPUTS_DIR, FIGURES_DIR, ensure_dirs
from intrusion.data import load_df
from intrusion.plots import (
    plot_top_precision,
    plot_top_lift,
    plot_cumulative_recall,
    plot_incremental_precision,
    plot_precision_by_severity,
    plot_numeric_distributions_grid,
    plot_numeric_distributions_individual,
)

pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 1200)
pd.set_option("display.max_columns", None)

ensure_dirs()

# %% [load artifacts]
df_eval = pd.read_csv(OUTPUTS_DIR / "rule_metrics.csv")
inc_detail = pd.read_csv(OUTPUTS_DIR / "incremental_detail.csv")
cum_table = pd.read_csv(OUTPUTS_DIR / "cumulative_coverage.csv")
alerts_priority = pd.read_csv(OUTPUTS_DIR / "alerts_priority.csv")

df = load_df(DATA_PATH)
baseline = float((df["attack_detected"] == 1).mean())
print(f"Baseline attack rate = {baseline:.3f}")

# %% [precision by severity]
merged = alerts_priority.merge(
    df[["session_id", "attack_detected"]], on="session_id", how="left"
)
by_sev = (
    merged.groupby("severity")["attack_detected"]
    .agg(rows="count", precision="mean")
    .sort_values("rows", ascending=False)
)
print(by_sev)

# %% [plots]
plot_top_precision(df_eval, baseline, FIGURES_DIR / "top_precision.png", top_n=15)
plot_top_lift(df_eval, FIGURES_DIR / "top_lift.png", top_n=15)
plot_cumulative_recall(cum_table, FIGURES_DIR / "cumulative_recall.png")
plot_incremental_precision(inc_detail, FIGURES_DIR / "incremental_precision.png")
plot_precision_by_severity(by_sev, baseline, FIGURES_DIR / "precision_by_severity.png")

print("Saved figures to:", FIGURES_DIR)

# %% [distributions]
num_cols_to_plot = [
    "network_packet_size",
    "protocol_type",
    "login_attempts",
    "session_duration",
    "ip_reputation_score",
    "failed_logins",
    "browser_type",
    "unusual_time_access",
    "attack_detected"
]
log_x = {"session_duration", "network_packet_size"}

plot_numeric_distributions_grid(
    df,
    FIGURES_DIR / "distributions.png",
    bins=30,
    cols_per_row=2,
    show_kde=True,
    density=True,
    dropna=True,
    height_per_row=4.5,
    include=num_cols_to_plot,
)

# plot_numeric_distributions_individual(
#     df,
#     FIGURES_DIR,
#     bins=30,
#     show_kde=True,
#     density=True,
#     dropna=True,
#     include=num_cols_to_plot,
# )

