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

# %%
import pandas as pd

from intrusion.paths import DATA_PATH, OUTPUTS_DIR, ensure_dirs
from intrusion.data import load_df
from intrusion.rules import build_rules
from intrusion.evaluation import mask_for_label  # supports "R5" or "R1 ^ R8"

pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 1000)
pd.set_option("display.max_columns", None)

ensure_dirs()

# %%
df = load_df(DATA_PATH)
rules = build_rules(df)
target = df["attack_detected"] == 1
# %%
prio_file = OUTPUTS_DIR / "priority.txt"
if prio_file.exists():
    PRIORITY = [
        line.strip() for line in prio_file.read_text().splitlines() if line.strip()
    ]
else:
    PRIORITY = [
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
print("Priority order:", PRIORITY)

# %%
rule_of_record = pd.Series(pd.NA, index=df.index, dtype="string")
for label in PRIORITY:
    m = mask_for_label(label, rules)
    unassigned = rule_of_record.isna()
    rule_of_record.loc[unassigned & m] = label

# %%
HIGH_SET = {"R5", "R7", "R4 ^ R6", "R1 ^ R8", "R3 ^ R8", "R8"}
LOW_SET = {"R6", "R3", "R1", "R2"}

severity = pd.Series("Low", index=df.index, dtype="string")
severity.loc[rule_of_record.isin(HIGH_SET)] = "High"

rule_short = {name.split()[0]: mask for name, mask in rules.items()}
fired_ids = list(rule_short.keys())


def row_reasons(i: int) -> str:
    ids = [rid for rid in fired_ids if bool(rule_short[rid].iloc[i])]
    return ",".join(sorted(ids))


any_rule_mask = pd.Series(False, index=df.index)
for m in rules.values():
    any_rule_mask |= m

reasons = pd.Series(pd.NA, index=df.index, dtype="string")
reasons.loc[any_rule_mask] = [row_reasons(i) for i in df.index[any_rule_mask]]

# %%
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
alerts["reasons"] = reasons.loc[alerts.index].values
alerts["severity"] = severity.loc[alerts.index].values
alerts["severity_rank"] = (
    alerts["severity"].map({"High": 3, "Medium": 2, "Low": 1}).astype("Int64")
)

alerts = alerts.sort_values(
    by=["severity_rank", "failed_logins", "ip_reputation_score", "session_duration"],
    ascending=[False, False, False, False],
).reset_index(drop=True)

out_path = OUTPUTS_DIR / "alerts_priority.csv"
alerts.to_csv(out_path, index=False)
print("Saved:", out_path)

# %%
baseline = float(target.mean())
merged = alerts.merge(
    df[["session_id", "attack_detected"]], on="session_id", how="left"
)
by_sev = (
    merged.groupby("severity")["attack_detected"]
    .agg(rows="count", precision="mean")
    .sort_values("rows", ascending=False)
)
print(f"Baseline attack rate: {baseline:.3f}\nPrecision by severity:\n{by_sev}")

reasons_summary = (
    alerts.groupby(["severity", "reasons"])
    .size()
    .reset_index(name="count")
    .sort_values(["severity", "count"], ascending=[True, False])
)
reasons_summary.to_csv(OUTPUTS_DIR / "reasons_summary.csv", index=False)
print(reasons_summary)
