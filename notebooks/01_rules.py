# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: notebooks//ipynb,notebooks//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
# ---

# %% [imports]
import pandas as pd

pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 1000)
pd.set_option("display.max_columns", None)

from intrusion.paths import DATA_PATH, OUTPUTS_DIR, ensure_dirs
from intrusion.data import load_df
from intrusion.rules import build_rules
from intrusion.evaluation import evaluate, mask_for_label, incremental_coverage
from IPython.display import display

ensure_dirs()

# %% [load data + rules]
df = load_df(DATA_PATH)
rules = build_rules(df)
target = df["attack_detected"] == 1

# %% [config]
MIN_SUPPORT = 20
PAIR_IDS   = ("R1", "R2", "R5", "R7")
TRIPLE_IDS = ("R1", "R3", "R5", "R6", "R7")

for sid in set(PAIR_IDS) | set(TRIPLE_IDS):
    assert any(name.startswith(sid) for name in rules), f"Missing rule with short ID {sid}"

# %% [evaluate: singles + pairs + triples]
df_eval, baseline, total_attacks = evaluate(
    rules,
    target,
    min_support=MIN_SUPPORT,
    pair_ids=PAIR_IDS,
    triple_ids=TRIPLE_IDS,
)

print(f"Baseline={baseline:.3f} | Total attacks={total_attacks}")
display(df_eval.head(15))

(df_eval
 .to_csv(OUTPUTS_DIR / "rule_metrics.csv", index=False))
(df_eval.head(25)
 .to_csv(OUTPUTS_DIR / "rule_metrics_top25.csv", index=False))

# %% [choose deployment priority]
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

(OUTPUTS_DIR / "priority.txt").write_text("\n".join(PRIORITY))
print("Saved priority to", OUTPUTS_DIR / "priority.txt")

# %% [incremental coverage]
incr_cov = incremental_coverage(rules, target, PRIORITY)
incr_cov.to_csv(OUTPUTS_DIR / "incremental_coverage.csv", index=False)
print("\nIncremental coverage:")
display(incr_cov)

# %% [incremental detail]
covered = pd.Series(False, index=target.index)
rows = []
for label in PRIORITY:
    m = mask_for_label(label, rules) # supports "R5" or combos like "R1 ^ R8"
    new_hits = int((m & ~covered).sum())
    new_tp   = int(((m & target) & ~covered).sum())
    inc_prec = (new_tp / new_hits) if new_hits else 0.0
    rows.append({
        "Rule": label,
        "New Hits (uncovered)": new_hits,
        "New Attacks (uncovered)": new_tp,
        "Incremental Precision": inc_prec,
    })
    covered |= m

inc_detail = pd.DataFrame(rows)
inc_detail.to_csv(OUTPUTS_DIR / "incremental_detail.csv", index=False)
print("\nIncremental detail:")
display(inc_detail)

# %% [cumulative coverage]
covered = pd.Series(False, index=target.index)
rows = []
cum_tp = 0
cum_hits = 0
total_tp = int(target.sum())

for label in PRIORITY:
    m = mask_for_label(label, rules)
    new_hits = int((m & ~covered).sum())
    new_tp   = int(((m & target) & ~covered).sum())
    cum_hits += new_hits
    cum_tp   += new_tp
    inc_prec = (new_tp / new_hits) if new_hits else 0.0
    cum_recall = (cum_tp / total_tp) if total_tp else 0.0
    rows.append({
        "Rule": label,
        "New Hits": new_hits,
        "New Attacks": new_tp,
        "Incremental Precision": inc_prec,
        "Cumulative Hits": cum_hits,
        "Cumulative Attacks": cum_tp,
        "Cumulative Recall": cum_recall,
    })
    covered |= m

cum_table = pd.DataFrame(rows)
cum_table.to_csv(OUTPUTS_DIR / "cumulative_coverage.csv", index=False)
print("\nCumulative coverage:")
display(cum_table)

