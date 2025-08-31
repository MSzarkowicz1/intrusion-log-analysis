from __future__ import annotations
import itertools
import re
import pandas as pd


def _metrics(
    name: str, mask: pd.Series, target: pd.Series, base_rate: float, total_attacks: int
) -> dict:
    hits = int(mask.sum())
    tp = int((mask & target).sum())
    precision = tp / hits if hits else 0.0
    recall = tp / total_attacks if total_attacks else 0.0
    lift = (precision / base_rate) if (base_rate and hits) else 0.0
    return {
        "Rule": name,
        "Hits": hits,
        "Attacks Detected": tp,
        "Precision": precision,
        "Recall": recall,
        "Lift": lift,
    }


def _piece_to_full_name(piece: str, rules: dict) -> str:
    """
    Accept either a full rule label (exact key in `rules`)
    or a short ID (R5) and return the full label.
    """
    if piece in rules:
        return piece
    alias = {full.split()[0]: full for full in rules.keys()}
    if piece in alias:
        return alias[piece]
    raise KeyError(f"Rule piece '{piece}' not found (neither full key nor short ID).")


def evaluate(
    rules: dict[str, pd.Series],
    target: pd.Series,
    min_support: int = 20,
    pair_ids: tuple[str, ...] = ("R1", "R2", "R5", "R7"),
    triple_ids: tuple[str, ...] | None = None,
    sort_by: tuple[str, ...] = ("Lift", "Precision", "Hits"),
) -> tuple[pd.DataFrame, float, int]:
    """
    Score singles + selected pairs (+ optional triples).
    Returns (df_eval, baseline_rate, total_attacks).
    """
    total_attacks = int(target.sum())
    base_rate = float(target.mean()) if total_attacks else 0.0

    rows: list[dict] = []

    # singles
    for full_name, mask in rules.items():
        rows.append(_metrics(full_name, mask, target, base_rate, total_attacks))

    # map short IDs
    short = {
        full_name.split()[0]: (full_name, mask) for full_name, mask in rules.items()
    }

    # pairs
    for a, b in itertools.combinations(pair_ids, 2):
        if a in short and b in short:
            name_a, m_a = short[a]
            name_b, m_b = short[b]
            combo = m_a & m_b
            if int(combo.sum()) >= min_support:
                rows.append(
                    _metrics(
                        f"{name_a} ^ {name_b}", combo, target, base_rate, total_attacks
                    )
                )

    # triples
    if triple_ids:
        for a, b, c in itertools.combinations(triple_ids, 3):
            if a in short and b in short and c in short:
                name_a, m_a = short[a]
                name_b, m_b = short[b]
                name_c, m_c = short[c]
                combo = m_a & m_b & m_c
                if int(combo.sum()) >= min_support:
                    rows.append(
                        _metrics(
                            f"{name_a} ^ {name_b} ^ {name_c}",
                            combo,
                            target,
                            base_rate,
                            total_attacks,
                        )
                    )

    df_eval = (
        pd.DataFrame(rows)
        .sort_values(list(sort_by), ascending=[False, False, False])
        .reset_index(drop=True)
    )
    return df_eval, base_rate, total_attacks


def mask_from_combo_label(label: str, rules: dict) -> pd.Series:
    """
    Build a mask for 'A ^ B ^ C' where A/B/C are full labels or short IDs.
    Works for singles, pairs, or triples.
    """
    parts = [p.strip() for p in re.split(r"\s*(?:\^|∧)\s*", label) if p.strip()]
    base_index = next(iter(rules.values())).index  # no dependency on a global df
    m = pd.Series(True, index=base_index)
    for p in parts:
        full = _piece_to_full_name(p, rules)
        m = m & rules[full]
    return m


def mask_for_label(label: str, rules: dict) -> pd.Series:
    """
    If `label` is a single full label or short ID, return it.
    If it's a combo ('A ^ B' …), delegate to mask_from_combo_label.
    """
    if "^" in label or "∧" in label:
        return mask_from_combo_label(label, rules)
    full = _piece_to_full_name(label, rules)
    return rules[full]


def incremental_coverage(
    rules: dict, target: pd.Series, priority: list[str]
) -> pd.DataFrame:
    """
    Given a priority-ordered list of rule labels (full or short IDs),
    report how many *new* attacks each rule catches beyond earlier ones.
    """
    covered = pd.Series(False, index=target.index)
    rows = []
    for label in priority:
        m = mask_for_label(label, rules)
        new_tp = int(((m & target) & ~covered).sum())
        rows.append({"Rule": label, "New Attacks (beyond earlier rules)": new_tp})
        covered |= m & target
    return pd.DataFrame(rows)
