from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class RuleConfig:
    packet_small_p: float = 0.05
    packet_large_p: float = 0.95
    long_session_p: float = 0.99
    failed_logins_t: int = 4
    ip_rep_t: float = 0.8


def derive_thresholds(df: pd.DataFrame, cfg: RuleConfig) -> dict:
    q_pkt = df["network_packet_size"].quantile([cfg.packet_small_p, cfg.packet_large_p])
    q_sess = df["session_duration"].quantile([cfg.long_session_p])
    return {
        "PACKET_SMALL_T": int(q_pkt.loc[cfg.packet_small_p]),
        "PACKET_LARGE_T": int(q_pkt.loc[cfg.packet_large_p]),
        "LONG_SESSION_T": int(q_sess.loc[cfg.long_session_p]),
        "FAILED_LOGINS_T": cfg.failed_logins_t,
        "IP_REP_T": cfg.ip_rep_t,
    }


def build_rules(
    df: pd.DataFrame, cfg: RuleConfig | None = None
) -> dict[str, pd.Series]:
    """Return {'R1 name': mask, ...} with human-readable names."""
    if cfg is None:
        cfg = RuleConfig()
    th = derive_thresholds(df, cfg)

    R1 = (df["network_packet_size"] <= th["PACKET_SMALL_T"]) | (
        df["network_packet_size"] >= th["PACKET_LARGE_T"]
    )
    R2 = df["protocol_type"] == "ICMP"
    R3 = df["encryption_used"].isin(["DES", "None"])
    R4 = df["session_duration"] >= th["LONG_SESSION_T"]
    R5 = df["failed_logins"] >= th["FAILED_LOGINS_T"]
    R6 = df["unusual_time_access"] == 1
    R7 = df["ip_reputation_score"] >= th["IP_REP_T"]
    R8 = df["browser_type"] == "Unknown"

    return {
        "R1 packet_size tails": R1,
        "R2 ICMP": R2,
        "R3 weak/none encryption": R3,
        "R4 long session": R4,
        "R5 failed_logins >= 4": R5,
        "R6 unusual_time": R6,
        "R7 ip_rep >= 0.8": R7,
        "R8 browser=Unknown": R8,
    }
