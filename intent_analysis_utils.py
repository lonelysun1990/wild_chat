"""
Intent spotcheck and temporal trends for WildChat. Used by intent_spotcheck_and_trends.ipynb.
Load classified data, sample conversations per category, and plot weekly / within-week / hourly intent trends.
"""

from __future__ import annotations

import random
import textwrap
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Load and parse
# ---------------------------------------------------------------------------


def _normalize_conversation_cell(conv: Any) -> list[dict]:
    """Ensure conversation is a list of plain Python dicts (handles parquet numpy list/record)."""
    if conv is None:
        return []
    if not isinstance(conv, list):
        try:
            conv = list(conv)
        except (TypeError, ValueError):
            return []
    out = []
    for msg in conv:
        d = _message_to_dict(msg)
        if d is not None:
            out.append(d)
    return out


def ensure_conversation_normalized(
    df: pd.DataFrame,
    conversation_col: str = "conversation",
) -> pd.DataFrame:
    """Ensure conversation column is list of plain dicts (handles parquet numpy list/record)."""
    if conversation_col not in df.columns:
        return df
    out = df.copy()
    out[conversation_col] = out[conversation_col].map(_normalize_conversation_cell)
    return out


def load_classified_for_analysis(
    output_dir: str | Path,
    full_name: str = "intent_classified.parquet",
) -> pd.DataFrame:
    """Load full classified parquet and ensure conversation column is list-of-dicts."""
    from eda_utils import ensure_conversation_parsed
    from intent_taxonomy import load_classified

    df = load_classified(output_dir, full_name=full_name)
    df = ensure_conversation_parsed(df)
    df = ensure_conversation_normalized(df)
    return df


# ---------------------------------------------------------------------------
# Spot check: sample and format conversations
# ---------------------------------------------------------------------------


def sample_by_category(
    df: pd.DataFrame,
    category_col: str,
    n_per_cat: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Stratified random sample: n_per_cat rows per category (or all if category has fewer)."""
    rng = random.Random(seed)
    rows = []
    for cat, group in df.groupby(category_col, dropna=False):
        n = min(n_per_cat, len(group))
        if n == 0:
            continue
        indices = rng.sample(list(group.index), k=n)
        rows.append(df.loc[indices])
    if not rows:
        return df.iloc[0:0].copy()
    return pd.concat(rows, ignore_index=True)


def _py_val(v: Any) -> Any:
    """Convert numpy scalar to Python scalar; leave rest as-is."""
    if hasattr(v, "item") and getattr(v, "ndim", -1) == 0:
        return v.item()
    return v


def _message_to_dict(msg: Any) -> dict | None:
    """Convert a message to a plain dict (handles numpy record, dict, etc.). Returns None if not usable."""
    if isinstance(msg, dict):
        return {k: _py_val(v) for k, v in msg.items()}
    try:
        if hasattr(msg, "keys"):
            return {k: _py_val(msg[k]) for k in msg.keys()}
        return {k: _py_val(v) for k, v in dict(msg).items()}
    except (TypeError, ValueError, AttributeError):
        return None


def format_conversation(conv: Any, width: int = 80) -> str:
    """Format a conversation (list of {role, content}) as a string with textwrap-filled content.
    Handles conv as list or numpy array; each message as dict or numpy record."""
    if conv is None:
        return "(invalid conversation)"
    if not isinstance(conv, list):
        try:
            conv = list(conv)
        except (TypeError, ValueError):
            return "(invalid conversation)"
    lines = []
    for msg in conv:
        d = _message_to_dict(msg)
        if not d:
            continue
        role = d.get("role", "?")
        content = d.get("content")
        if content is None:
            content = ""
        content = str(content).strip()
        if content:
            wrapped = textwrap.fill(content, width=width)
            lines.append(f"[{role}]\n{wrapped}")
        else:
            lines.append(f"[{role}]\n(empty)")
    return "\n\n".join(lines) if lines else "(no messages)"


# ---------------------------------------------------------------------------
# Weekly intent counts and percentages
# ---------------------------------------------------------------------------


def weekly_intent_counts(
    df: pd.DataFrame,
    label_col: str,
    timestamp_col: str = "timestamp",
    freq: str = "W",
) -> pd.DataFrame:
    """Count per (period, label). period is week (or freq) as string."""
    out = df[[timestamp_col, label_col]].copy()
    out["period"] = pd.to_datetime(out[timestamp_col], utc=True).dt.to_period(freq).astype(str)
    return out.groupby(["period", label_col]).size().reset_index(name="count")


def weekly_intent_percentages(
    df: pd.DataFrame,
    label_col: str,
    timestamp_col: str = "timestamp",
    freq: str = "W",
) -> pd.DataFrame:
    """Weekly counts per label with percentage of that week's total."""
    counts = weekly_intent_counts(df, label_col=label_col, timestamp_col=timestamp_col, freq=freq)
    week_totals = counts.groupby("period")["count"].transform("sum")
    counts["pct"] = 100.0 * counts["count"] / week_totals
    return counts


def _parse_events(events: list[tuple[str, str]] | str | Path | None) -> list[tuple[pd.Timestamp, str]]:
    """Convert events to list of (timestamp, label). events can be list of (date_str, label) or path to CSV (date, event_name)."""
    if events is None:
        return []
    out = []
    if isinstance(events, (str, Path)):
        path = Path(events)
        if not path.exists():
            return []
        ev_df = pd.read_csv(path)
        date_col = "date" if "date" in ev_df.columns else ev_df.columns[0]
        name_col = "event_name" if "event_name" in ev_df.columns else (ev_df.columns[1] if len(ev_df.columns) > 1 else "event")
        for _, row in ev_df.iterrows():
            ts = pd.to_datetime(row[date_col], utc=True)
            label = str(row.get(name_col, row.get(ev_df.columns[1] if len(ev_df.columns) > 1 else "event", "?")))
            out.append((ts, label))
        return out
    for date_str, label in events:
        ts = pd.to_datetime(date_str, utc=True)
        out.append((ts, str(label)))
    return out


def plot_weekly_intent_percentages(
    weekly_pct: pd.DataFrame,
    period_col: str = "period",
    label_col: str | None = None,
    pct_col: str = "pct",
    events: list[tuple[str, str]] | str | Path | None = None,
    stacked: bool = False,
    figsize: tuple[float, float] = (12, 6),
):
    """Plot weekly intent percentages (line or stacked area). Optionally add vertical lines for events."""
    import matplotlib.pyplot as plt

    if label_col is None:
        # infer: column that is not period_col or pct_col
        cand = [c for c in weekly_pct.columns if c not in (period_col, pct_col, "count")]
        label_col = cand[0] if cand else "label"

    piv = weekly_pct.pivot(index=period_col, columns=label_col, values=pct_col).fillna(0)
    # ensure period is sortable (string periods like 2023-W21 sort fine)
    piv = piv.reindex(sorted(piv.index))

    fig, ax = plt.subplots(figsize=figsize)
    if stacked:
        piv.plot.area(ax=ax, stacked=True, alpha=0.7)
    else:
        piv.plot(ax=ax, marker="o", markersize=3)

    ax.set_xlabel("Week (period)")
    ax.set_ylabel("Percentage of conversations")
    ax.legend(title=label_col, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_title("Intent percentage by week")
    plt.tight_layout()

    event_list = _parse_events(events)
    if event_list:
        period_vals = list(piv.index)
        ytop = ax.get_ylim()[1]
        for ts, ev_label in event_list:
            try:
                per_str = str(pd.Period(ts, freq="W"))
            except Exception:
                continue
            pos = None
            for i, p in enumerate(period_vals):
                if str(p) == per_str:
                    pos = i
                    break
            if pos is not None:
                ax.axvline(x=pos, color="gray", linestyle="--", alpha=0.8)
                ax.text(pos, ytop * 0.98, ev_label, rotation=90, fontsize=8, va="top", ha="right")
    return fig, ax


# ---------------------------------------------------------------------------
# Within-week: day-of-week percentages
# ---------------------------------------------------------------------------


def within_week_intent_percentages(
    df: pd.DataFrame,
    label_col: str,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Intent count and percentage per (week, weekday). weekday 0=Monday, 6=Sunday."""
    out = df[[timestamp_col, label_col]].copy()
    ts = pd.to_datetime(out[timestamp_col], utc=True)
    out["period"] = ts.dt.to_period("W").astype(str)
    out["weekday"] = ts.dt.weekday  # 0=Mon, 6=Sun
    counts = out.groupby(["period", "weekday", label_col]).size().reset_index(name="count")
    # pct within that (period, weekday) over all labels
    day_totals = counts.groupby(["period", "weekday"])["count"].transform("sum")
    counts["pct"] = 100.0 * counts["count"] / day_totals
    return counts


def plot_within_week_heatmap(
    within_week_df: pd.DataFrame,
    label_value: str,
    label_col: str = "intent_major",
    period_col: str = "period",
    weekday_col: str = "weekday",
    pct_col: str = "pct",
    figsize: tuple[float, float] = (14, 6),
):
    """Heatmap: rows = week, cols = weekday, values = percentage for the given intent."""
    import matplotlib.pyplot as plt
    import numpy as np

    sub = within_week_df[within_week_df[label_col] == label_value].copy()
    if sub.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(f"No data for {label_value}")
        return fig, ax

    piv = sub.pivot(index=period_col, columns=weekday_col, values=pct_col).fillna(0)
    piv = piv.reindex(sorted(piv.index))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(piv.values, aspect="auto", cmap="Blues")
    ax.set_xticks(range(7))
    ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels(list(piv.index), fontsize=7)
    ax.set_ylabel("Week")
    ax.set_xlabel("Day of week")
    ax.set_title(f"Within-week percentage: {label_value}")
    plt.colorbar(im, ax=ax, label="%")
    plt.tight_layout()
    return fig, ax


def plot_weekday_profile(
    within_week_df: pd.DataFrame,
    label_col: str = "intent_major",
    weekday_col: str = "weekday",
    pct_col: str = "pct",
    figsize: tuple[float, float] = (10, 5),
):
    """Average percentage by weekday (0–6) per intent: typical week profile."""
    import matplotlib.pyplot as plt

    avg = within_week_df.groupby([label_col, weekday_col])[pct_col].mean().reset_index()
    piv = avg.pivot(index=weekday_col, columns=label_col, values=pct_col).fillna(0)

    fig, ax = plt.subplots(figsize=figsize)
    piv.plot(ax=ax, marker="o")
    ax.set_xticks(range(7))
    ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Average % of conversations")
    ax.set_title("Typical week profile by intent")
    ax.legend(title=label_col, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Hour-of-day trends (UTC or optional timezone)
# ---------------------------------------------------------------------------


def hourly_intent_counts(
    df: pd.DataFrame,
    label_col: str,
    timestamp_col: str = "timestamp",
    tz: str | None = None,
) -> pd.DataFrame:
    """Count per (hour, label). Hour is 0–23 in UTC or in tz if provided."""
    out = df[[timestamp_col, label_col]].copy()
    ts = pd.to_datetime(out[timestamp_col], utc=True)
    if tz:
        ts = ts.dt.tz_convert(tz)
    out["hour"] = ts.dt.hour
    return out.groupby(["hour", label_col]).size().reset_index(name="count")


def hourly_intent_percentages(
    df: pd.DataFrame,
    label_col: str,
    timestamp_col: str = "timestamp",
    tz: str | None = None,
) -> pd.DataFrame:
    """Hourly counts per label with percentage of that hour's total."""
    counts = hourly_intent_counts(df, label_col=label_col, timestamp_col=timestamp_col, tz=tz)
    hour_totals = counts.groupby("hour")["count"].transform("sum")
    counts["pct"] = 100.0 * counts["count"] / hour_totals
    return counts


def plot_hourly_intent_distribution(
    hourly_df: pd.DataFrame,
    label_col: str = "intent_major",
    value_col: str = "pct",
    title_suffix: str = "(UTC)",
    figsize: tuple[float, float] = (12, 5),
):
    """Bar or line: hour 0–23 vs value (pct or count) per intent."""
    import matplotlib.pyplot as plt

    piv = hourly_df.pivot(index="hour", columns=label_col, values=value_col).fillna(0)
    piv = piv.reindex(range(24), fill_value=0).fillna(0)

    fig, ax = plt.subplots(figsize=figsize)
    piv.plot(ax=ax, kind="bar", width=0.8, alpha=0.8)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Percentage" if value_col == "pct" else "Count")
    ax.set_title(f"Intent by hour of day {title_suffix}")
    ax.legend(title=label_col, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels(range(0, 24, 2))
    plt.tight_layout()
    return fig, ax
