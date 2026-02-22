"""
Utilities for vertical insights notebook. Load parquet data from vertical_output/all,
build summary tables, and produce survival-style and comparison plots.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_vertical_parquets(vertical_output_all_dir: str | Path) -> pd.DataFrame:
    """
    Discover all *.parquet in vertical_output_all_dir, read each, attach a category
    column from filename (pattern: commercial_vertical_brands_{category}_{range}.parquet),
    parse brands and brands_enriched from JSON when present, and concatenate.
    """
    base = Path(vertical_output_all_dir)
    if not base.exists():
        raise FileNotFoundError(f"Directory not found: {base}")

    pattern = re.compile(r"commercial_vertical_brands_(.+?)_([^_]+)\.parquet$", re.IGNORECASE)
    frames: list[pd.DataFrame] = []

    for path in sorted(base.glob("*.parquet")):
        match = pattern.search(path.name)
        category = match.group(1) if match else path.stem
        df = pd.read_parquet(path)
        df = df.copy()
        df["category"] = category
        # Parse JSON columns if present
        if "brands" in df.columns:
            df["brands"] = df["brands"].apply(_safe_json_list)
        if "brands_enriched" in df.columns:
            df["brands_enriched"] = df["brands_enriched"].apply(_safe_json_list)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)

    # Ensure n_messages: if missing, compute from conversation length
    if "n_messages" not in out.columns and "conversation" in out.columns:
        out["n_messages"] = out["conversation"].map(
            lambda c: len(c) if isinstance(c, (list, np.ndarray)) else 0
        )

    return out


def _safe_json_list(x: Any) -> list:
    """Parse JSON list from string or return list as-is."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            v = json.loads(x)
            return v if isinstance(v, list) else []
        except (json.JSONDecodeError, TypeError):
            return []
    return []


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def table_conversation_length_by_category(
    df: pd.DataFrame,
    n_messages_col: str = "n_messages",
    category_col: str = "category",
) -> pd.DataFrame:
    """Per category: count, mean, median, percentiles (25, 75, 90) of n_messages."""
    if n_messages_col not in df.columns or category_col not in df.columns:
        return pd.DataFrame()
    agg = df.groupby(category_col)[n_messages_col].agg(
        count="count",
        mean="mean",
        median="median",
        p25=lambda s: s.quantile(0.25),
        p75=lambda s: s.quantile(0.75),
        p90=lambda s: s.quantile(0.90),
    ).reset_index()
    return agg.round(2)


def table_conversation_length_by_category_vertical(
    df: pd.DataFrame,
    n_messages_col: str = "n_messages",
    category_col: str = "category",
    vertical_col: str = "vertical_tier1_llm",
) -> pd.DataFrame:
    """Per (category, vertical): count, mean, median of n_messages."""
    for c in (n_messages_col, category_col, vertical_col):
        if c not in df.columns:
            return pd.DataFrame()
    agg = df.groupby([category_col, vertical_col])[n_messages_col].agg(
        count="count",
        mean="mean",
        median="median",
    ).reset_index()
    return agg.round(2)


def table_deal_size_by_vertical(
    df: pd.DataFrame,
    deal_col: str = "deal_size_usd",
    vertical_col: str = "vertical_tier1_llm",
    total_brands_col: str = "total_brands",
    require_has_brand: bool = True,
) -> pd.DataFrame:
    """
    By vertical: count (non-null), mean, median, percentiles of deal_size_usd.
    If require_has_brand is True, only include conversations with at least one brand
    (total_brands > 0); excludes null/none deal_size_usd and zero/negative values.
    """
    if deal_col not in df.columns or vertical_col not in df.columns:
        return pd.DataFrame()
    valid = df[df[deal_col].notna()].copy()
    if require_has_brand and total_brands_col in df.columns:
        valid = valid[valid[total_brands_col].fillna(0) > 0]
    valid = valid[valid[deal_col] > 0]
    if valid.empty:
        return pd.DataFrame()
    agg = valid.groupby(vertical_col)[deal_col].agg(
        count="count",
        mean="mean",
        median="median",
        p25=lambda s: s.quantile(0.25),
        p75=lambda s: s.quantile(0.75),
        p90=lambda s: s.quantile(0.90),
    ).reset_index()
    return agg.round(2)


def table_brand_engagement_by_category(
    df: pd.DataFrame,
    category_col: str = "category",
    total_brands_col: str = "total_brands",
) -> pd.DataFrame:
    """Per category: total conversations, count with at least one brand, percentage."""
    if category_col not in df.columns or total_brands_col not in df.columns:
        return pd.DataFrame()
    g = pd.DataFrame({"total": df.groupby(category_col).size()})
    g["with_brand_count"] = df.groupby(category_col)[total_brands_col].apply(lambda s: (s.fillna(0) > 0).sum())
    g["pct_with_brand"] = (g["with_brand_count"] / g["total"].replace(0, np.nan) * 100).round(2)
    return g.reset_index()


def table_follow_up_pct_by_vertical(
    df: pd.DataFrame,
    follow_up_col: str = "any_user_follow_up_on_brand",
    vertical_col: str = "vertical_tier1_llm",
) -> pd.DataFrame:
    """By vertical: count of conversations, count with follow-up, percentage."""
    if follow_up_col not in df.columns or vertical_col not in df.columns:
        return pd.DataFrame()
    g = df.groupby(vertical_col).agg(
        count=(follow_up_col, "count"),
        follow_up_sum=(follow_up_col, "sum"),
    )
    g.columns = ["count", "follow_up_count"]
    g["pct"] = (g["follow_up_count"] / g["count"].replace(0, np.nan) * 100).round(2)
    return g.reset_index()


def expand_brands_enriched(
    df: pd.DataFrame,
    brands_enriched_col: str = "brands_enriched",
    vertical_col: str = "vertical_tier1_llm",
    category_col: str = "category",
) -> pd.DataFrame:
    """
    Expand to one row per brand (per conversation). Each row has mention_span,
    mention_count_total, vertical_tier1_llm, category. Drops rows with no brands.
    """
    if brands_enriched_col not in df.columns:
        return pd.DataFrame()
    keep_cols = [c for c in [vertical_col, category_col, "conversation_id"] if c in df.columns]
    rows: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        enriched = row.get(brands_enriched_col) or []
        if not isinstance(enriched, list):
            continue
        for b in enriched:
            if not isinstance(b, dict):
                continue
            r: dict[str, Any] = {
                "mention_span": b.get("mention_span"),
                "mention_count_total": b.get("mention_count_total"),
                "user_follow_up": b.get("user_follow_up"),
            }
            for c in keep_cols:
                r[c] = row.get(c)
            rows.append(r)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out = out[out["mention_span"].notna()].copy()
    return out


def table_brand_mention_span_by_vertical(
    brand_df: pd.DataFrame,
    vertical_col: str = "vertical_tier1_llm",
    span_col: str = "mention_span",
) -> pd.DataFrame:
    """By vertical: count of brand-conversation pairs, mean/median mention_span."""
    if vertical_col not in brand_df.columns or span_col not in brand_df.columns:
        return pd.DataFrame()
    agg = brand_df.groupby(vertical_col)[span_col].agg(
        count="count",
        mean="mean",
        median="median",
    ).reset_index()
    return agg.round(2)


# ---------------------------------------------------------------------------
# Survival curves (empirical: S(t) = fraction with value >= t)
# ---------------------------------------------------------------------------

def _empirical_survival(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted unique values and S(t) = P(value >= t). Step curve."""
    v = values.dropna()
    if len(v) == 0:
        return np.array([0]), np.array([1.0])
    n = len(v)
    uniq = np.sort(v.unique())
    s = np.array([(v >= t).sum() / n for t in uniq])
    return uniq, s


def plot_conversation_length_survival_by_category(
    df: pd.DataFrame,
    n_messages_col: str = "n_messages",
    category_col: str = "category",
    figsize: tuple[float, float] = (8, 5),
    title: str = "Conversation length survival by category",
) -> "matplotlib.figure.Figure":
    """Survival curve S(t) = P(n_messages >= t) per category."""
    import matplotlib.pyplot as plt

    if n_messages_col not in df.columns or category_col not in df.columns:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title("No data")
        return fig

    fig, ax = plt.subplots(figsize=figsize)
    for cat in df[category_col].dropna().unique():
        sub = df[df[category_col] == cat][n_messages_col]
        t, s = _empirical_survival(sub)
        ax.step(t, s, where="post", label=str(cat))

    ax.set_xlabel("Message index t")
    ax.set_ylabel("P(conversation length ≥ t)")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    return fig


def plot_conversation_length_survival_by_vertical(
    df: pd.DataFrame,
    n_messages_col: str = "n_messages",
    category_col: str = "category",
    vertical_col: str = "vertical_tier1_llm",
    figsize: tuple[float, float] = (10, 6),
    max_curves_per_plot: int = 10,
) -> list["matplotlib.figure.Figure"]:
    """
    One figure per category: survival curves stratified by vertical_tier1_llm.
    Returns list of figures (one per category with multiple verticals).
    """
    import matplotlib.pyplot as plt

    if n_messages_col not in df.columns or category_col not in df.columns or vertical_col not in df.columns:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title("No data")
        return [fig]

    figures: list["matplotlib.figure.Figure"] = []
    for cat in df[category_col].dropna().unique():
        sub = df[df[category_col] == cat]
        verticals = sub[vertical_col].dropna().unique()
        if len(verticals) > max_curves_per_plot:
            verticals = list(verticals)[:max_curves_per_plot]
        if len(verticals) == 0:
            continue
        fig, ax = plt.subplots(figsize=figsize)
        for v in verticals:
            vals = sub[sub[vertical_col] == v][n_messages_col]
            t, s = _empirical_survival(vals)
            ax.step(t, s, where="post", label=str(v))
        ax.set_xlabel("Message index t")
        ax.set_ylabel("P(conversation length ≥ t)")
        ax.set_title(f"Conversation length survival by vertical — {cat}")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.set_ylim(-0.02, 1.02)
        plt.tight_layout()
        figures.append(fig)
    return figures if figures else [_create_empty_figure(figsize)]


def _create_empty_figure(figsize: tuple[float, float]) -> "matplotlib.figure.Figure":
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title("No data")
    return fig


def plot_deal_size_by_vertical(
    df: pd.DataFrame,
    deal_col: str = "deal_size_usd",
    vertical_col: str = "vertical_tier1_llm",
    total_brands_col: str = "total_brands",
    require_has_brand: bool = True,
    figsize: tuple[float, float] = (10, 5),
    use_log_scale: bool = True,
    title: str = "Deal size (USD) by business vertical",
) -> "matplotlib.figure.Figure":
    """Box plot of deal_size_usd by vertical. If require_has_brand, only conversations with at least one brand."""
    import matplotlib.pyplot as plt

    if deal_col not in df.columns or vertical_col not in df.columns:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title("No data")
        return fig

    valid = df[[vertical_col, deal_col]].copy()
    if total_brands_col in df.columns and require_has_brand:
        valid = valid.loc[df[total_brands_col].fillna(0) > 0]
    valid = valid.dropna()
    valid = valid[valid[deal_col] > 0]
    if valid.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)
        return fig

    fig, ax = plt.subplots(figsize=figsize)
    verts = valid[vertical_col].unique()
    data = [valid[valid[vertical_col] == v][deal_col].values for v in verts]
    ax.boxplot(data, labels=verts, patch_artist=True)
    if use_log_scale:
        ax.set_yscale("log")
    ax.set_ylabel("Deal size (USD)")
    ax.set_xlabel("Business vertical")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def plot_deal_size_by_vertical_per_category(
    df: pd.DataFrame,
    category_col: str = "category",
    deal_col: str = "deal_size_usd",
    vertical_col: str = "vertical_tier1_llm",
    total_brands_col: str = "total_brands",
    require_has_brand: bool = True,
    figsize: tuple[float, float] = (10, 5),
    use_log_scale: bool = True,
) -> list["matplotlib.figure.Figure"]:
    """One box plot of deal size by vertical per category. Returns list of figures."""
    import matplotlib.pyplot as plt

    if category_col not in df.columns or deal_col not in df.columns or vertical_col not in df.columns:
        return [_create_empty_figure(figsize)]

    figures: list["matplotlib.figure.Figure"] = []
    for cat in df[category_col].dropna().unique():
        sub = df[df[category_col] == cat]
        fig = plot_deal_size_by_vertical(
            sub,
            deal_col=deal_col,
            vertical_col=vertical_col,
            total_brands_col=total_brands_col,
            require_has_brand=require_has_brand,
            figsize=figsize,
            use_log_scale=use_log_scale,
            title=f"Deal size (USD) by vertical — {cat}",
        )
        figures.append(fig)
    return figures


def plot_brand_engagement_by_category(
    table: pd.DataFrame,
    category_col: str = "category",
    pct_col: str = "pct_with_brand",
    figsize: tuple[float, float] = (8, 5),
    title: str = "Percentage of conversations with brand mentioned, by category",
) -> "matplotlib.figure.Figure":
    """Bar chart comparing % of conversations with at least one brand across categories."""
    import matplotlib.pyplot as plt

    if category_col not in table.columns or pct_col not in table.columns:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title("No data")
        return fig

    fig, ax = plt.subplots(figsize=figsize)
    table = table.sort_values(pct_col, ascending=False)
    bars = ax.bar(table[category_col].astype(str), table[pct_col])
    ax.bar_label(bars, labels=[f"{x:.1f}%" for x in table[pct_col]], padding=4)
    ax.set_ylabel("% with brand mentioned")
    ax.set_xlabel("Category")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def plot_follow_up_pct_by_vertical_per_category(
    df: pd.DataFrame,
    category_col: str = "category",
    follow_up_col: str = "any_user_follow_up_on_brand",
    vertical_col: str = "vertical_tier1_llm",
    figsize: tuple[float, float] = (10, 5),
) -> list["matplotlib.figure.Figure"]:
    """One bar chart of follow-up % by vertical per category. Returns list of figures."""
    import matplotlib.pyplot as plt

    if category_col not in df.columns or follow_up_col not in df.columns or vertical_col not in df.columns:
        return [_create_empty_figure(figsize)]

    figures: list["matplotlib.figure.Figure"] = []
    for cat in df[category_col].dropna().unique():
        sub = df[df[category_col] == cat]
        tbl = table_follow_up_pct_by_vertical(sub, follow_up_col=follow_up_col, vertical_col=vertical_col)
        if tbl.empty:
            figures.append(_create_empty_figure(figsize))
            continue
        fig = plot_follow_up_pct_by_vertical(
            tbl,
            vertical_col=vertical_col,
            pct_col="pct",
            figsize=figsize,
            title=f"User follow-up on brand by vertical — {cat}",
        )
        figures.append(fig)
    return figures


def plot_brand_mention_survival_by_vertical(
    brand_df: pd.DataFrame,
    vertical_col: str = "vertical_tier1_llm",
    span_col: str = "mention_span",
    figsize: tuple[float, float] = (8, 5),
    title: str = "Brand mention span survival by vertical",
    max_curves: int = 15,
) -> "matplotlib.figure.Figure":
    """Survival curve S(t) = P(mention_span >= t) per vertical."""
    import matplotlib.pyplot as plt

    if vertical_col not in brand_df.columns or span_col not in brand_df.columns:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title("No data")
        return fig

    fig, ax = plt.subplots(figsize=figsize)
    verticals = brand_df[vertical_col].dropna().unique()
    if len(verticals) > max_curves:
        verticals = list(verticals)[:max_curves]
    for v in verticals:
        sub = brand_df[brand_df[vertical_col] == v][span_col]
        t, s = _empirical_survival(sub)
        ax.step(t, s, where="post", label=str(v))
    ax.set_xlabel("Round span t")
    ax.set_ylabel("P(mention span ≥ t)")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    return fig


def plot_follow_up_pct_by_vertical(
    table: pd.DataFrame,
    vertical_col: str = "vertical_tier1_llm",
    pct_col: str = "pct",
    figsize: tuple[float, float] = (10, 5),
    title: str = "User follow-up on brand by business vertical (%)",
) -> "matplotlib.figure.Figure":
    """Bar chart of follow-up percentage by vertical."""
    import matplotlib.pyplot as plt

    if vertical_col not in table.columns or pct_col not in table.columns:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title("No data")
        return fig

    fig, ax = plt.subplots(figsize=figsize)
    table = table.sort_values(pct_col, ascending=True)
    ax.barh(table[vertical_col].astype(str), table[pct_col])
    ax.set_xlabel("Follow-up %")
    ax.set_ylabel("Business vertical")
    ax.set_title(title)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Save figure helper
# ---------------------------------------------------------------------------

def save_figure(
    fig: "matplotlib.figure.Figure",
    plot_dir: str | Path,
    filename: str,
    dpi: int = 150,
) -> Path:
    """Save figure as PNG and close it. Creates plot_dir if needed."""
    from insights_utils import ensure_output_dir

    plot_path = Path(plot_dir)
    ensure_output_dir(plot_path)
    path = plot_path / filename
    if not filename.endswith(".png"):
        path = plot_path / f"{filename}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path
