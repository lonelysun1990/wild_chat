"""
Utilities for WildChat EDA. Used by explore.ipynb.
Sampling (conversation-level) and table builders for steps 3–9.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Sampling (conversation-level: whole conversations only)
# ---------------------------------------------------------------------------


def sample_by_conversation(
    dataset,
    n: Optional[int] = 5000,
    pct: Optional[float] = None,
    seed: int = 42,
):
    """
    Sample the dataset by conversation (row). Every selected row keeps its
    full `conversation` list—no conversation is split.

    Args:
        dataset: HuggingFace Dataset (train split).
        n: Exact number of conversations to sample; used if not None.
        pct: Fraction of dataset to sample (e.g. 0.01); used only when n is None.
        seed: Random seed for reproducibility.

    Returns:
        Sampled HuggingFace Dataset (same type as input).
    """
    size = len(dataset)
    if n is not None:
        k = min(n, size)
    elif pct is not None:
        k = max(1, math.ceil(size * pct))
        k = min(k, size)
    else:
        # Both None: use full dataset (no sampling)
        k = size

    if k >= size:
        return dataset
    rng = random.Random(seed)
    indices = rng.sample(range(size), k=k)
    return dataset.select(indices)


# ---------------------------------------------------------------------------
# Step 3: Basic counts and distributions
# ---------------------------------------------------------------------------


def build_step3_basic_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Model counts, turn stats, language counts (top N)."""
    rows = []

    # Total conversations
    rows.append({"metric": "total_conversations", "value": len(df)})

    # Model value counts
    for model, count in df["model"].value_counts().items():
        rows.append({"metric": f"model_{model}", "value": count})

    # Turn distribution
    turn = df["turn"]
    rows.append({"metric": "turn_min", "value": int(turn.min())})
    rows.append({"metric": "turn_max", "value": int(turn.max())})
    rows.append({"metric": "turn_mean", "value": float(turn.mean())})
    rows.append({"metric": "turn_median", "value": float(turn.median())})
    rows.append({"metric": "single_turn_count", "value": int((turn == 1).sum())})
    rows.append({"metric": "multi_turn_count", "value": int((turn > 1).sum())})

    # Language (top 20)
    lang_counts = df["language"].value_counts().head(20)
    for lang, count in lang_counts.items():
        rows.append({"metric": f"language_{lang}", "value": count})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 4: Conversation and message length
# ---------------------------------------------------------------------------


def _message_lengths_and_roles(df: pd.DataFrame):
    """Per-conversation: n_messages, total_words, user word counts per msg, assistant word counts per msg."""
    convos = df["conversation"]
    n_msgs = []
    total_words = []
    user_lens = []
    asst_lens = []

    for conv in convos:
        # Accept list or numpy array (e.g. from parquet)
        if not isinstance(conv, list):
            try:
                conv = list(conv)
            except TypeError:
                n_msgs.append(0)
                total_words.append(0)
                user_lens.append([])
                asst_lens.append([])
                continue
        u_lens = []
        a_lens = []
        tot = 0
        for msg in conv:
            c = (msg.get("content") or "") if isinstance(msg, dict) else ""
            word_count = len(str(c).split())
            tot += word_count
            role = msg.get("role") if isinstance(msg, dict) else ""
            if role == "user":
                u_lens.append(word_count)
            elif role == "assistant":
                a_lens.append(word_count)
        n_msgs.append(len(conv))
        total_words.append(tot)
        user_lens.append(u_lens)
        asst_lens.append(a_lens)

    return n_msgs, total_words, user_lens, asst_lens


def build_step4_message_length(df: pd.DataFrame) -> pd.DataFrame:
    """Per-conversation message counts and word counts; user vs assistant."""
    n_msgs, total_words, user_lens, asst_lens = _message_lengths_and_roles(df)

    rows = []
    for i in range(len(df)):
        u_mean = sum(user_lens[i]) / len(user_lens[i]) if user_lens[i] else 0
        a_mean = sum(asst_lens[i]) / len(asst_lens[i]) if asst_lens[i] else 0
        rows.append({
            "conversation_idx": i,
            "n_messages": n_msgs[i],
            "total_words": total_words[i],
            "user_mean_words": round(u_mean, 2),
            "user_total_words": sum(user_lens[i]),
            "assistant_mean_words": round(a_mean, 2),
            "assistant_total_words": sum(asst_lens[i]),
            "n_user_msgs": len(user_lens[i]),
            "n_assistant_msgs": len(asst_lens[i]),
        })

    return pd.DataFrame(rows)


def build_step4_message_length_summary(step4_df: pd.DataFrame) -> pd.DataFrame:
    """Summary stats for message word count (percentiles, etc.) for display."""
    all_user = []
    all_asst = []
    for _, row in step4_df.iterrows():
        if row["n_user_msgs"]:
            all_user.append(row["user_mean_words"])
        if row["n_assistant_msgs"]:
            all_asst.append(row["assistant_mean_words"])
    s = pd.Series({
        "user_mean_words_median": pd.Series(all_user).median() if all_user else 0,
        "user_mean_words_mean": pd.Series(all_user).mean() if all_user else 0,
        "assistant_mean_words_median": pd.Series(all_asst).median() if all_asst else 0,
        "assistant_mean_words_mean": pd.Series(all_asst).mean() if all_asst else 0,
    })
    return s.to_frame("value").T


# ---------------------------------------------------------------------------
# Step 5: Temporal distribution
# ---------------------------------------------------------------------------


def build_step5_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """Conversation count by date (e.g. by day or week)."""
    ts = pd.to_datetime(df["timestamp"], utc=True)
    daily = ts.dt.date.value_counts().sort_index().reset_index()
    daily.columns = ["date", "count"]
    return daily


# ---------------------------------------------------------------------------
# Step 6: Moderation and quality flags
# ---------------------------------------------------------------------------


def build_step6_moderation(df: pd.DataFrame) -> pd.DataFrame:
    """Toxic/redacted counts; optional per-conversation max detoxify scores."""
    rows = []

    # Toxic / redacted at conversation level
    rows.append({"metric": "toxic_true", "value": int(df["toxic"].sum())})
    rows.append({"metric": "toxic_false", "value": int((~df["toxic"]).sum())})
    rows.append({"metric": "redacted_true", "value": int(df["redacted"].sum())})
    rows.append({"metric": "redacted_false", "value": int((~df["redacted"]).sum())})

    # Per-conversation max detoxify toxicity (if present)
    try:
        max_tox = []
        for d in df["detoxify_moderation"]:
            if not isinstance(d, list):
                max_tox.append(0.0)
                continue
            vals = []
            for u in d:
                if isinstance(u, dict) and "toxicity" in u:
                    vals.append(float(u["toxicity"]))
            max_tox.append(max(vals) if vals else 0.0)
        if max_tox:
            rows.append({"metric": "detoxify_toxicity_max_mean", "value": sum(max_tox) / len(max_tox)})
    except Exception:
        pass

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 7: Empty and short user inputs
# ---------------------------------------------------------------------------


def build_step7_empty_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Conversations with at least one empty/whitespace user message; counts."""
    convos = df["conversation"]
    ids = df.get("conversation_id", pd.Series(range(len(df))))

    has_empty = []
    empty_conversation_ids = []

    for idx, conv in enumerate(convos):
        if not isinstance(conv, list):
            has_empty.append(False)
            continue
        found = False
        for msg in conv:
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            c = msg.get("content") or ""
            if str(c).strip() == "":
                found = True
                break
        has_empty.append(found)
        if found:
            empty_conversation_ids.append(ids.iloc[idx] if hasattr(ids, "iloc") else ids[idx])

    n_empty = sum(has_empty)
    rows = [
        {"metric": "conversations_with_empty_user_input", "value": n_empty},
        {"metric": "pct_with_empty_user_input", "value": 100.0 * n_empty / len(df) if len(df) else 0},
    ]
    for i, cid in enumerate(empty_conversation_ids[:20]):
        rows.append({"metric": f"empty_sample_id_{i+1}", "value": str(cid)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 8: Keyword counts (optional)
# ---------------------------------------------------------------------------


def build_step8_keyword_counts(df: pd.DataFrame, keywords: Optional[list[str]] = None) -> pd.DataFrame:
    """Rough topic signal: count conversations containing each keyword (in user messages)."""
    if keywords is None:
        keywords = ["code", "translate", "write", "python", "explain"]

    convos = df["conversation"]
    counts = {kw: 0 for kw in keywords}

    for conv in convos:
        if not isinstance(conv, list):
            continue
        for msg in conv:
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            c = (msg.get("content") or "").lower()
            for kw in keywords:
                if kw.lower() in c:
                    counts[kw] += 1
                    break

    return pd.DataFrame([
        {"keyword": k, "conversation_count": v} for k, v in counts.items()
    ])


# ---------------------------------------------------------------------------
# Step 9: Summary table
# ---------------------------------------------------------------------------


def build_step9_summary(
    df: pd.DataFrame,
    step3_df: pd.DataFrame,
    step4_df: pd.DataFrame,
    step5_df: pd.DataFrame,
    step6_df: pd.DataFrame,
    step7_df: pd.DataFrame,
) -> pd.DataFrame:
    """One-row (or small) summary: total convos, turns, date range, top langs, etc."""
    ts = pd.to_datetime(df["timestamp"], utc=True)
    date_min = ts.min()
    date_max = ts.max()

    # Top 3 languages
    top3 = df["language"].value_counts().head(3)
    top3_str = ", ".join([f"{lang}({c})" for lang, c in top3.items()])

    # From step3
    total_conv = len(df)
    turn_sum = int(df["turn"].sum())

    # From step4 (use step4_df if provided)
    if step4_df is not None and len(step4_df):
        med_user = step4_df["user_mean_words"].median()
        mean_user = step4_df["user_mean_words"].mean()
        med_asst = step4_df["assistant_mean_words"].median()
        mean_asst = step4_df["assistant_mean_words"].mean()
    else:
        med_user = mean_user = med_asst = mean_asst = 0.0

    # From step6/7
    pct_redacted = 100.0 * df["redacted"].sum() / len(df) if len(df) else 0
    pct_empty = 0.0
    if step7_df is not None and len(step7_df):
        row = step7_df[step7_df["metric"] == "pct_with_empty_user_input"]
        if not row.empty:
            try:
                pct_empty = float(row["value"].iloc[0])
            except (TypeError, ValueError):
                pct_empty = 0.0

    model_counts = df["model"].value_counts()
    model_str = ", ".join([f"{m}({c})" for m, c in model_counts.items()])

    summary = pd.DataFrame([{
        "total_conversations": total_conv,
        "total_turns": turn_sum,
        "date_min": date_min,
        "date_max": date_max,
        "top3_languages": top3_str,
        "model_counts": model_str,
        "pct_redacted": pct_redacted,
        "pct_empty_user_input": pct_empty,
        "median_user_msg_len": med_user,
        "mean_user_msg_len": mean_user,
        "median_assistant_msg_len": med_asst,
        "mean_assistant_msg_len": mean_asst,
    }])
    return summary


# ---------------------------------------------------------------------------
# Load/save helpers (for notebook use)
# ---------------------------------------------------------------------------


def ensure_output_dir(path: str | Path) -> Path:
    """Create output directory if it doesn't exist. Do not use data/."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_or_build(
    out_path: str | Path,
    load_from_saved: bool,
    build_fn,
    *args,
    **kwargs,
) -> pd.DataFrame:
    """If load_from_saved and file exists, load parquet; else run build_fn(*args, **kwargs) and save."""
    out_path = Path(out_path)
    if load_from_saved and out_path.exists():
        return pd.read_parquet(out_path)
    df = build_fn(*args, **kwargs)
    ensure_output_dir(out_path.parent)
    df.to_parquet(out_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Export English-only conversations (full dataset, chunked parquet)
# ---------------------------------------------------------------------------


def export_english_chunked_parquet(
    dataset,
    output_dir: str | Path,
    chunk_size: int = 50_000,
    language_col: str = "language",
    language_value: str = "English",
) -> tuple[Path, int]:
    """
    Filter to English conversations only and write to chunked parquet files.
    Uses the full dataset (no sampling). Caller must pass the full train split.

    Args:
        dataset: HuggingFace Dataset (train split, full).
        output_dir: Directory to write parquet chunks (e.g. data/english_chunks).
        chunk_size: Max rows per parquet file.
        language_col: Column name for language.
        language_value: Value to keep (e.g. "English").

    Returns:
        (output_dir Path, total rows written).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = dataset.to_pandas()
    english = df[df[language_col].astype(str).str.strip().eq(language_value)]
    total = len(english)
    if total == 0:
        return output_dir, 0
    written = 0
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk = english.iloc[start:end]
        path = output_dir / f"english_{start:06d}_{end:06d}.parquet"
        chunk.to_parquet(path, index=False)
        written += len(chunk)
    return output_dir, written


def _parse_conversation_cell(val: Any) -> Any:
    """If val is a string (e.g. JSON from parquet round-trip), parse to list of dicts; else return as-is."""
    if val is None or (isinstance(val, list) and (not val or isinstance(val[0], dict))):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            import ast
            try:
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                return val
    return val


def ensure_conversation_parsed(df: pd.DataFrame, conversation_col: str = "conversation") -> pd.DataFrame:
    """
    If the conversation column was stored as strings (e.g. after parquet round-trip),
    parse each cell to list of dicts so extract_text_column / get_first_user_message work.
    """
    if conversation_col not in df.columns:
        return df
    out = df.copy()
    first = out[conversation_col].dropna().iloc[0] if not out[conversation_col].isna().all() else None
    if first is not None and isinstance(first, str):
        out[conversation_col] = out[conversation_col].map(_parse_conversation_cell)
    return out


def load_english_chunked_parquet(output_dir: str | Path) -> pd.DataFrame:
    """
    Load all English chunked parquet files from output_dir and concatenate.
    Expects files named english_*.parquet (from export_english_chunked_parquet).
    Parses the conversation column from string to list-of-dicts when needed so that
    first-user-message extraction (e.g. in insights_analysis) works correctly.

    Returns:
        DataFrame with all English conversations. Empty DataFrame if no files found.
    """
    output_dir = Path(output_dir)
    paths = sorted(output_dir.glob("english_*.parquet"))
    if not paths:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    return ensure_conversation_parsed(df)
