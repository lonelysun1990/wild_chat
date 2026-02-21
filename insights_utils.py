"""
Utilities for WildChat in-depth insights analysis. Used by insights_analysis.ipynb.
Text extraction, topic modeling (LDA/NMF), and OpenAI-backed labeling with caching.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Text extraction from conversation
# ---------------------------------------------------------------------------


def get_first_user_message(conv: list) -> str:
    """Extract the first user message content from a conversation list (or numpy array)."""
    if not isinstance(conv, list):
        try:
            conv = list(conv)
        except TypeError:
            return ""
    for msg in conv:
        if isinstance(msg, dict) and msg.get("role") == "user":
            return (msg.get("content") or "").strip()
    return ""


def get_all_user_content(conv: list, sep: str = " ") -> str:
    """Concatenate all user message contents in order (conv may be list or numpy array)."""
    if not isinstance(conv, list):
        try:
            conv = list(conv)
        except TypeError:
            return ""
    parts = []
    for msg in conv:
        if isinstance(msg, dict) and msg.get("role") == "user":
            c = (msg.get("content") or "").strip()
            if c:
                parts.append(c)
    return sep.join(parts)


def get_all_assistant_content(conv: list, sep: str = " ") -> str:
    """Concatenate all assistant message contents in order (conv may be list or numpy array)."""
    if not isinstance(conv, list):
        try:
            conv = list(conv)
        except TypeError:
            return ""
    parts = []
    for msg in conv:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            c = (msg.get("content") or "").strip()
            if c:
                parts.append(c)
    return sep.join(parts)


def extract_text_column(
    df: pd.DataFrame,
    mode: str = "first_user",
    conversation_col: str = "conversation",
) -> pd.Series:
    """
    Add a column of text per conversation for classification or topic modeling.

    Args:
        df: DataFrame with a list column of messages per row.
        mode: "first_user" (first user message only) or "all_user" (all user messages concatenated).
        conversation_col: Name of the conversation column.

    Returns:
        Series of strings, same index as df.
    """
    convos = df[conversation_col]
    if mode == "first_user":
        return convos.map(get_first_user_message)
    if mode == "all_user":
        return convos.map(get_all_user_content)
    raise ValueError(f"mode must be 'first_user' or 'all_user', got {mode!r}")


# ---------------------------------------------------------------------------
# Commercial intent (keyword heuristic, no API)
# ---------------------------------------------------------------------------

COMMERCIAL_KEYWORDS = [
    "best", "recommend", "recommendation", "compare", "comparison",
    "buy", "purchase", "price", "cost", "review", "reviews",
    "which one", "should i get", "top 10", "top 5", "cheapest",
    "alternative", "vs ", " versus ", "worth it",
]


def has_commercial_intent_heuristic(text: str) -> bool:
    """Flag text that suggests product-seeking or commercial intent (no brand required)."""
    if not text or not isinstance(text, str):
        return False
    lower = text.lower()
    return any(kw in lower for kw in COMMERCIAL_KEYWORDS)


def add_commercial_intent_heuristic(
    df: pd.DataFrame,
    text_col: str = "first_user_text",
) -> pd.DataFrame:
    """Add column has_commercial_intent (bool) using keyword heuristic."""
    out = df.copy()
    out["has_commercial_intent"] = out[text_col].fillna("").map(has_commercial_intent_heuristic)
    return out


# ---------------------------------------------------------------------------
# Output dir and load/save
# ---------------------------------------------------------------------------


def ensure_output_dir(path: str | Path) -> Path:
    """Create output directory if it doesn't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_or_build(
    out_path: str | Path,
    load_from_saved: bool,
    build_fn: Callable[..., pd.DataFrame],
    *args: Any,
    **kwargs: Any,
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
# Topic modeling (TF-IDF + NMF / LDA)
# ---------------------------------------------------------------------------


def _tfidf_matrix(texts: pd.Series, max_df: float = 0.95, min_df: int = 1):
    """Build TF-IDF matrix. Returns (matrix, vectorizer). Fits on non-empty docs only so vocabulary is real; empty docs get zero vector."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    clean = texts.fillna("").astype(str).str.strip()
    non_empty = clean != ""
    # Fit only on non-empty documents so we get a real vocabulary (not just "_empty_")
    docs_to_fit = clean[non_empty]
    if len(docs_to_fit) == 0:
        docs_to_fit = pd.Series(["placeholder so vocabulary is non-empty"])
    vectorizer = TfidfVectorizer(
        max_df=1.0,
        min_df=min_df,
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 1),
        token_pattern=r"(?u)\b\w+\b",
    )
    try:
        vectorizer.fit(docs_to_fit)
    except ValueError:
        vectorizer = TfidfVectorizer(
            max_df=1.0,
            min_df=1,
            stop_words=None,
            max_features=5000,
            ngram_range=(1, 1),
            token_pattern=r"(?u)\b\w+\b",
        )
        vectorizer.fit(docs_to_fit)
    # Transform all documents (empty ones get zero vector)
    X = vectorizer.transform(clean.replace("", " "))
    return X, vectorizer


def run_nmf(
    texts: pd.Series,
    n_topics: int = 15,
    max_df: float = 0.95,
    min_df: int = 1,
    random_state: int = 42,
):
    """
    Run NMF on text series. Returns (model, doc_topics, vectorizer).
    doc_topics: array of shape (n_samples, n_topics); each row sums to 1 (normalized).
    """
    import numpy as np
    from sklearn.decomposition import NMF
    from sklearn.preprocessing import normalize

    X, vectorizer = _tfidf_matrix(texts, max_df=max_df, min_df=min_df)
    # NMF requires at least one positive entry; if X is all zeros (e.g. all empty docs), add one
    if hasattr(X, "toarray"):
        x_max = X.max()
    else:
        x_max = np.max(X)
    if x_max <= 0:
        X = X.copy()
        if hasattr(X, "tolil"):
            X = X.tolil()
            X[0, 0] = 1.0
            X = X.tocsr()
        else:
            X[0, 0] = 1.0
    n_components = min(n_topics, X.shape[1], max(1, X.shape[0]))
    nmf = NMF(n_components=n_components, random_state=random_state).fit(X)
    W = nmf.transform(X)
    # Rows that are all zero get NaN after normalize; give them uniform topic distribution
    row_sums = np.array(W.sum(axis=1)).flatten()
    zero_rows = row_sums <= 0
    if zero_rows.any():
        W = np.asarray(W)
        W[zero_rows, :] = 1.0 / W.shape[1]
    W_norm = normalize(W, norm="l1", axis=1)
    return nmf, W_norm, vectorizer


def run_lda(
    texts: pd.Series,
    n_topics: int = 15,
    max_df: float = 0.95,
    min_df: int = 1,
    random_state: int = 42,
):
    """
    Run LDA on text series. Returns (model, doc_topics, vectorizer).
    doc_topics: array of shape (n_samples, n_topics); each row is distribution over topics.
    """
    from sklearn.decomposition import LatentDirichletAllocation

    X, vectorizer = _tfidf_matrix(texts, max_df=max_df, min_df=min_df)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_state,
        max_iter=20,
    ).fit(X)
    doc_topics = lda.transform(X)
    return lda, doc_topics, vectorizer


def get_top_terms_nmf(model, vectorizer, n: int = 10) -> dict[int, list[str]]:
    """For each topic, return top n terms by weight (NMF components)."""
    feature_names = vectorizer.get_feature_names_out()
    if hasattr(feature_names, "tolist"):
        feature_names = feature_names.tolist()
    n_feat = len(feature_names)
    out = {}
    for i in range(model.components_.shape[0]):
        top_idx = model.components_[i].argsort()[-min(n, n_feat):][::-1]
        out[i] = [str(feature_names[j]) for j in top_idx if j < n_feat and feature_names[j]]
    return out


def get_top_terms_lda(model, vectorizer, n: int = 10) -> dict[int, list[str]]:
    """For each topic, return top n terms by weight (LDA components)."""
    feature_names = vectorizer.get_feature_names_out()
    if hasattr(feature_names, "tolist"):
        feature_names = feature_names.tolist()
    n_feat = len(feature_names)
    out = {}
    for i in range(model.components_.shape[0]):
        top_idx = model.components_[i].argsort()[-min(n, n_feat):][::-1]
        out[i] = [str(feature_names[j]) for j in top_idx if j < n_feat and feature_names[j]]
    return out


def assign_dominant_topic(doc_topics) -> pd.Series:
    """For each document, return the topic index with highest weight."""
    import numpy as np
    return pd.Series(np.argmax(doc_topics, axis=1))


def build_topic_distribution_df(doc_topics: Any) -> pd.DataFrame:
    """Count and share of conversations per topic."""
    import numpy as np
    dominant = np.argmax(doc_topics, axis=1)
    counts = pd.Series(dominant).value_counts().sort_index()
    total = len(dominant)
    return pd.DataFrame({
        "topic_id": counts.index,
        "count": counts.values,
        "pct": 100.0 * counts.values / total,
    })


# ---------------------------------------------------------------------------
# Theme/purpose over time and underserved
# ---------------------------------------------------------------------------


def theme_counts_over_time(
    df: pd.DataFrame,
    label_col: str,
    timestamp_col: str = "timestamp",
    freq: str = "W",
) -> pd.DataFrame:
    """Count conversations per label per time period (e.g. weekly)."""
    out = df[[timestamp_col, label_col]].copy()
    out["period"] = pd.to_datetime(out[timestamp_col], utc=True).dt.to_period(freq).astype(str)
    return out.groupby(["period", label_col]).size().reset_index(name="count")


def underserved_metrics(
    df: pd.DataFrame,
    label_col: str,
    turn_col: str = "turn",
) -> pd.DataFrame:
    """
    Per-label: share of conversations and share of total turns.
    Low share of conversations but similar share of turns => higher engagement per convo.
    """
    total_conv = len(df)
    total_turns = df[turn_col].sum()
    g = df.groupby(label_col).agg(
        conv_count=(label_col, "count"),
        turn_sum=(turn_col, "sum"),
    ).reset_index()
    g["conv_share_pct"] = 100.0 * g["conv_count"] / total_conv
    g["turn_share_pct"] = 100.0 * g["turn_sum"] / total_turns
    return g


# ---------------------------------------------------------------------------
# OpenAI API helpers (with caching)
# ---------------------------------------------------------------------------

# Default taxonomies for labeling
PURPOSE_OPTIONS = (
    "informational, educational, creative writing, coding, commercial/product, "
    "support, casual, other"
)
THEME_OPTIONS = (
    "technology & software, health & fitness, finance & money, education & learning, "
    "creative arts & writing, shopping & products, travel & places, food & cooking, "
    "career & work, science & research, entertainment, news & current events, "
    "personal advice, general knowledge, other"
)

# Max chars to send to API (avoid token limits)
MAX_TEXT_LEN = 4000


def _truncate(text: str, max_len: int = MAX_TEXT_LEN) -> str:
    if not text or len(text) <= max_len:
        return text or ""
    return text[: max_len - 3].rstrip() + "..."


def _get_client():
    """Lazy import and return OpenAI client (requires OPENAI_API_KEY in env)."""
    from openai import OpenAI
    return OpenAI()


def _call_chat(
    client,
    user_content: str,
    system_content: str,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> str:
    """Single chat completion with retries."""
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=500,
                temperature=0,
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(retry_delay * (attempt + 1))
    return ""


def load_label_cache(cache_path: str | Path) -> pd.DataFrame | None:
    """Load existing cache parquet if it exists. Expected: conversation_id + label column(s)."""
    p = Path(cache_path)
    if not p.exists():
        return None
    cache_df = pd.read_parquet(p)
    if "brands" in cache_df.columns and cache_df["brands"].dtype == object:
        def _parse_brands(x):
            if isinstance(x, list):
                return x
            if isinstance(x, str) and x.strip().startswith("["):
                try:
                    return json.loads(x)
                except Exception:
                    return []
            return []
        cache_df["brands"] = cache_df["brands"].map(_parse_brands)
    return cache_df


def merge_cached_labels(
    df: pd.DataFrame,
    cache_df: pd.DataFrame | None,
    id_col: str = "conversation_id",
    label_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Index]:
    """
    Merge cached labels into df. Return (df_with_cached, index_of_missing).
    label_cols: columns to take from cache; if None, use all except id_col.
    """
    if cache_df is None or len(cache_df) == 0:
        return df.copy(), df.index
    if label_cols is None:
        label_cols = [c for c in cache_df.columns if c != id_col]
    if not label_cols:
        return df.copy(), df.index
    df = df.copy()
    merged = df[[id_col]].merge(
        cache_df[[id_col] + label_cols],
        on=id_col,
        how="left",
        suffixes=("", "_cache"),
    )
    for c in label_cols:
        df[c] = merged[c].fillna(df[c]) if c in df.columns else merged[c]
    missing = df[label_cols[0]].isna() if label_cols else pd.Series(True, index=df.index)
    return df, df.index[missing]


def save_label_cache(
    cache_path: str | Path,
    ids: Any,
    label_data: dict[str, Any],
    id_col: str = "conversation_id",
) -> None:
    """Append or overwrite cache with new label_data. label_data: dict of column -> list/Series."""
    p = Path(cache_path)
    ensure_output_dir(p.parent)
    rows = []
    n = len(ids) if hasattr(ids, "__len__") else len(next(iter(label_data.values())))
    for i in range(n):
        row = {id_col: ids.iloc[i] if hasattr(ids, "iloc") else ids[i]}
        for col, vals in label_data.items():
            v = vals.iloc[i] if hasattr(vals, "iloc") else vals[i]
            if isinstance(v, list):
                v = json.dumps(v)  # store list as JSON string
            row[col] = v
        rows.append(row)
    new_df = pd.DataFrame(rows)
    if p.exists():
        existing = pd.read_parquet(p)
        combined = pd.concat([existing, new_df], ignore_index=True).drop_duplicates(
            subset=[id_col], keep="last"
        )
        combined.to_parquet(p, index=False)
    else:
        new_df.to_parquet(p, index=False)


def label_purpose_openai(
    df: pd.DataFrame,
    text_col: str = "first_user_text",
    id_col: str = "conversation_id",
    cache_path: str | Path = "insights_output/purpose_labels.parquet",
    use_cache: bool = True,
    batch_size: int = 100,
) -> pd.DataFrame:
    """
    Add column 'purpose' via OpenAI zero-shot. Uses cache to skip already-labeled rows.
    """
    df = df.copy()
    cache_df = load_label_cache(cache_path) if use_cache else None
    df, missing_idx = merge_cached_labels(
        df, cache_df, id_col=id_col, label_cols=["purpose"]
    )
    to_label = df.loc[missing_idx]
    if len(to_label) == 0:
        return df

    client = _get_client()
    purposes = []
    ids = to_label[id_col].tolist()
    texts = to_label[text_col].fillna("").astype(str).tolist()

    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch_ids = ids[start:end]
        batch_texts = texts[start:end]
        new_labels = []
        for i, text in enumerate(batch_texts):
            truncated = _truncate(text)
            prompt = f"Select exactly one primary purpose for this user message. Reply with only the label, nothing else.\n\nOptions: {PURPOSE_OPTIONS}\n\nUser message:\n{truncated}"
            try:
                label = _call_chat(
                    client,
                    prompt,
                    "You are a classifier. Reply with only one of the given options.",
                )
                # normalize to one of the options (take first matching word if messy)
                for opt in PURPOSE_OPTIONS.replace(",", " ").split():
                    opt = opt.strip()
                    if opt and opt.lower() in (label.lower() or ""):
                        label = opt
                        break
            except Exception:
                label = "other"
            new_labels.append(label or "other")
        save_label_cache(cache_path, to_label[id_col].iloc[start:end], {"purpose": new_labels}, id_col=id_col)
        purposes.extend(new_labels)
        time.sleep(0.5)  # rate limit

    df.loc[missing_idx, "purpose"] = purposes
    return df


def extract_brands_openai(
    df: pd.DataFrame,
    text_col: str = "first_user_text",
    id_col: str = "conversation_id",
    cache_path: str | Path = "insights_output/brands.parquet",
    use_cache: bool = True,
    batch_size: int = 100,
) -> pd.DataFrame:
    """
    Add column 'brands' (list of strings) via OpenAI. Uses cache to skip already-labeled rows.
    """
    df = df.copy()
    cache_df = load_label_cache(cache_path) if use_cache else None
    df, missing_idx = merge_cached_labels(
        df, cache_df, id_col=id_col, label_cols=["brands"]
    )
    to_label = df.loc[missing_idx]
    if len(to_label) == 0:
        return df

    client = _get_client()
    ids = to_label[id_col].tolist()
    texts = to_label[text_col].fillna("").astype(str).tolist()

    all_brands: list[list[str]] = []
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]
        new_brands = []
        for _i, text in enumerate(batch_texts):
            truncated = _truncate(text)
            prompt = f"List all brand, product, or company names mentioned in the following text. Reply with a JSON array of strings only, e.g. [\"Apple\", \"Nike\"]. If none, reply []\n\nText:\n{truncated}"
            try:
                raw = _call_chat(
                    client,
                    prompt,
                    "You output only a JSON array of strings. No other text.",
                )
                raw = raw.strip()
                if raw.startswith("["):
                    arr = json.loads(raw)
                else:
                    # try to find [...] in response
                    match = re.search(r"\[.*\]", raw, re.DOTALL)
                    arr = json.loads(match.group(0)) if match else []
                if not isinstance(arr, list):
                    arr = []
                arr = [str(x).strip() for x in arr if x]
            except Exception:
                arr = []
            new_brands.append(arr)
        all_brands.extend(new_brands)
        save_label_cache(
            cache_path,
            to_label[id_col].iloc[start:end],
            {"brands": new_brands},
            id_col=id_col,
        )
        time.sleep(0.5)
    # Assign as Series so pandas doesn't treat list-of-lists as 2D array
    df.loc[missing_idx, "brands"] = pd.Series(all_brands, index=missing_idx)
    return df


def label_theme_openai(
    df: pd.DataFrame,
    text_col: str = "first_user_text",
    id_col: str = "conversation_id",
    cache_path: str | Path = "insights_output/theme_labels.parquet",
    use_cache: bool = True,
    batch_size: int = 100,
) -> pd.DataFrame:
    """
    Add column 'theme' via OpenAI zero-shot. Uses cache to skip already-labeled rows.
    """
    df = df.copy()
    cache_df = load_label_cache(cache_path) if use_cache else None
    df, missing_idx = merge_cached_labels(
        df, cache_df, id_col=id_col, label_cols=["theme"]
    )
    to_label = df.loc[missing_idx]
    if len(to_label) == 0:
        return df

    client = _get_client()
    themes = []
    texts = to_label[text_col].fillna("").astype(str).tolist()

    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]
        new_labels = []
        for text in batch_texts:
            truncated = _truncate(text)
            prompt = f"Select exactly one primary theme for this user message. Reply with only the label, nothing else.\n\nOptions: {THEME_OPTIONS}\n\nUser message:\n{truncated}"
            try:
                label = _call_chat(
                    client,
                    prompt,
                    "You are a classifier. Reply with only one of the given options.",
                )
                for opt in THEME_OPTIONS.replace(",", " ").split():
                    opt = opt.strip()
                    if len(opt) > 2 and opt.lower() in (label.lower() or ""):
                        label = opt
                        break
            except Exception:
                label = "other"
            new_labels.append(label or "other")
        save_label_cache(
            cache_path,
            to_label[id_col].iloc[start:end],
            {"theme": new_labels},
            id_col=id_col,
        )
        themes.extend(new_labels)
        time.sleep(0.5)
    df.loc[missing_idx, "theme"] = themes
    return df


def _label_one_theme(client: Any, text: str) -> str:
    """Label a single text for theme (used by label_theme_openai_parallel)."""
    truncated = _truncate(text)
    prompt = f"Select exactly one primary theme for this user message. Reply with only the label, nothing else.\n\nOptions: {THEME_OPTIONS}\n\nUser message:\n{truncated}"
    try:
        label = _call_chat(
            client,
            prompt,
            "You are a classifier. Reply with only one of the given options.",
        )
        for opt in THEME_OPTIONS.replace(",", " ").split():
            opt = opt.strip()
            if len(opt) > 2 and opt.lower() in (label.lower() or ""):
                label = opt
                break
    except Exception:
        label = "other"
    return label or "other"


def label_theme_openai_parallel(
    df: pd.DataFrame,
    text_col: str = "first_user_text",
    id_col: str = "conversation_id",
    cache_path: str | Path = "insights_output/theme_labels.parquet",
    use_cache: bool = True,
    batch_size: int = 100,
    max_workers: int = 10,
) -> pd.DataFrame:
    """
    Add column 'theme' via OpenAI zero-shot with parallel API calls (faster).
    Uses cache to skip already-labeled rows. max_workers limits concurrent requests.
    """
    df = df.copy()
    cache_df = load_label_cache(cache_path) if use_cache else None
    df, missing_idx = merge_cached_labels(
        df, cache_df, id_col=id_col, label_cols=["theme"]
    )
    to_label = df.loc[missing_idx]
    if len(to_label) == 0:
        return df

    client = _get_client()
    texts = to_label[text_col].fillna("").astype(str).tolist()
    themes: list[str] = []

    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            new_labels = list(ex.map(lambda t: _label_one_theme(client, t), batch_texts))
        save_label_cache(
            cache_path,
            to_label[id_col].iloc[start:end],
            {"theme": new_labels},
            id_col=id_col,
        )
        themes.extend(new_labels)
        time.sleep(0.2)  # short pause between batches to ease rate limits

    df.loc[missing_idx, "theme"] = themes
    return df


def label_commercial_intent_openai(
    df: pd.DataFrame,
    text_col: str = "first_user_text",
    id_col: str = "conversation_id",
    cache_path: str | Path = "insights_output/commercial_intent.parquet",
    use_cache: bool = True,
    batch_size: int = 100,
) -> pd.DataFrame:
    """
    Add column 'commercial_intent_llm' (Yes/No) via OpenAI. Uses cache.
    """
    df = df.copy()
    cache_df = load_label_cache(cache_path) if use_cache else None
    df, missing_idx = merge_cached_labels(
        df, cache_df, id_col=id_col, label_cols=["commercial_intent_llm"]
    )
    to_label = df.loc[missing_idx]
    if len(to_label) == 0:
        return df

    client = _get_client()
    labels = []
    texts = to_label[text_col].fillna("").astype(str).tolist()

    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]
        new_labels = []
        for text in batch_texts:
            truncated = _truncate(text)
            prompt = f"Does this message have commercial or product-seeking intent (e.g. comparing products, asking for recommendations, buying), even if no brand is named? Reply only: Yes or No\n\nMessage:\n{truncated}"
            try:
                raw = _call_chat(
                    client,
                    prompt,
                    "Reply only with Yes or No.",
                )
                raw = (raw or "").strip().lower()
                new_labels.append("Yes" if raw.startswith("yes") else "No")
            except Exception:
                new_labels.append("No")
        save_label_cache(
            cache_path,
            to_label[id_col].iloc[start:end],
            {"commercial_intent_llm": new_labels},
            id_col=id_col,
        )
        labels.extend(new_labels)
        time.sleep(0.5)
    df.loc[missing_idx, "commercial_intent_llm"] = labels
    return df
