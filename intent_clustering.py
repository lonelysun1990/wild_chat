"""
Fast intent clustering for WildChat. Used by intent_clustering.ipynb.
Data prep, rule-based intent, embedding + K-means, optional LLM summarization per cluster.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------


def prepare_sample(
    df: pd.DataFrame,
    n: int = 1000,
    text_mode: str = "first_user",
    seed: int = 42,
    conversation_col: str = "conversation",
) -> pd.DataFrame:
    """
    Sample n rows, add text from conversation (first or all user messages), drop empty text.
    Returns dataframe with 'text' column; index preserved from sampled df.
    """
    from insights_utils import extract_text_column

    if len(df) == 0:
        return df.copy()
    n = min(n, len(df))
    rng = random.Random(seed)
    indices = rng.sample(range(len(df)), k=n)
    out = df.iloc[indices].copy()
    out["text"] = extract_text_column(
        out, mode=text_mode, conversation_col=conversation_col
    )
    # Drop rows where text is empty or whitespace
    out = out[out["text"].fillna("").astype(str).str.strip() != ""].copy()
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Rule-based intent (coarse categories, no API)
# ---------------------------------------------------------------------------

# Intent label -> list of keywords (first match wins). Commercial is first so it takes priority when a message matches both commercial and another category.
INTENT_RULES: list[tuple[str, list[str]]] = [
    ("commercial_product", ["best", "recommend", "recommendation", "compare", "comparison", "buy", "purchase", "price", "cost", "review", "reviews", "which one", "should i get", "top 10", "top 5", "cheapest", "alternative", " vs ", " versus ", "worth it"]),
    ("coding", ["code", "function", "script", "python", "javascript", "programming", "debug", "api ", "sql", "regex", "algorithm", "implement", "bug", "error in my code"]),
    ("creative_writing", ["write a", "write me", "story", "poem", "essay", "dialogue", "character", "plot", "fiction", "creative", "song lyrics", "script for"]),
    ("education", ["explain", "how does", "what is", "why does", "learn", "teach", "lesson", "homework", "assignment", "study", "definition of", "meaning of"]),
    ("support", ["help me", "fix my", "not working", "broken", "issue", "problem with", "error", "support", "troubleshoot", "how do i fix", "why is my"]),
    ("casual_other", []),  # fallback
]

DEFAULT_INTENT_LABEL = "casual_other"


def assign_intent_rule_based(
    text_series: pd.Series,
    intent_rules: Optional[list[tuple[str, list[str]]]] = None,
    default_label: str = DEFAULT_INTENT_LABEL,
) -> pd.Series:
    """
    For each text, check keyword rules in priority order; assign first matching intent.
    If no rule matches, assign default_label (e.g. casual_other).
    """
    intent_rules = intent_rules or INTENT_RULES
    labels = []
    for _, text in text_series.items():
        t = (text or "").lower().strip()
        if not t:
            labels.append(default_label)
            continue
        assigned = False
        for label, keywords in intent_rules:
            if label == default_label and not keywords:
                continue
            if any(kw in t for kw in keywords):
                labels.append(label)
                assigned = True
                break
        if not assigned:
            labels.append(default_label)
    return pd.Series(labels, index=text_series.index)


# ---------------------------------------------------------------------------
# Embedding-based clustering (sentence-transformers + sklearn)
# ---------------------------------------------------------------------------


def embed_texts(
    text_series: pd.Series,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    show_progress: bool = False,
) -> np.ndarray:
    """Embed texts with sentence-transformers. Returns array of shape (n, dim)."""
    from sentence_transformers import SentenceTransformer

    texts = text_series.fillna("").astype(str).tolist()
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=show_progress
    )
    return np.asarray(embeddings, dtype=np.float64)


def cluster_embeddings(
    embeddings: np.ndarray,
    method: str = "kmeans",
    n_clusters: int = 8,
    random_state: int = 42,
) -> np.ndarray:
    """
    Cluster embeddings. Returns 1d array of labels (0..K-1 for kmeans; -1 possible for HDBSCAN).
    """
    if method == "kmeans":
        from sklearn.cluster import KMeans
        n_clusters = min(n_clusters, max(1, len(embeddings)))
        labels = KMeans(
            n_clusters=n_clusters, random_state=random_state, n_init=10
        ).fit_predict(embeddings)
        return np.asarray(labels, dtype=np.int32)
    if method == "hdbscan":
        try:
            from sklearn.cluster import HDBSCAN
            clusterer = HDBSCAN(min_cluster_size=max(2, len(embeddings) // 50))
            return np.asarray(clusterer.fit_predict(embeddings), dtype=np.int32)
        except Exception:
            from sklearn.cluster import KMeans
            n_clusters = min(n_clusters, max(1, len(embeddings)))
            return np.asarray(
                KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit_predict(embeddings),
                dtype=np.int32,
            )
    raise ValueError(f"method must be 'kmeans' or 'hdbscan', got {method!r}")


# ---------------------------------------------------------------------------
# Sub-clustering per intent (embedding + K-means within each category)
# ---------------------------------------------------------------------------

def add_subclusters_per_intent(
    df: pd.DataFrame,
    intent_col: str = "intent_rule",
    text_col: str = "text",
    total_count: Optional[int] = None,
    target_pct: float = 0.05,
    random_state: int = 42,
    model_name: str = "all-MiniLM-L6-v2",
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    For each intent category, run embedding + K-means so each sub-cluster is ~target_pct of total.
    Adds column sub_cluster_id (local to each intent: 0, 1, 2, ...).
    """
    out = df.copy()
    total = total_count if total_count is not None else len(out)
    target_size = max(1, int(total * target_pct))
    out["sub_cluster_id"] = -1

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    for intent in out[intent_col].dropna().unique():
        mask = out[intent_col] == intent
        subset_idx = out.index[mask].tolist()
        n_in = len(subset_idx)
        n_sub = max(1, min(n_in, n_in // target_size))
        if n_sub <= 1:
            out.loc[mask, "sub_cluster_id"] = 0
            continue
        texts = out.loc[mask, text_col].fillna("").astype(str).tolist()
        emb = model.encode(texts, batch_size=64, show_progress_bar=show_progress)
        emb = np.asarray(emb, dtype=np.float64)
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=n_sub, random_state=random_state, n_init=10).fit_predict(emb)
        out.loc[mask, "sub_cluster_id"] = labels
    return out


# ---------------------------------------------------------------------------
# LLM summarization (once per cluster)
# ---------------------------------------------------------------------------

MAX_SUMMARY_TEXT_LEN = 300


def summarize_cluster_with_llm(
    texts_in_cluster: list[str],
    client: Any,
    n_sample: int = 15,
    max_tokens: int = 150,
    seed: Optional[int] = 42,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Take a random sample of texts, send one prompt asking for common theme/intent in 1-2 sentences.
    Returns the summary string.
    """
    if not texts_in_cluster:
        return "(no messages)"
    if seed is not None:
        rng = random.Random(seed)
        sample = rng.sample(texts_in_cluster, min(n_sample, len(texts_in_cluster)))
    else:
        sample = random.sample(texts_in_cluster, min(n_sample, len(texts_in_cluster)))
    truncated = [
        (t[:MAX_SUMMARY_TEXT_LEN] + "..." if len(t) > MAX_SUMMARY_TEXT_LEN else t)
        for t in sample
    ]
    bullet = "\n".join(f"- {t}" for t in truncated)
    prompt = (
        "Below are sample first messages from users in a single cluster.\n\n"
        "Summarize the common theme or intent in 1-2 short sentences (what are these users typically asking for or doing?).\n\n"
        f"{bullet}"
    )
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"(LLM error: {e})"


def summarize_all_clusters(
    df: pd.DataFrame,
    cluster_col: str = "cluster_id",
    text_col: str = "text",
    client: Optional[Any] = None,
    n_sample: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    For each cluster, sample messages and call LLM once. Returns DataFrame with columns
    cluster_id, count, summary.
    """
    if client is None:
        from openai import OpenAI
        client = OpenAI()
    rows = []
    for cid in sorted(df[cluster_col].dropna().unique()):
        subset = df[df[cluster_col] == cid][text_col].dropna().astype(str)
        texts = subset[subset.str.strip() != ""].tolist()
        summary = summarize_cluster_with_llm(texts, client, n_sample=n_sample, seed=seed)
        rows.append({"cluster_id": cid, "count": len(texts), "summary": summary})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LLM name + description for sub-categories (within each intent)
# ---------------------------------------------------------------------------

def name_and_describe_subcategory(
    texts_in_subcategory: list[str],
    client: Any,
    parent_intent: str,
    n_sample: int = 15,
    max_tokens: int = 120,
    seed: Optional[int] = 42,
    model: str = "gpt-4o-mini",
) -> tuple[str, str]:
    """
    Ask LLM for a very short name and a one-sentence description for this sub-category.
    Returns (name, description).
    """
    if not texts_in_subcategory:
        return ("(empty)", "(no messages)")
    if seed is not None:
        rng = random.Random(seed)
        sample = rng.sample(texts_in_subcategory, min(n_sample, len(texts_in_subcategory)))
    else:
        sample = random.sample(texts_in_subcategory, min(n_sample, len(texts_in_subcategory)))
    truncated = [
        (t[:MAX_SUMMARY_TEXT_LEN] + "..." if len(t) > MAX_SUMMARY_TEXT_LEN else t)
        for t in sample
    ]
    bullet = "\n".join(f"- {t}" for t in truncated)
    prompt = (
        f"These are sample user messages from one sub-category (parent category: {parent_intent}).\n\n"
        "Reply with exactly two lines:\n"
        "Line 1: A very short name for this sub-category (1-3 words, snake_case style like coding, creative_writing, product_review).\n"
        "Line 2: One sentence describing what these users are typically asking for or doing.\n\n"
        f"Sample messages:\n{bullet}"
    )
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
        )
        raw = (r.choices[0].message.content or "").strip()
        lines = [s.strip() for s in raw.split("\n") if s.strip()]
        name = lines[0] if lines else "(unnamed)"
        description = lines[1] if len(lines) > 1 else ""
        return (name, description)
    except Exception as e:
        return (f"(error: {e})", "")


def name_all_subcategories(
    df: pd.DataFrame,
    intent_col: str = "intent_rule",
    sub_cluster_col: str = "sub_cluster_id",
    text_col: str = "text",
    client: Optional[Any] = None,
    n_sample: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    For each (intent, sub_cluster_id), get LLM-generated name and one-sentence description.
    Returns DataFrame with columns: intent_rule, sub_cluster_id, count, name, description.
    """
    if client is None:
        from openai import OpenAI
        client = OpenAI()
    rows = []
    for intent in sorted(df[intent_col].dropna().unique()):
        sub_df = df[(df[intent_col] == intent) & (df[sub_cluster_col] >= 0)]
        for sid in sorted(sub_df[sub_cluster_col].unique()):
            subset = sub_df[sub_df[sub_cluster_col] == sid][text_col].dropna().astype(str)
            texts = subset[subset.str.strip() != ""].tolist()
            name, desc = name_and_describe_subcategory(
                texts, client, parent_intent=intent, n_sample=n_sample, seed=seed
            )
            rows.append({
                "intent_rule": intent,
                "sub_cluster_id": sid,
                "count": len(texts),
                "name": name,
                "description": desc,
            })
    return pd.DataFrame(rows)
