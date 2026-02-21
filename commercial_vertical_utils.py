"""
Commercial investigation vertical extraction. Used by commercial_vertical_extraction.ipynb.
Load commercial_investigation data, add all_queries/all_answers, assign business verticals via
IAB taxonomy keyword match, LLM (structured + intent re-classification, parallel), and embedding-based.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional
from urllib.request import urlopen

import numpy as np
import pandas as pd

# IAB Content Taxonomy 3.0 raw TSV URL
IAB_TAXONOMY_URL = (
    "https://raw.githubusercontent.com/InteractiveAdvertisingBureau/Taxonomies/main/"
    "Content%20Taxonomies/Content%20Taxonomy%203.0.tsv"
)

# Fallback minimal Tier 1 list if URL fails (business vertical style)
IAB_TIER1_FALLBACK = [
    "Arts, Entertainment, and Media",
    "Automotive",
    "Books and Literature",
    "Business and Finance",
    "Careers",
    "Education",
    "Family and Relationships",
    "Fine Art",
    "Food & Drink",
    "Healthy Living",
    "Hobbies & Interests",
    "Home & Garden",
    "Law, Gov't and Politics",
    "News",
    "Personal Finance",
    "Real Estate",
    "Science",
    "Shopping",
    "Sports",
    "Style & Fashion",
    "Technology & Computing",
    "Travel",
]

# Reference verticals for embedding-based assignment (short labels)
REFERENCE_VERTICALS = [
    "Sports",
    "Health",
    "Real Estate",
    "Technology",
    "Finance",
    "Travel",
    "Education",
    "Automotive",
    "Shopping",
    "Food & Drink",
    "Other",
]

# Rule-based vertical keywords: vertical_label -> list of keywords (first match wins)
RULE_VERTICAL_KEYWORDS: list[tuple[str, list[str]]] = [
    ("Sports", ["sport", "game", "team", "player", "league", "match", "fitness", "workout", "running", "gym"]),
    ("Health", ["health", "medical", "doctor", "medicine", "symptom", "treatment", "therapy", "wellness", "diet"]),
    ("Real Estate", ["real estate", "house", "apartment", "mortgage", "property", "rent", "buying a home", "realtor"]),
    ("Technology", ["software", "laptop", "phone", "computer", "gadget", "app", "tech", "camera", "tablet"]),
    ("Finance", ["invest", "stock", "bank", "loan", "credit", "insurance", "retirement", "savings"]),
    ("Travel", ["travel", "hotel", "flight", "vacation", "trip", "destination", "booking"]),
    ("Education", ["course", "university", "degree", "learn", "school", "training", "certification"]),
    ("Automotive", ["car", "vehicle", "tire", "automotive", "sedan", "suv", "electric car"]),
    ("Shopping", ["buy", "product", "brand", "compare", "review", "best", "recommend"]),
    ("Food & Drink", ["restaurant", "recipe", "food", "wine", "coffee", "kitchen", "cook"]),
]


# ---------------------------------------------------------------------------
# Text columns: all_queries, all_answers
# ---------------------------------------------------------------------------


def add_all_queries_answers_columns(
    df: pd.DataFrame,
    conversation_col: str = "conversation",
) -> pd.DataFrame:
    """Add all_queries and all_answers columns from conversation (all user and all assistant messages)."""
    from insights_utils import get_all_user_content, get_all_assistant_content

    out = df.copy()
    convos = out[conversation_col]
    out["all_queries"] = convos.map(lambda c: get_all_user_content(c) if c is not None else "")
    out["all_answers"] = convos.map(lambda c: get_all_assistant_content(c) if c is not None else "")
    return out


# ---------------------------------------------------------------------------
# IAB taxonomy load and keyword-based assignment
# ---------------------------------------------------------------------------


def _normalize_word(w: str) -> str:
    return re.sub(r"[^a-z0-9]", "", w.lower()) if w else ""


def _tokenize_for_match(text: str) -> set[str]:
    """Lowercase words, alphanumeric only, length >= 2."""
    if not text or not isinstance(text, str):
        return set()
    words = re.findall(r"[a-z0-9]{2,}", text.lower())
    return set(words)


def load_iab_taxonomy(
    url: str = IAB_TAXONOMY_URL,
    data_dir: Optional[Path] = None,
) -> list[tuple[str, str, set[str]]]:
    """
    Load IAB taxonomy and return list of (tier1, tier2, keywords) for matching.
    keywords is derived from category names (Name, Tier 1, Tier 2).
    Tries: (1) data_dir/Content_Taxonomy_3.0.tsv, (2) data_dir/iab_content_taxonomy_tier1_tier2.csv,
    (3) url, (4) embedded fallback.
    """
    if data_dir is not None:
        data_dir = Path(data_dir)
        tsv_path = data_dir / "Content_Taxonomy_3.0.tsv"
        if tsv_path.exists():
            try:
                return _parse_iab_tsv(tsv_path.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                pass
        csv_path = data_dir / "iab_content_taxonomy_tier1_tier2.csv"
        if csv_path.exists():
            try:
                return _parse_iab_csv(csv_path)
            except Exception:
                pass
    try:
        with urlopen(url, timeout=15) as resp:
            text = resp.read().decode("utf-8", errors="replace")
        return _parse_iab_tsv(text)
    except Exception:
        # Fallback: tier1 only with keywords from name
        return [
            (t1, "", _tokenize_for_match(t1))
            for t1 in IAB_TIER1_FALLBACK
        ]


def _parse_iab_csv(csv_path: Path) -> list[tuple[str, str, set[str]]]:
    """Parse local CSV with columns Tier 1, Tier 2 (optional Name)."""
    small = pd.read_csv(csv_path)
    tier1_col = next((c for c in small.columns if "tier" in c.lower() and "1" in c), small.columns[0])
    tier2_col = next((c for c in small.columns if "tier" in c.lower() and "2" in c), None)
    name_col = next((c for c in small.columns if "name" in c.lower()), None)
    result = []
    seen: set[tuple[str, str]] = set()
    for _, row in small.iterrows():
        t1 = str(row.get(tier1_col, "")).strip()
        if not t1:
            continue
        t2 = str(row.get(tier2_col, t1)).strip() if tier2_col else ""
        if (t1, t2) in seen:
            continue
        seen.add((t1, t2))
        name = str(row.get(name_col, "")).strip() if name_col else ""
        kw = _tokenize_for_match(" ".join([name, t1, t2]))
        if not kw:
            kw = _tokenize_for_match(t1)
        result.append((t1, t2, kw))
    return result


def _parse_iab_tsv(text: str) -> list[tuple[str, str, set[str]]]:
    """Parse IAB TSV content. Returns (tier1, tier2, keywords)."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        return []
    # Skip title line; second line is header
    header = lines[1].split("\t")
    try:
        idx_id = header.index("Unique ID")
        idx_name = header.index("Name")
        idx_t1 = header.index("Tier 1")
        idx_t2 = header.index("Tier 2")
    except ValueError:
        # Try by position: Unique ID, Parent, Name, Tier 1, Tier 2, Tier 3, Tier 4
        if len(header) < 5:
            return []
        idx_id, idx_name, idx_t1, idx_t2 = 0, 2, 3, 4
    seen: set[tuple[str, str]] = set()
    result: list[tuple[str, str, set[str]]] = []
    for line in lines[2:]:
        parts = line.split("\t")
        if len(parts) <= max(idx_t1, idx_t2):
            continue
        tier1 = (parts[idx_t1] or "").strip()
        tier2 = (parts[idx_t2] or "").strip()
        name = (parts[idx_name] if idx_name < len(parts) else "").strip()
        if not tier1:
            continue
        key = (tier1, tier2)
        if key in seen:
            continue
        seen.add(key)
        kw = _tokenize_for_match(" ".join([name, tier1, tier2]))
        if not kw:
            kw = _tokenize_for_match(tier1)
        result.append((tier1, tier2, kw))
    return result


def assign_vertical_iab(
    df: pd.DataFrame,
    text_col_queries: str = "all_queries",
    text_col_answers: str = "all_answers",
    taxonomy: Optional[list[tuple[str, str, set[str]]]] = None,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Assign vertical_tier1_iab and vertical_tier2_iab by keyword matching over combined queries + answers.
    Uses score: sum of keyword matches; assigns tier1 and tier2 with highest score.
    """
    if taxonomy is None:
        taxonomy = load_iab_taxonomy(data_dir=data_dir)
    out = df.copy()
    combined = (
        out[text_col_queries].fillna("").astype(str)
        + " "
        + out[text_col_answers].fillna("").astype(str)
    )
    tier1_scores: dict[str, list[float]] = {}
    tier2_scores: dict[tuple[str, str], list[float]] = {}
    for (t1, t2, kw) in taxonomy:
        tier1_scores.setdefault(t1, [0.0] * len(df))
        tier2_scores.setdefault((t1, t2), [0.0] * len(df))
    for i, text in enumerate(combined):
        tokens = _tokenize_for_match(text)
        if not tokens:
            continue
        for t1, t2, kw in taxonomy:
            matches = len(tokens & kw)
            if matches > 0:
                tier1_scores[t1][i] += matches
                tier2_scores[(t1, t2)][i] += matches
    # Assign best tier1 per row
    t1_col = []
    t2_col = []
    for i in range(len(df)):
        best_t1 = ""
        best_score = 0.0
        for t1, scores in tier1_scores.items():
            if scores[i] > best_score:
                best_score = scores[i]
                best_t1 = t1
        if not best_t1:
            best_t1 = "Other"
        t1_col.append(best_t1)
        # Best tier2 under best_t1
        best_t2 = ""
        best_t2_score = 0.0
        for (t1, t2), scores in tier2_scores.items():
            if t1 == best_t1 and t2 and scores[i] > best_t2_score:
                best_t2_score = scores[i]
                best_t2 = t2
        t2_col.append(best_t2 if best_t2 else best_t1)
    out["vertical_tier1_iab"] = t1_col
    out["vertical_tier2_iab"] = t2_col
    return out


# ---------------------------------------------------------------------------
# LLM-based vertical + intent re-classification (parallel, cached)
# ---------------------------------------------------------------------------

VERTICAL_LLM_LABELS = "Sports, Health, Real Estate, Technology, Finance, Travel, Education, Automotive, Shopping, Food & Drink, Other"
INTENT_OPTIONS = "informational, navigational, commercial_investigation, transactional"


def _label_one_vertical_intent(
    client: Any,
    all_queries: str,
    all_answers: str,
    max_combined_len: int = 6000,
) -> dict[str, str]:
    """Call LLM once for one conversation; return dict with vertical_tier1, vertical_tier2, intent_revised."""
    from insights_utils import _call_chat, _truncate

    combined = (all_queries or "") + "\n\n" + (all_answers or "")
    truncated = _truncate(combined, max_len=max_combined_len)
    prompt = f"""Based on the full conversation below, output a JSON object with exactly these keys:
- "vertical_tier1": one of [{VERTICAL_LLM_LABELS}]
- "vertical_tier2": a more specific vertical or same as vertical_tier1
- "intent_revised": one of [{INTENT_OPTIONS}]

Reply with only the JSON object, no other text.

Conversation:
{truncated}"""
    try:
        raw = _call_chat(
            client,
            prompt,
            "You output only a valid JSON object with keys vertical_tier1, vertical_tier2, intent_revised.",
        )
        raw = (raw or "").strip()
        # Extract JSON
        if raw.startswith("{"):
            obj = json.loads(raw)
        else:
            start = raw.find("{")
            end = raw.rfind("}")
            obj = json.loads(raw[start : end + 1]) if start >= 0 and end > start else {}
        v1 = str(obj.get("vertical_tier1", "Other")).strip() or "Other"
        v2 = str(obj.get("vertical_tier2", v1)).strip() or v1
        intent = str(obj.get("intent_revised", "informational")).strip()
        if intent not in INTENT_OPTIONS.split(", "):
            intent = "informational"
        return {"vertical_tier1_llm": v1, "vertical_tier2_llm": v2, "intent_revised": intent}
    except Exception:
        return {"vertical_tier1_llm": "Other", "vertical_tier2_llm": "Other", "intent_revised": "informational"}


def label_vertical_intent_llm_parallel(
    df: pd.DataFrame,
    queries_col: str = "all_queries",
    answers_col: str = "all_answers",
    id_col: str = "conversation_id",
    cache_path: str | Path = "intent_output/vertical_intent_llm.parquet",
    use_cache: bool = True,
    batch_size: int = 100,
    max_workers: int = 10,
) -> pd.DataFrame:
    """
    Add vertical_tier1_llm, vertical_tier2_llm, intent_revised via OpenAI with parallel calls and cache.
    """
    from insights_utils import (
        _get_client,
        load_label_cache,
        merge_cached_labels,
        save_label_cache,
    )

    label_cols = ["vertical_tier1_llm", "vertical_tier2_llm", "intent_revised"]
    out = df.copy()
    cache_df = load_label_cache(cache_path) if use_cache else None
    out, missing_idx = merge_cached_labels(out, cache_df, id_col=id_col, label_cols=label_cols)
    to_label = out.loc[missing_idx]
    if len(to_label) == 0:
        return out

    client = _get_client()
    queries_list = to_label[queries_col].fillna("").astype(str).tolist()
    answers_list = to_label[answers_col].fillna("").astype(str).tolist()

    all_v1: list[str] = []
    all_v2: list[str] = []
    all_intent: list[str] = []
    for start in range(0, len(queries_list), batch_size):
        end = min(start + batch_size, len(queries_list))
        batch_q = queries_list[start:end]
        batch_a = answers_list[start:end]
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(
                ex.map(
                    lambda qa: _label_one_vertical_intent(client, qa[0], qa[1]),
                    zip(batch_q, batch_a),
                )
            )
        for r in results:
            all_v1.append(r["vertical_tier1_llm"])
            all_v2.append(r["vertical_tier2_llm"])
            all_intent.append(r["intent_revised"])
        save_label_cache(
            cache_path,
            to_label[id_col].iloc[start:end],
            {
                "vertical_tier1_llm": all_v1[-len(results) :],
                "vertical_tier2_llm": all_v2[-len(results) :],
                "intent_revised": all_intent[-len(results) :],
            },
            id_col=id_col,
        )
        time.sleep(0.2)
    out.loc[missing_idx, "vertical_tier1_llm"] = all_v1
    out.loc[missing_idx, "vertical_tier2_llm"] = all_v2
    out.loc[missing_idx, "intent_revised"] = all_intent
    return out


# ---------------------------------------------------------------------------
# Embedding-based vertical assignment
# ---------------------------------------------------------------------------


def assign_vertical_embedding(
    df: pd.DataFrame,
    text_col_queries: str = "all_queries",
    text_col_answers: str = "all_answers",
    reference_vertical_descriptions: Optional[list[tuple[str, str]]] = None,
    model_name: str = "all-MiniLM-L6-v2",
) -> pd.DataFrame:
    """
    Assign vertical_embedding by cosine similarity between (queries+answers) embedding and reference vertical embeddings.
    reference_vertical_descriptions: list of (label, description). If None, uses REFERENCE_VERTICALS with generic descriptions.
    """
    from intent_clustering import embed_texts

    if reference_vertical_descriptions is None:
        reference_vertical_descriptions = [
            (v, f"Content about {v}.") for v in REFERENCE_VERTICALS
        ]
    labels = [x[0] for x in reference_vertical_descriptions]
    descs = [x[1] for x in reference_vertical_descriptions]
    ref_emb = embed_texts(pd.Series(descs), model_name=model_name)
    combined = (
        df[text_col_queries].fillna("").astype(str)
        + " "
        + df[text_col_answers].fillna("").astype(str)
    )
    emb = embed_texts(combined, model_name=model_name)
    # Cosine similarity: (n, dim) @ (dim, k) -> (n, k)
    ref_norm = ref_emb / (np.linalg.norm(ref_emb, axis=1, keepdims=True) + 1e-9)
    sim = emb @ ref_norm.T
    best_idx = np.argmax(sim, axis=1)
    out = df.copy()
    out["vertical_embedding"] = [labels[i] for i in best_idx]
    return out


# ---------------------------------------------------------------------------
# Rule-based vertical (optional)
# ---------------------------------------------------------------------------


def assign_vertical_rule_based(
    df: pd.DataFrame,
    text_col_queries: str = "all_queries",
    text_col_answers: str = "all_answers",
    default: str = "Other",
) -> pd.DataFrame:
    """Assign vertical_rule from keyword rules (first match wins)."""
    combined = (
        df[text_col_queries].fillna("").astype(str)
        + " "
        + df[text_col_answers].fillna("").astype(str)
    ).str.lower()
    labels = []
    for _, text in combined.items():
        assigned = default
        for label, keywords in RULE_VERTICAL_KEYWORDS:
            if any(kw in text for kw in keywords):
                assigned = label
                break
        labels.append(assigned)
    out = df.copy()
    out["vertical_rule"] = labels
    return out
