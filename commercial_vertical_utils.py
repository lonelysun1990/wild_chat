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

# Intent guidelines (aligned with docs/rule-based_intent_taxonomy_plan.md) so the LLM uses the same definitions.
# Transactional narrowed to real purchase/booking/registration only (see docs/transactional_intent_analysis.md).
INTENT_GUIDELINES = """Use these definitions for intent_major:
- informational: User wants to learn something or have the AI do a task (how/what/why, guide, tutorial, explain, definition, understand, tips, examples, FAQ; or create/write/build/generate code, text, or content; or fix/debug; or roleplay/scenario).
- navigational: User wants to find a site or entity (name of, website for, link to, app for, where to find, official site, URL, contact).
- commercial_investigation: User is comparing or researching before buying (best, top 10, review(s), comparison, compare, versus, recommend, which one, should I get, worth it, specs, pros and cons, buying guide).
- transactional: User intends to complete a real-world commercial or service transaction now: purchase, pay, order, checkout, add to cart, book now, reserve, sign up for a paid service/event, refund, return, get a ticket. Do NOT use transactional when the user is only asking the AI to create/write/build/generate content, to write a letter or post, to convert files, to follow agent commands, or to do a coding task; those are informational (coding or creative_writing)."""

# Sub-category guidelines for intent-only classification (aligned with intent_taxonomy.SUB_INTENT_RULES).
INTENT_SUBCATEGORY_GUIDELINES = """For intent_sub, use exactly one of these per intent_major:
- If intent_major is informational: coding (code, script, API, debug, programming), creative_writing (story, poem, write, fiction, character), education (explain, learn, concept, exam, textbook), support (fix, broken, error, troubleshoot, not working), casual_other (general learning or chat).
- If intent_major is navigational: use navigational.
- If intent_major is commercial_investigation: use commercial_product.
- If intent_major is transactional: use transactional."""

# Valid sub per major for validation
VALID_SUB_BY_MAJOR: dict[str, list[str]] = {
    "informational": ["coding", "creative_writing", "education", "support", "casual_other"],
    "navigational": ["navigational"],
    "commercial_investigation": ["commercial_product"],
    "transactional": ["transactional"],
}


def _is_rate_limit_error(e: BaseException) -> bool:
    """True if the exception is OpenAI rate limit (429)."""
    if getattr(e, "status_code", None) == 429:
        return True
    try:
        from openai import RateLimitError
        return isinstance(e, RateLimitError)
    except ImportError:
        return "rate" in str(e).lower() and "limit" in str(e).lower()


def _label_one_vertical_intent(
    client: Any,
    all_queries: str,
    all_answers: str,
    max_combined_len: int = 6000,
) -> dict[str, Any]:
    """Call LLM once for one conversation; return dict with vertical_tier1_llm, vertical_tier2_llm, intent_revised, and optionally rate_limit_hit."""
    from insights_utils import _call_chat, _truncate

    combined = (all_queries or "") + "\n\n" + (all_answers or "")
    truncated = _truncate(combined, max_len=max_combined_len)
    prompt = f"""Based on the full conversation below, output a JSON object with exactly these keys:
- "vertical_tier1": one of [{VERTICAL_LLM_LABELS}]
- "vertical_tier2": a more specific vertical or same as vertical_tier1
- "intent_revised": one of [{INTENT_OPTIONS}]

{INTENT_GUIDELINES}

Reply with only the JSON object, no other text.

Conversation:
{truncated}"""
    default = {"vertical_tier1_llm": "Other", "vertical_tier2_llm": "Other", "intent_revised": "informational", "rate_limit_hit": False}
    for attempt in range(3):
        try:
            raw = _call_chat(
                client,
                prompt,
                "You output only a valid JSON object with keys vertical_tier1, vertical_tier2, intent_revised.",
            )
            raw = (raw or "").strip()
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
            return {"vertical_tier1_llm": v1, "vertical_tier2_llm": v2, "intent_revised": intent, "rate_limit_hit": attempt > 0}
        except Exception as e:
            if _is_rate_limit_error(e) and attempt < 2:
                time.sleep(60 * (attempt + 1))  # 60s, then 120s backoff
                continue
            return {**default, "rate_limit_hit": _is_rate_limit_error(e)}
    return default


def label_vertical_intent_llm_parallel(
    df: pd.DataFrame,
    queries_col: str = "all_queries",
    answers_col: str = "all_answers",
    id_col: str = "conversation_id",
    cache_path: str | Path = "intent_output/vertical_intent_llm.parquet",
    use_cache: bool = True,
    batch_size: int = 150,
    max_workers: int = 150,
) -> pd.DataFrame:
    """
    Add vertical_tier1_llm, vertical_tier2_llm, intent_revised via OpenAI with parallel calls and cache.

    To fully use max_workers concurrent requests, set batch_size >= max_workers.
    If you see rate limit (429) messages in the summary, reduce max_workers or increase the sleep between batches.
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
    total_rate_limit_hits = 0
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
            if r.get("rate_limit_hit"):
                total_rate_limit_hits += 1
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
    if total_rate_limit_hits > 0:
        print(f"Note: OpenAI rate limit (429) was hit for {total_rate_limit_hits} request(s); retried with backoff. Consider lowering LLM_MAX_WORKERS if this is frequent.")
    return out


# ---------------------------------------------------------------------------
# LLM-based vertical + brands only (no intent re-classification, parallel, cached)
# ---------------------------------------------------------------------------

BRANDS_WHERE_OPTIONS = ("query_only", "answer_only", "both")


def _label_one_vertical_brands(
    client: Any,
    all_queries: str,
    all_answers: str,
    max_combined_len: int = 6000,
) -> dict[str, Any]:
    """Call LLM once for one conversation; return dict with vertical_tier1_llm, vertical_tier2_llm, brands, rate_limit_hit.
    brands is a list of {"name": str, "where": "query_only"|"answer_only"|"both"}.
    """
    from insights_utils import _call_chat, _truncate

    combined = (all_queries or "") + "\n\n" + (all_answers or "")
    truncated = _truncate(combined, max_len=max_combined_len)
    prompt = f"""Based on the full conversation below, output a JSON object with exactly these keys:
- "vertical_tier1": one of [{VERTICAL_LLM_LABELS}]
- "vertical_tier2": a more specific vertical or same as vertical_tier1
- "brands": list of objects, each with "name" and "where" (one of: query_only, answer_only, both). Extract ANY specific named entity that is commercial or a business reference:
  * Product or brand names (e.g. iPhone, Nike, Samsung)
  * Company or business names (e.g. Acme Corp, Starbucks, Tesla)
  * Website or domain names: if a URL or site is mentioned (e.g. "check example.com", "visit nike.com"), use the site/domain name as the brand (e.g. "example", "Nike")
  * App names, service names, or store names (e.g. Spotify, Amazon, Netflix)
  Use query_only if the name appears only in user messages, answer_only if only in assistant messages, both if in both. Empty list [] only if no such named entities appear.
  Important: Do NOT list the vertical category itself (e.g. "Technology", "Shopping") as a brand. We need specific companies, products, sites, or services that are explicitly named.
  When in doubt, include any clearly named company, product, website, or service; it is better to include a plausible brand than to omit one.
- "product_mentioned": string or null. If the conversation explicitly mentions a specific product (e.g. "iPhone 15", "Nike Air Max"), give that product name; otherwise null.
- "product_inferred": string or null. If no specific product was mentioned but brands/context suggest a likely product (e.g. brand is Nike -> "running shoes" or a typical product line), give the most relevant product; otherwise null.
- "deal_size_usd": number or null. Estimated potential purchase or deal size in USD for this conversation. Use typical price ranges for the product/category (e.g. $50 for a course, $500 for a laptop, $30 for a book). Do not leave null unless there is no commercial intent or no identifiable product; prefer a plausible mid-range estimate.

Reply with only the JSON object, no other text.

Conversation:
{truncated}"""
    default = {
        "vertical_tier1_llm": "Other",
        "vertical_tier2_llm": "Other",
        "brands": [],
        "product_mentioned": None,
        "product_inferred": None,
        "deal_size_usd": None,
        "rate_limit_hit": False,
    }
    for attempt in range(3):
        try:
            raw = _call_chat(
                client,
                prompt,
                "You output only a valid JSON object with keys vertical_tier1, vertical_tier2, brands, product_mentioned, product_inferred, deal_size_usd.",
            )
            raw = (raw or "").strip()
            if raw.startswith("{"):
                obj = json.loads(raw)
            else:
                start = raw.find("{")
                end = raw.rfind("}")
                obj = json.loads(raw[start : end + 1]) if start >= 0 and end > start else {}
            v1 = str(obj.get("vertical_tier1", "Other")).strip() or "Other"
            v2 = str(obj.get("vertical_tier2", v1)).strip() or v1
            brands_raw = obj.get("brands")
            if not isinstance(brands_raw, list):
                brands_raw = []
            brands = []
            for b in brands_raw:
                if not isinstance(b, dict):
                    continue
                name = str(b.get("name", "")).strip()
                if not name:
                    continue
                where = str(b.get("where", "both")).strip().lower()
                if where not in BRANDS_WHERE_OPTIONS:
                    where = "both"
                brands.append({"name": name, "where": where})
            product_mentioned = obj.get("product_mentioned")
            if product_mentioned is not None:
                product_mentioned = str(product_mentioned).strip() or None
            product_inferred = obj.get("product_inferred")
            if product_inferred is not None:
                product_inferred = str(product_inferred).strip() or None
            deal_size_usd = obj.get("deal_size_usd")
            if deal_size_usd is not None:
                try:
                    deal_size_usd = float(deal_size_usd)
                    if deal_size_usd < 0 or not np.isfinite(deal_size_usd):
                        deal_size_usd = None
                except (TypeError, ValueError):
                    deal_size_usd = None
            return {
                "vertical_tier1_llm": v1,
                "vertical_tier2_llm": v2,
                "brands": brands,
                "product_mentioned": product_mentioned,
                "product_inferred": product_inferred,
                "deal_size_usd": deal_size_usd,
                "rate_limit_hit": attempt > 0,
            }
        except Exception as e:
            if _is_rate_limit_error(e) and attempt < 2:
                time.sleep(60 * (attempt + 1))
                continue
            return {**default, "rate_limit_hit": _is_rate_limit_error(e)}
    return default


def label_vertical_brands_llm_parallel(
    df: pd.DataFrame,
    queries_col: str = "all_queries",
    answers_col: str = "all_answers",
    id_col: str = "conversation_id",
    cache_path: str | Path = "intent_output/commercial_vertical_brands_llm.parquet",
    use_cache: bool = True,
    batch_size: int = 100,
    max_workers: int = 100,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Add vertical_tier1_llm, vertical_tier2_llm, and brands via OpenAI with parallel calls and cache.
    brands: list of {"name": str, "where": "query_only"|"answer_only"|"both"}.
    Uses tqdm on the batch loop when show_progress is True.
    """
    from insights_utils import (
        _get_client,
        load_label_cache,
        merge_cached_labels,
        save_label_cache,
    )

    base_label_cols = ["vertical_tier1_llm", "vertical_tier2_llm", "brands"]
    optional_label_cols = ["product_mentioned", "product_inferred", "deal_size_usd"]
    out = df.copy()
    cache_df = load_label_cache(cache_path) if use_cache else None
    label_cols = base_label_cols + [c for c in optional_label_cols if cache_df is None or c in cache_df.columns]
    out, missing_idx = merge_cached_labels(out, cache_df, id_col=id_col, label_cols=label_cols)
    for c in optional_label_cols:
        if c not in out.columns:
            out[c] = None
    to_label = out.loc[missing_idx]
    if len(to_label) == 0:
        return out

    client = _get_client()
    queries_list = to_label[queries_col].fillna("").astype(str).tolist()
    answers_list = to_label[answers_col].fillna("").astype(str).tolist()

    all_v1: list[str] = []
    all_v2: list[str] = []
    all_brands: list[list[dict[str, str]]] = []
    all_product_mentioned: list[str | None] = []
    all_product_inferred: list[str | None] = []
    all_deal_size_usd: list[float | None] = []
    total_rate_limit_hits = 0

    batch_starts = list(range(0, len(queries_list), batch_size))
    iterator = batch_starts
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(batch_starts, desc="LLM vertical+brands", unit="batch")
        except ImportError:
            pass

    for start in iterator:
        end = min(start + batch_size, len(queries_list))
        batch_q = queries_list[start:end]
        batch_a = answers_list[start:end]
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(
                ex.map(
                    lambda qa: _label_one_vertical_brands(client, qa[0], qa[1]),
                    zip(batch_q, batch_a),
                )
            )
        for r in results:
            all_v1.append(r["vertical_tier1_llm"])
            all_v2.append(r["vertical_tier2_llm"])
            all_brands.append(r["brands"])
            all_product_mentioned.append(r.get("product_mentioned"))
            all_product_inferred.append(r.get("product_inferred"))
            all_deal_size_usd.append(r.get("deal_size_usd"))
            if r.get("rate_limit_hit"):
                total_rate_limit_hits += 1
        save_label_cache(
            cache_path,
            to_label[id_col].iloc[start:end],
            {
                "vertical_tier1_llm": all_v1[-len(results) :],
                "vertical_tier2_llm": all_v2[-len(results) :],
                "brands": all_brands[-len(results) :],
                "product_mentioned": all_product_mentioned[-len(results) :],
                "product_inferred": all_product_inferred[-len(results) :],
                "deal_size_usd": all_deal_size_usd[-len(results) :],
            },
            id_col=id_col,
        )
        time.sleep(0.2)
    out.loc[missing_idx, "vertical_tier1_llm"] = all_v1
    out.loc[missing_idx, "vertical_tier2_llm"] = all_v2
    # Assign brands as Series to avoid ValueError (inhomogeneous shape) from variable-length lists
    out.loc[missing_idx, "brands"] = pd.Series(all_brands, index=missing_idx).values
    out.loc[missing_idx, "product_mentioned"] = all_product_mentioned
    out.loc[missing_idx, "product_inferred"] = all_product_inferred
    out.loc[missing_idx, "deal_size_usd"] = all_deal_size_usd
    if total_rate_limit_hits > 0:
        print(f"Note: OpenAI rate limit (429) was hit for {total_rate_limit_hits} request(s); retried with backoff. Consider lowering max_workers if this is frequent.")
    return out


# ---------------------------------------------------------------------------
# Brand mention dynamics (post-processing for survival-style analytics)
# ---------------------------------------------------------------------------


def _normalize_brand_for_match(name: str) -> str:
    """Normalize brand name for case-insensitive substring match (strip, lower, collapse spaces)."""
    if not name or not isinstance(name, str):
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", name.lower())).strip()


def _brand_in_content(brand_name: str, content: str) -> bool:
    """Return True if brand (normalized, case-insensitive) appears in content."""
    if not content or not isinstance(content, str):
        return False
    norm_brand = _normalize_brand_for_match(brand_name)
    if not norm_brand:
        return False
    norm_content = content.lower()
    # Require brand tokens to appear (avoids very short false matches)
    tokens = norm_brand.split()
    if len(tokens) == 1:
        return norm_brand in norm_content
    return norm_brand in norm_content or all(t in norm_content for t in tokens)


def enrich_brands_with_mention_dynamics(
    conversation: list[Any],
    brands: list[dict[str, Any]],
    ended_soon_threshold: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Compute per-brand mention dynamics from conversation messages.
    Returns (brands_enriched, summary_dict).
    - brands_enriched: list of dicts, each original brand + first_mention_round, first_mention_role,
      mention_count_total, last_mention_round, mention_span, user_follow_up, brand_mentioned_once,
      conversation_ended_soon_after.
    - summary_dict: first_brand_first_round, first_brand_first_role, any_brand_user_introduced,
      any_brand_assistant_introduced, any_user_follow_up_on_brand, total_brands, n_messages.
    """
    n_messages = len(conversation) if conversation else 0
    last_msg_index = n_messages - 1 if n_messages else 0
    summary: dict[str, Any] = {
        "first_brand_first_round": None,
        "first_brand_first_role": None,
        "any_brand_user_introduced": False,
        "any_brand_assistant_introduced": False,
        "any_user_follow_up_on_brand": False,
        "total_brands": len(brands) if brands else 0,
        "n_messages": n_messages,
    }
    if not brands or not conversation:
        return (
            [dict(b, first_mention_round=None, first_mention_role=None, mention_count_total=0, last_mention_round=None, mention_span=0, user_follow_up=False, brand_mentioned_once=False, conversation_ended_soon_after=False) for b in (brands or [])],
            summary,
        )

    brands_enriched: list[dict[str, Any]] = []
    first_brand_first_round_min: int | None = None
    first_brand_first_role_val: str | None = None

    for b in brands:
        name = (b.get("name") or "").strip() if isinstance(b, dict) else ""
        where = b.get("where", "both") if isinstance(b, dict) else "both"
        if not name:
            brands_enriched.append({
                **dict(b),
                "first_mention_round": None,
                "first_mention_role": None,
                "mention_count_total": 0,
                "last_mention_round": None,
                "mention_span": 0,
                "user_follow_up": False,
                "brand_mentioned_once": False,
                "conversation_ended_soon_after": False,
            })
            continue

        mention_rounds: list[int] = []
        mention_roles: list[str] = []
        for i, msg in enumerate(conversation):
            if not isinstance(msg, dict):
                continue
            content = (msg.get("content") or "") if isinstance(msg.get("content"), str) else str(msg.get("content", ""))
            role = (msg.get("role") or "").strip().lower()
            if role not in ("user", "assistant"):
                continue
            if _brand_in_content(name, content):
                mention_rounds.append(i)
                mention_roles.append(role)

        first_round = mention_rounds[0] if mention_rounds else None
        first_role = mention_roles[0] if mention_roles else None
        last_round = mention_rounds[-1] if mention_rounds else None
        count_total = len(mention_rounds)
        mention_span = (last_round - first_round) if (first_round is not None and last_round is not None) else 0
        user_follow_up = False
        if first_round is not None and len(mention_rounds) > 1:
            for j, r in zip(mention_rounds, mention_roles):
                if j > first_round and r == "user":
                    user_follow_up = True
                    break
        brand_mentioned_once = count_total == 1
        ended_soon = (first_round is not None and last_msg_index >= 0 and (last_msg_index - first_round) <= ended_soon_threshold)

        enriched = {
            **dict(b),
            "name": name,
            "where": where,
            "first_mention_round": first_round,
            "first_mention_role": first_role,
            "mention_count_total": count_total,
            "last_mention_round": last_round,
            "mention_span": mention_span,
            "user_follow_up": user_follow_up,
            "brand_mentioned_once": brand_mentioned_once,
            "conversation_ended_soon_after": ended_soon,
        }
        brands_enriched.append(enriched)

        if first_round is not None and (first_brand_first_round_min is None or first_round < first_brand_first_round_min):
            first_brand_first_round_min = first_round
            first_brand_first_role_val = first_role
        if first_role == "user":
            summary["any_brand_user_introduced"] = True
        if first_role == "assistant":
            summary["any_brand_assistant_introduced"] = True
        if user_follow_up:
            summary["any_user_follow_up_on_brand"] = True

    summary["first_brand_first_round"] = first_brand_first_round_min
    summary["first_brand_first_role"] = first_brand_first_role_val
    return brands_enriched, summary


def add_brand_mention_dynamics(
    df: pd.DataFrame,
    conversation_col: str = "conversation",
    brands_col: str = "brands",
    ended_soon_threshold: int = 2,
) -> pd.DataFrame:
    """
    Add brands_enriched and per-conversation summary columns (first_brand_first_round,
    first_brand_first_role, any_brand_user_introduced, any_brand_assistant_introduced,
    any_user_follow_up_on_brand, total_brands, n_messages) by running enrich_brands_with_mention_dynamics
    on each row. Does not remove existing brands column.
    """
    out = df.copy()
    convos = out[conversation_col]
    brands_list = out[brands_col] if brands_col in out.columns else [[]] * len(out)

    enriched_list: list[list[dict[str, Any]]] = []
    first_rounds: list[Any] = []
    first_roles: list[Any] = []
    any_user_intro: list[bool] = []
    any_asst_intro: list[bool] = []
    any_follow_up: list[bool] = []
    total_brands_list: list[int] = []
    n_messages_list: list[int] = []

    for conv, brands in zip(convos, brands_list):
        conv_list = conv if isinstance(conv, list) else (list(conv) if conv is not None else [])
        b_list = brands if isinstance(brands, list) else []
        enriched, summary = enrich_brands_with_mention_dynamics(conv_list, b_list, ended_soon_threshold=ended_soon_threshold)
        enriched_list.append(enriched)
        first_rounds.append(summary["first_brand_first_round"])
        first_roles.append(summary["first_brand_first_role"])
        any_user_intro.append(summary["any_brand_user_introduced"])
        any_asst_intro.append(summary["any_brand_assistant_introduced"])
        any_follow_up.append(summary["any_user_follow_up_on_brand"])
        total_brands_list.append(summary["total_brands"])
        n_messages_list.append(summary["n_messages"])

    out["brands_enriched"] = pd.Series(enriched_list, index=out.index).values
    out["first_brand_first_round"] = first_rounds
    out["first_brand_first_role"] = first_roles
    out["any_brand_user_introduced"] = any_user_intro
    out["any_brand_assistant_introduced"] = any_asst_intro
    out["any_user_follow_up_on_brand"] = any_follow_up
    out["total_brands"] = total_brands_list
    out["n_messages"] = n_messages_list
    return out


# ---------------------------------------------------------------------------
# LLM-based intent-only classification (parallel, cached)
# ---------------------------------------------------------------------------


def _label_one_intent(
    client: Any,
    query_text: str,
    max_query_len: int = 4000,
) -> dict[str, Any]:
    """Call LLM once for one query; return dict with intent_major, intent_sub, and optionally rate_limit_hit."""
    from insights_utils import _call_chat, _truncate

    truncated = _truncate(query_text or "", max_len=max_query_len)
    prompt = f"""Classify the user query below. Output a JSON object with exactly these keys:
- "intent_major": one of [{INTENT_OPTIONS}]
- "intent_sub": the correct sub-category for that major (see guidelines)

{INTENT_GUIDELINES}

{INTENT_SUBCATEGORY_GUIDELINES}

Reply with only the JSON object, no other text.

Query:
{truncated}"""
    default = {"intent_major": "informational", "intent_sub": "casual_other", "rate_limit_hit": False}
    for attempt in range(3):
        try:
            raw = _call_chat(
                client,
                prompt,
                "You output only a valid JSON object with keys intent_major, intent_sub.",
            )
            raw = (raw or "").strip()
            if raw.startswith("{"):
                obj = json.loads(raw)
            else:
                start = raw.find("{")
                end = raw.rfind("}")
                obj = json.loads(raw[start : end + 1]) if start >= 0 and end > start else {}
            major = str(obj.get("intent_major", "informational")).strip()
            if major not in INTENT_OPTIONS.split(", "):
                major = "informational"
            valid_subs = VALID_SUB_BY_MAJOR.get(major, ["casual_other"])
            sub = str(obj.get("intent_sub", valid_subs[0])).strip()
            if sub not in valid_subs:
                sub = valid_subs[0]
            return {"intent_major": major, "intent_sub": sub, "rate_limit_hit": attempt > 0}
        except Exception as e:
            if _is_rate_limit_error(e) and attempt < 2:
                time.sleep(60 * (attempt + 1))
                continue
            return {**default, "rate_limit_hit": _is_rate_limit_error(e)}
    return default


def label_intent_llm_parallel(
    df: pd.DataFrame,
    text_col: str = "text",
    id_col: str = "conversation_id",
    cache_path: str | Path = "intent_output/intent_llm.parquet",
    use_cache: bool = True,
    batch_size: int = 150,
    max_workers: int = 150,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Add intent_major and intent_sub via OpenAI with parallel calls and cache (query text only).

    To fully use max_workers concurrent requests, set batch_size >= max_workers.
    If you see rate limit (429) messages, reduce max_workers or increase the sleep between batches.
    """
    from insights_utils import (
        _get_client,
        load_label_cache,
        merge_cached_labels,
        save_label_cache,
    )

    label_cols = ["intent_major", "intent_sub"]
    out = df.copy()
    cache_df = load_label_cache(cache_path) if use_cache else None
    out, missing_idx = merge_cached_labels(out, cache_df, id_col=id_col, label_cols=label_cols)
    to_label = out.loc[missing_idx]
    if len(to_label) == 0:
        return out

    client = _get_client()
    text_list = to_label[text_col].fillna("").astype(str).tolist()

    all_major: list[str] = []
    all_sub: list[str] = []
    total_rate_limit_hits = 0
    batch_starts = list(range(0, len(text_list), batch_size))
    batch_iter = batch_starts
    if show_progress:
        try:
            from tqdm.auto import tqdm
            batch_iter = tqdm(batch_starts, desc="Intent LLM", unit="batch")
        except ImportError:
            pass
    for start in batch_iter:
        end = min(start + batch_size, len(text_list))
        batch_texts = text_list[start:end]
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(
                ex.map(
                    lambda t: _label_one_intent(client, t),
                    batch_texts,
                )
            )
        for r in results:
            all_major.append(r["intent_major"])
            all_sub.append(r["intent_sub"])
            if r.get("rate_limit_hit"):
                total_rate_limit_hits += 1
        save_label_cache(
            cache_path,
            to_label[id_col].iloc[start:end],
            {
                "intent_major": all_major[-len(results) :],
                "intent_sub": all_sub[-len(results) :],
            },
            id_col=id_col,
        )
        time.sleep(0.2)
    out.loc[missing_idx, "intent_major"] = all_major
    out.loc[missing_idx, "intent_sub"] = all_sub
    if total_rate_limit_hits > 0:
        print(f"Note: OpenAI rate limit (429) was hit for {total_rate_limit_hits} request(s); retried with backoff. Consider lowering max_workers if this is frequent.")
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
