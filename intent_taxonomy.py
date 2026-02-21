"""
4-category intent taxonomy (informational, navigational, commercial_investigation, transactional)
with sub-categories. Used by intent_classification.ipynb.
Classification uses query text only (first user message).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Major intent rules (first match wins; order: transactional, commercial_investigation, navigational, informational)
# ---------------------------------------------------------------------------

KEYWORDS_TRANSACTIONAL = [
    "buy", "coupon", "order", "purchase", "cheap", "price", "pricing", "local", "store", "shop",
    "discount", "cart", "checkout", "pay", "payment", "shipping", "deliver", "delivery",
    "subscribe", "sign up", "get it", "add to cart", "buy now", "place order", "order now",
    "purchase now", "in stock", "availability", "where to buy", "buy from", "sell", "selling",
    "refund", "return", "promo code", "voucher", "deal", "sale", "on sale", "free shipping",
    "buy local", "near me", "in my area", "book now", "reserve", "appointment", "sign up for",
    "register for",
]

KEYWORDS_COMMERCIAL_INVESTIGATION = [
    "best", "top", "top 10", "top 5", "pricing", "review", "reviews", "comparison", "compare",
    "versus", " vs ", "size", "color", "alternative", "which one", "should i get", "worth it",
    "recommend", "recommendation", "cost", "price", "prices", "specs", "specification",
    "specifications", "rating", "ratings", "pros and cons", "quality", "reliable", "durable",
    "compare to", "compared to", "better than", "difference between", "good for", "suitable for",
    "fit for", "options for", "choices for", "alternatives to", "instead of", "or should i",
    "cheapest", "affordable", "value for money", "user review", "customer review", "expert review",
    "buying guide", "product comparison", "side by side", "which one should i get", "review of",
]

KEYWORDS_NAVIGATIONAL = [
    "name of", "what is the name", "what's the name", "find me", "find a", "website for",
    "url for", "link to", "app for", "where can i find", "where to find", "locate", "look up",
    "search for", "official site", "official website", "homepage", "login page", "sign in page",
    "which app", "which website", "which site", "which service", "which company", "brand name",
    "product name", "service name", "store name", "restaurant name", "address of", "contact for",
    "customer service number", "phone number for", "email for",
]

KEYWORDS_INFORMATIONAL = [
    "how", "what", "who", "where", "why", "guide", "tutorial", "resource", "help", "ideas",
    "tips", "learn", "examples", "explain", "meaning of", "definition of", "understand",
    "describe", "tell me about", "overview", "introduction", "intro", "basics", "step by step",
    "walkthrough", "outline", "summarize", "summary", "clarify", "concept", "theory", "process",
    "procedure", "difference between", "when to use", "how it works", "can you explain",
    "what does", "why do", "how do i", "where do i", "what are", "how are", "why are",
    "teach me", "show me how", "get started", "beginner", "advanced", "reference",
    "documentation", "faq", "frequently asked",
]

# Order: transactional and commercial_investigation first, then navigational, then informational (fallback).
MAJOR_INTENT_RULES: list[tuple[str, list[str]]] = [
    ("transactional", KEYWORDS_TRANSACTIONAL),
    ("commercial_investigation", KEYWORDS_COMMERCIAL_INVESTIGATION),
    ("navigational", KEYWORDS_NAVIGATIONAL),
    ("informational", KEYWORDS_INFORMATIONAL),
]

DEFAULT_MAJOR_INTENT = "informational"

# ---------------------------------------------------------------------------
# Sub-intent rules (per major intent; first match wins within that major)
# ---------------------------------------------------------------------------

KEYWORDS_CODING = [
    "code", "function", "script", "python", "javascript", "programming", "debug", " api ", "sql",
    "regex", "algorithm", "implement", "bug", "error in my code", "variable", "loop", "class",
    "import", "syntax", "compile", "runtime", "function call", "return value", "array", "list",
    "dict", "string", "integer", "boolean", "framework", "library", "package", "npm", "pip",
    "git", "branch", "merge", "refactor", "test case", "unit test", "exception", "stack trace",
    "console.log", "print statement", "endpoint", "request", "response", "database", "query",
]

KEYWORDS_CREATIVE_WRITING = [
    "write a", "write me", "story", "poem", "essay", "dialogue", "character", "plot", "fiction",
    "creative", "song lyrics", "script for", "short story", "novel", "scene", "chapter",
    "narrative", "protagonist", "antagonist", "setting", "theme", "metaphor", "rhyme", "verse",
    "stanza", "draft", "rewrite", "edit", "proofread", "prompt for", "idea for a story",
    "opening line", "ending", "twist",
]

KEYWORDS_EDUCATION = [
    "explain", "how does", "what is", "why does", "learn", "teach", "lesson", "homework",
    "assignment", "study", "definition of", "meaning of", "course", "textbook", "concept",
    "theory", "exam", "test", "quiz", "practice", "understand", "describe", "summarize",
    "outline", "basics", "introduction", "step by step", "clarify", "difference between",
    "when to use", "how it works", "reference", "documentation", "faq",
]

KEYWORDS_SUPPORT = [
    "help me", "fix my", "not working", "broken", "issue", "problem with", "error", "support",
    "troubleshoot", "how do i fix", "why is my", "fix this", "resolve", "solution", "failed",
    "failure", "crash", "crashed", "doesn't work", "won't work", "not responding", "stuck",
    "hang", "freeze", "slow", "timeout", "connection refused", "permission denied", "invalid",
    "corrupted", "restore", "recover", "reset", "reinstall", "update failed", "install error",
]

# Informational sub-categories in priority order: coding, creative_writing, education, support, then casual_other.
SUB_INTENT_RULES: dict[str, list[tuple[str, list[str]]]] = {
    "informational": [
        ("coding", KEYWORDS_CODING),
        ("creative_writing", KEYWORDS_CREATIVE_WRITING),
        ("education", KEYWORDS_EDUCATION),
        ("support", KEYWORDS_SUPPORT),
        ("casual_other", []),  # fallback
    ],
    "commercial_investigation": [
        ("commercial_product", KEYWORDS_COMMERCIAL_INVESTIGATION),
    ],
    "transactional": [
        ("transactional", KEYWORDS_TRANSACTIONAL),
    ],
    "navigational": [
        ("navigational", KEYWORDS_NAVIGATIONAL),
    ],
}


def _match_keywords(text: str, keywords: list[str]) -> bool:
    """True if any keyword (case-insensitive) appears in text."""
    if not text or not keywords:
        return False
    t = text.lower().strip()
    return any(kw.lower() in t for kw in keywords)


def assign_major_intent(
    text_series: pd.Series,
    major_rules: Optional[list[tuple[str, list[str]]]] = None,
    default: str = DEFAULT_MAJOR_INTENT,
) -> pd.Series:
    """
    Assign major intent per text using keyword rules (first match wins).
    Returns a Series with index aligned to text_series.
    """
    major_rules = major_rules or MAJOR_INTENT_RULES
    labels = []
    for _, text in text_series.items():
        t = (text or "").strip()
        if not t:
            labels.append(default)
            continue
        assigned = False
        for label, keywords in major_rules:
            if _match_keywords(t, keywords):
                labels.append(label)
                assigned = True
                break
        if not assigned:
            labels.append(default)
    return pd.Series(labels, index=text_series.index)


def assign_sub_intent(
    text_series: pd.Series,
    major_series: pd.Series,
    sub_rules: Optional[dict[str, list[tuple[str, list[str]]]]] = None,
) -> pd.Series:
    """
    Assign sub-intent per text given its major intent. Uses SUB_INTENT_RULES for that major;
    first matching sub-rule wins. If major has no sub-rules or no match, sub = major (or casual_other for informational).
    Returns a Series with index aligned to text_series.
    """
    sub_rules = sub_rules or SUB_INTENT_RULES
    labels = []
    for idx in text_series.index:
        text = text_series.loc[idx]
        major = major_series.loc[idx] if idx in major_series.index else DEFAULT_MAJOR_INTENT
        t = (text or "").strip()
        rules_for_major = sub_rules.get(major, [])
        if not rules_for_major:
            labels.append(major)
            continue
        assigned = False
        for sub_label, keywords in rules_for_major:
            if not keywords and sub_label == "casual_other":
                labels.append(sub_label)
                assigned = True
                break
            if keywords and _match_keywords(t, keywords):
                labels.append(sub_label)
                assigned = True
                break
        if not assigned:
            # fallback: use first sub that has no keywords (casual_other) or last sub
            for sub_label, keywords in rules_for_major:
                if not keywords:
                    labels.append(sub_label)
                    assigned = True
                    break
            if not assigned:
                labels.append(rules_for_major[-1][0])
    return pd.Series(labels, index=text_series.index)


def assign_intent_with_sub(
    df: pd.DataFrame,
    text_col: str = "text",
) -> pd.DataFrame:
    """
    Add intent_major and intent_sub columns to df using query text in text_col.
    Returns a copy of df with those columns added.
    """
    out = df.copy()
    out["intent_major"] = assign_major_intent(out[text_col])
    out["intent_sub"] = assign_sub_intent(out[text_col], out["intent_major"])
    return out


# ---------------------------------------------------------------------------
# Save / load by category
# ---------------------------------------------------------------------------

def save_classified_by_category(
    df: pd.DataFrame,
    output_dir: str | Path,
    major_col: str = "intent_major",
    sub_col: str = "intent_sub",
    full_name: str = "intent_classified.parquet",
) -> None:
    """
    Save full classified dataframe and per-category subsets.
    - Full: output_dir / full_name
    - By major: output_dir / "by_major" / {major}.parquet
    - By sub: output_dir / "by_sub" / {sub}.parquet
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "by_major").mkdir(parents=True, exist_ok=True)
    (output_dir / "by_sub").mkdir(parents=True, exist_ok=True)

    out_path = output_dir / full_name
    df.to_parquet(out_path, index=False)

    for major in df[major_col].dropna().unique():
        subset = df[df[major_col] == major]
        safe_name = str(major).replace(" ", "_")
        subset.to_parquet(output_dir / "by_major" / f"{safe_name}.parquet", index=False)

    for sub in df[sub_col].dropna().unique():
        subset = df[df[sub_col] == sub]
        safe_name = str(sub).replace(" ", "_")
        subset.to_parquet(output_dir / "by_sub" / f"{safe_name}.parquet", index=False)


def load_classified(output_dir: str | Path, full_name: str = "intent_classified.parquet") -> pd.DataFrame:
    """Load the full classified parquet from output_dir."""
    path = Path(output_dir) / full_name
    if not path.exists():
        raise FileNotFoundError(f"Classified file not found: {path}")
    return pd.read_parquet(path)


def load_category(
    output_dir: str | Path,
    category_name: str,
    which: str = "major",
) -> pd.DataFrame:
    """
    Load a single category subset. which must be 'major' or 'sub'.
    category_name is the intent label (e.g. 'informational', 'coding').
    """
    output_dir = Path(output_dir)
    if which == "major":
        path = output_dir / "by_major" / f"{category_name.replace(' ', '_')}.parquet"
    elif which == "sub":
        path = output_dir / "by_sub" / f"{category_name.replace(' ', '_')}.parquet"
    else:
        raise ValueError(f"which must be 'major' or 'sub', got {which!r}")
    if not path.exists():
        raise FileNotFoundError(f"Category file not found: {path}")
    return pd.read_parquet(path)
