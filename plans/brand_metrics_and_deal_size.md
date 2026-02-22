# Brand metrics, survival-style analytics, and deal size

## Overview

Extend the commercial vertical + brands pipeline to compute per-brand mention dynamics (first round/role, persistence, user follow-up), add product and deal-size fields via LLM (with optional search), and persist all new metrics as columns or enriched JSON before saving to parquet.

## Current state

- [commercial_vertical_brands_llm.ipynb](../commercial_vertical_brands_llm.ipynb) loads commercial (or subcategory) data, runs one LLM pass that returns `vertical_tier1_llm`, `vertical_tier2_llm`, and `brands` (list of `{name, where}` with `where` in `query_only` | `answer_only` | `both`).
- Conversation structure (from [README.md](../README.md) and [insights_utils.py](../insights_utils.py)): each row has `conversation` = list of messages; each message has `role` ("user" | "assistant") and `content`. Message order is 0, 1, 2, … (no per-message timestamps).
- Save step writes the table to parquet under `vertical_output`; `brands` is serialized as a JSON string.

## Metrics to compute and store

Two levels: **per-brand within a conversation** (multiple brands per row) and **per-conversation summary** (single value per row for easy filtering/analysis).

### 1. Per-brand metrics (store in enriched `brands` structure)

| Metric | Type | Description |
|--------|------|-------------|
| `first_mention_round` | int | 0-based message index where this brand first appears. |
| `first_mention_role` | str | "user" or "assistant" for that message. |
| `mention_count_total` | int | Number of messages in the conversation that mention this brand. |
| `last_mention_round` | int | Message index of last mention (for persistence). |
| `mention_span` | int | `last_mention_round - first_mention_round` (how long the brand keeps appearing). |
| `user_follow_up` | bool | True if the user mentioned this brand in any message *after* the first mention (by either side). |
| `brand_mentioned_once` | bool | True if brand appears in only one message (never again). |
| `conversation_ended_soon_after` | bool | True if the last message in the conversation is within 1–2 messages after first mention (optional; useful for "quick exit"). |

These can be stored in an expanded list, e.g. `brands_enriched`: each element = current `{name, where}` plus the new keys above. Kept in a single column (list of dicts) and serialized as JSON when saving.

### 2. Per-conversation summary columns (flat columns for analysis)

| Column | Type | Description |
|--------|------|-------------|
| `first_brand_first_round` | int or None | For the first-mentioned brand (by message order), the message index of first mention. |
| `first_brand_first_role` | str or None | "user" or "assistant" for that brand. |
| `any_brand_user_introduced` | bool | At least one brand first appeared in a user message. |
| `any_brand_assistant_introduced` | bool | At least one brand first appeared in an assistant message. |
| `any_user_follow_up_on_brand` | bool | For at least one brand, the user had a follow-up mention. |
| `total_brands` | int | Number of distinct brands in the conversation. |
| `n_messages` | int | Length of `conversation` (for survival/context). |

These are scalar columns; no JSON.

### 3. Product and deal-size metrics (per conversation or per primary brand)

| Column | Type | Description |
|--------|------|-------------|
| `product_mentioned` | str or None | Specific product name if explicitly mentioned in the conversation; else None. |
| `product_inferred` | str or None | If no product mentioned, LLM (or search) inferred most relevant product for the brand/context; else None. |
| `deal_size_usd` | float or None | Estimated potential purchase/deal size in USD. None only when no reasonable inference possible. |

Implementation choice: compute **per conversation** (one product/deal per row) for the "primary" or most salient brand, to avoid explosion of columns and to keep parquet one row per conversation. If multiple brands need product/deal, store as part of `brands_enriched` (e.g. each brand dict gets `product_mentioned`, `product_inferred`, `deal_size_usd`).

Recommendation: start with **per-conversation** product/deal (primary brand only); optionally later add per-brand product/deal inside `brands_enriched`.

---

## Implementation approach

### Phase A: Brand mention dynamics (deterministic post-processing)

- **Where:** New functions in [commercial_vertical_utils.py](../commercial_vertical_utils.py) (or a small helper module used by the notebook).
- **Input:** One row: `conversation` (list of `{role, content}`), `brands` (list of `{name, where}`).
- **Logic:**
  - For each message index `i` and each brand name, detect if the brand (normalized: case-insensitive, maybe strip punctuation) appears in `content`.
  - For each brand: first index where it appears → `first_mention_round`, `first_mention_role`; last index → `last_mention_round`; count of indices → `mention_count_total`; `mention_span` = last − first.
  - `user_follow_up`: for that brand, after `first_mention_round`, is there any message with `role == "user"` whose content contains the brand?
  - `brand_mentioned_once` = (mention_count_total == 1).
  - `conversation_ended_soon_after` = e.g. (last message index) − (first_mention_round) ≤ 1 (or 2).
- **Output:** Same row with `brands_enriched` (list of dicts with extra keys) and the new scalar summary columns above. Do **not** remove existing `brands` until downstream is updated; then you can keep only `brands_enriched` if desired.
- **Notebook:** After the LLM cell that produces `vertical_tier1_llm`, `vertical_tier2_llm`, `brands`, add a cell that runs this enrichment over `df` (with `conversation` and `brands`) and assigns the new columns.

### Phase B: Product and deal-size (LLM with optional search)

- **Option (1) – LLM-only (recommended first):**  
  One additional LLM pass (or extend the existing vertical+brands prompt) that, for each conversation, returns:
  - `product_mentioned`: string or null (explicit product name in the thread).
  - `product_inferred`: string or null (if no product mentioned, most relevant product for the brand/context).
  - `deal_size_usd`: number or null (estimated deal/purchase value in USD; instruct model to use typical price ranges for that product category and to prefer a point estimate or mid-range).
  Prompt instructions: "Do not leave deal_size_usd null unless the conversation has no commercial intent or no identifiable product; prefer a plausible range mid-point (e.g. $50 for a course, $500 for a laptop)."
- **Option (2) – LLM + external search:**  
  If LLM often returns null for `deal_size_usd`, add a step that, for the inferred/mentioned product or brand, calls a search API (e.g. SerpAPI, or a product/price dataset) and fills in a typical price; then store that in `deal_size_usd` and optionally keep an `deal_size_source` column (e.g. "llm" vs "search").
- **Storage:** Add columns `product_mentioned`, `product_inferred`, `deal_size_usd` (and optionally `deal_size_source`) to the table before save. If later you add per-brand product/deal, add the same fields into each element of `brands_enriched` and keep the flat columns for "primary" brand only.

### Phase C: Save and schema

- **Columns to add before `to_parquet`:**
  - Enriched brands: `brands_enriched` (JSON-serialized list of dicts).
  - Summary: `first_brand_first_round`, `first_brand_first_role`, `any_brand_user_introduced`, `any_brand_assistant_introduced`, `any_user_follow_up_on_brand`, `total_brands`, `n_messages`.
  - Product/deal: `product_mentioned`, `product_inferred`, `deal_size_usd` (and optionally `deal_size_source`).
- Keep existing columns (`conversation_id`, `conversation`, `vertical_tier1_llm`, `vertical_tier2_llm`, `brands`, etc.). Decide whether to keep `brands` as-is for backward compatibility or to replace it with `brands_enriched` in the file (recommend keeping both during transition: `brands` for compatibility, `brands_enriched` for new analytics).

---

## Suggested order of work

1. **Implement brand-dynamics helpers** in `commercial_vertical_utils.py`: e.g. `enrich_brands_with_mention_dynamics(conversation, brands) -> (brands_enriched, summary_dict)`.
2. **Notebook:** After LLM vertical+brands, apply enrichment to each row and assign `brands_enriched` and the summary columns.
3. **Implement product/deal LLM:** Either extend the existing LLM response schema with `product_mentioned`, `product_inferred`, `deal_size_usd` or add a second, cached LLM pass that reads conversation + brands and outputs these three. Prefer extending the same call to avoid double token cost and to keep one cache key per conversation.
4. **Notebook:** Add product/deal columns to `df` (from extended or second LLM), then run the existing save cell (and extend it to serialize `brands_enriched` and any new list/dict columns to JSON).
5. **Optional:** If many `deal_size_usd` are null, add a search-based fallback and `deal_size_source`.

---

## Metrics summary (for survival / business potential)

- **Survival / engagement:** `first_mention_round`, `first_mention_role`, `mention_span`, `user_follow_up`, `brand_mentioned_once`, `conversation_ended_soon_after`, `any_user_follow_up_on_brand`, `n_messages`.
- **Business potential:** `product_mentioned`, `product_inferred`, `deal_size_usd` (and optional `deal_size_source`).
- **Compatibility:** Keep `brands`; add `brands_enriched` and flat summary columns so existing parquet consumers still work and new analyses can use the enriched fields.
