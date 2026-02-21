# Commercial investigation vertical extraction notebook

## Goal

Build a notebook that answers: **which business verticals (e.g. sports, health, real estate) do commercial_investigation conversations belong to?** Use taxonomy-based matching, LLM-based extraction (with concise structured output and parallel calls, plus intent re-classification), and one or more other methods.

---

## 1. Data loading and sampling

- **Source**: Load commercial_investigation from the existing intent output via `intent_taxonomy.load_category` (e.g. `load_category(INTENT_OUTPUT_DIR, "commercial_investigation", which="major")`). Path: `intent_output/by_major/commercial_investigation.parquet`.
- **Control variables** (top-of-notebook cell): `INTENT_OUTPUT_DIR = "intent_output"`, `USE_SAMPLE = True`, `SAMPLE_N = 2000`, `RANDOM_SEED = 42`.
- **Parsing**: After load, run `eda_utils.ensure_conversation_parsed` so the `conversation` column is list-of-dicts. If sampling, use a simple random sample.
- **Text inputs**: Use **all user messages (all queries)** and **all assistant messages (all answers)**. Add `get_all_assistant_content(conv, sep=" ")` in `insights_utils.py`. Add columns `all_queries` and `all_answers`.

## 2. Taxonomy (IAB keyword-based)

- Use **IAB Content Taxonomy** for business vertical (content aboutness). Load from IAB GitHub TSV or minimal tier1/tier2 CSV in `data/`. Keyword match over all_queries + all_answers; assign `vertical_tier1_iab`, `vertical_tier2_iab`.

## 3. LLM-based (structured output, intent re-classification, parallel)

- Pass all_queries + all_answers into each LLM call. Structured JSON: `vertical_tier1`, `vertical_tier2`, `intent_revised`. Parallel with ThreadPoolExecutor; cache by conversation_id.

## 4. Other approaches

- **Embedding-based**: Embed all_queries + all_answers; assign vertical by cosine similarity to reference verticals.
- **Rule-based**: Curated vertical keywords; first-match over all_queries + all_answers.

## 5. Notebook structure

- Control variables at top; sections: Load + parse + sample; Add all_queries/all_answers; IAB taxonomy; LLM; Embedding; Rule-based; Summary (value_counts, agreement, intent_revised vs intent_major). Logic in `.py`; notebook imports and calls.

## 6. Files

- New notebook: `commercial_vertical_extraction.ipynb`
- New module: `commercial_vertical_utils.py` (IAB loader, assign_vertical_iab, label_vertical_intent_llm_parallel, assign_vertical_embedding, assign_vertical_rule_based)
- `insights_utils.py`: add `get_all_assistant_content`
- IAB data: `data/iab_content_taxonomy_tier1_tier2.csv` (minimal) or load from IAB GitHub URL
