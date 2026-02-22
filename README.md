# WildChat research: intent, verticals, and insights

This repository contains notebooks and utilities for analyzing [WildChat](https://huggingface.co/datasets/allenai/WildChat) conversation data: intent classification, commercial vertical and brand extraction, and downstream insights. Run the notebooks in the order below (skip `legacy_code/`).

---

## Pipeline overview

| Step | Notebook | Input | Output |
|------|----------|--------|--------|
| 1 | **explore.ipynb** | Hugging Face / `data/` | `eda_output/`, `data/english_chunks/` |
| 2 | **intent_classification_llm.ipynb** | `data/english_chunks/` | `intent_output/` |
| 3 | **intent_trends.ipynb** | `intent_output/` | (plots only) |
| 4 | **commercial_vertical_brands_llm.ipynb** | `intent_output/` | `vertical_output/` |
| 5 | **vertical_insights.ipynb** | `vertical_output/` | `vertical_insights_plots/` |

**Data flow:**  
`Hugging Face / data/` → **explore** → `data/english_chunks/` → **intent_classification_llm** → `intent_output/` → **intent_trends** (read-only) and **commercial_vertical_brands_llm** → `vertical_output/` → **vertical_insights**.

---

## 1. explore.ipynb — EDA and English export

- Loads WildChat from Hugging Face and builds EDA tables (counts, message length, temporal, moderation, etc.) under `eda_output/`.
- **Export English-only:** filters to English and writes chunked parquet to `data/english_chunks/`. Run this once so downstream notebooks can load from disk instead of Hugging Face.

---

## 2. intent_classification_llm.ipynb — Intent classification (LLM)

- **Input:** `data/english_chunks/` (from explore’s Export English-only).
- Adds **intent_major** and **intent_sub** (e.g. informational, commercial_investigation, transactional, education) via LLM.
- **Output:** `intent_output/` — full table and/or per-category parquets in `by_major/` and `by_sub/` (optionally by index range, e.g. `0_10000`).

---

## 3. intent_trends.ipynb — Intent trends over time

- **Input:** `intent_output/all` (or range subfolders).
- Visualizes weekly intent category and subcategory percentages over time (UTC). No output files.

---

## 4. commercial_vertical_brands_llm.ipynb — Commercial vertical and brands

- **Input:** `intent_output/` — loads by **major category** (e.g. commercial_investigation, transactional) or by **subcategory** (e.g. education) from `by_major/` or `by_sub/`. Supports index range or all; optional **random sample** of N rows (e.g. 5000).
- Adds **vertical_tier1_llm**, **vertical_tier2_llm**, **brands** (with `where`: query_only / answer_only / both), brand mention dynamics, and optional product/deal size via LLM.
- **Output:** `vertical_output/{range}/` (e.g. `vertical_output/all/`) — one parquet per category/subcategory run.

---

## 5. vertical_insights.ipynb — Vertical insights

- **Input:** `vertical_output/all/` — all parquet files there (one per category/subcategory).
- Survival-style analyses (conversation length, brand mention span), deal-size comparison, user follow-up % by vertical; tables and plots.
- **Output:** `vertical_insights_plots/` — PNGs when `SAVE_PLOTS` is True.

---

## Key modules

- **intent_taxonomy.py** — Intent labels, loading by category/subcategory from `intent_output` (by range or all).
- **commercial_vertical_utils.py** — All-queries/answers columns, vertical and brand LLM prompting, parallel runs, cache.
- **eda_utils.py** — Conversation parsing, EDA table builders.
- **intent_analysis_utils.py** — Conversation normalization and analysis helpers.
- **insights_utils.py** — Vertical insights aggregation and plotting (e.g. survival, deal size, follow-up).

---

## Requirements

Install dependencies (e.g. `pandas`, `openai`, `tqdm`, Hugging Face `datasets`) as needed for each notebook. Control variables (paths, categories, sampling, batch sizes) are at the top of each notebook; run the control cell first, then “Run All” for reproducible runs.
