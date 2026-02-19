# WildChat In-Depth Insights Analysis Plan

## Scope and constraints

- **New notebook only**: Create a new notebook (`insights_analysis.ipynb`); do not modify `explore.ipynb`.
- **Logic in .py**: Implement analysis and API calls in a new Python module so the notebook stays thin (config, load, call helpers, display).
- **Plan doc**: This document lives in `docs/insights_analysis_plan.md`.
- **Secrets**: Use `.env.example`; user copies to `.env` and loads via `python-dotenv`.

Data source: WildChat train split (~529K conversations). Each row has `conversation` (list of `{role, content, language, ...}`), `timestamp`, `model`, `language`, etc. Reuse conversation-level sampling (e.g. from `eda_utils.sample_by_conversation`) so analyses run on a configurable sample (e.g. 5K–20K) to control cost and runtime.

---

## 1. Environment and dependencies

- **`.env.example`**: `OPENAI_API_KEY=` (required for LLM-based labeling); optional `HF_TOKEN=`. Comment: copy to `.env` and fill in.
- **`.gitignore`**: `.env` is already ignored.
- **Dependencies**: `openai`, `python-dotenv`, `scikit-learn` in `requirements.txt`; optional `sentence-transformers` for local embeddings.

---

## 2. Code layout

| Asset | Purpose |
|-------|--------|
| **New notebook** (`insights_analysis.ipynb`) | Config (sample size, LOAD_FROM_SAVED, output dir, API flags); load dataset and sample; call module; display tables and plots. |
| **New Python module** (`insights_utils.py`) | Text extraction from `conversation`, aggregation helpers, topic modeling (LDA/NMF), OpenAI wrappers for brand extraction and purpose/theme labeling with caching. |
| **Plan doc** | This file. |

Notebook flow: load env from `.env`; load dataset → sample → optionally load saved parquet from `insights_output/`; run each analysis via module calls; save intermediate results for reruns.

---

## 3. Topic / industry and commercial content

### 3.1 Conversation purpose / domain (prerequisite)

- **Approach**: Small taxonomy (e.g. informational, educational, creative writing, coding, commercial/product, support, casual, other). Label via OpenAI zero-shot on first user message (or concatenated user content).
- **Output**: Per-conversation `purpose`; distribution by language, model, timestamp.
- **Implementation**: Helper to extract text to classify; batch OpenAI calls with caching (conversation_id → label parquet).

### 3.2 Brand names and commercial content

- **Brand extraction**: OpenAI prompt to list brand/company names as JSON array on user content. Per-conversation list of brands; aggregate to brand frequency table.
- **Industry of brands**: Lookup for known brands + OpenAI for unseen; cache. Industries: tech, health, finance, retail, media, etc.
- **Commercial without brand**: Keyword/heuristic (best, recommend, compare, buy, price, review) + optional LLM Yes/No. Output `has_commercial_intent`; analyze overlap with `has_brand_mention`.

---

## 4. Intent / semantic analysis (themes and trends)

### 4.1 Theme labeling

- **Fixed taxonomy**: 10–20 theme labels; OpenAI assigns one primary theme per conversation (first user message).
- **Output**: Per-conversation `theme`; theme frequency; optional time series.

### 4.2 Emerging vs underserved

- Volume over time per theme; identify emerging (rising) vs stable vs declining.
- Underserved: low volume themes; or share of conversations vs share of total turns.

---

## 5. Topic modeling (landscape and frequency)

- **Input**: First user message (or all user messages) per conversation; optionally English-only.
- **Methods**: TF-IDF + NMF or LDA (sklearn); K = 10–30. Optional: embeddings + K-means, label clusters with LLM.
- **Output**: Topic id per conversation; distribution (counts, %); top terms or representative snippets; trends over time.

---

## 6. OpenAI usage and caching

- **Batching**: Chunks of 100–500; retries with backoff.
- **Caching**: Save responses keyed by conversation_id under `insights_output/`; reruns skip already-labeled rows.
- **Cost control**: Sample size and “use API” flags in notebook.

---

## 7. Deliverables checklist

- [x] `.env.example` with OPENAI_API_KEY and optional HF_TOKEN
- [x] `.gitignore` contains `.env`
- [x] `docs/insights_analysis_plan.md` (this file)
- [x] `insights_utils.py`: text extraction, cache helpers, topic modeling, OpenAI helpers
- [x] `insights_analysis.ipynb`: config, load, sections for Topic/Industry, Intent/Themes, Topic Modeling
- [x] `requirements.txt`: openai, python-dotenv, scikit-learn

---

## 8. Implementation notes

- **File paths**: Module `insights_utils.py` in project root; notebook `insights_analysis.ipynb` in project root; outputs under `insights_output/` (e.g. `purpose_labels.parquet`, `brands.parquet`, `theme_labels.parquet`, `topic_assignments.parquet`).
- **Order of steps**: (1) .env.example + requirements + this doc; (2) insights_utils: text extraction, cache, topic modeling (no API); (3) OpenAI helpers with caching; (4) notebook sections.
- **Suggested sample sizes**: 5K for quick runs; 10K–20K for richer brand/theme stats. Use `LOAD_FROM_SAVED` to avoid re-calling API when iterating.
