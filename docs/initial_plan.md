---
name: WildChat EDA Plan
overview: Plan for building an exploratory data analysis (EDA) notebook for the WildChat conversation dataset, covering standard practices for LLM conversation datasets and WildChat-specific analyses (moderation, languages, turns, content length).
todos: []
isProject: false
---

# WildChat Exploratory Data Analysis Plan

## What people usually do with LLM conversation datasets

Common EDA steps for conversation/chat datasets with LLMs:

1. **Load and schema** – Load data (from disk or Hugging Face Hub), inspect columns and dtypes, view sample rows.
2. **Scale and coverage** – Count conversations, date range, train/val/test splits if any.
3. **Conversation shape** – Distribution of turns per conversation; single-turn vs multi-turn; user/assistant balance.
4. **Message length** – Length (characters/tokens) per message and per conversation; user vs assistant; histograms and percentiles.
5. **Temporal** – Volume over time (daily/weekly); timezone or hour patterns if relevant.
6. **Demographics / metadata** – Any user or model metadata (here: `model`); language distribution.
7. **Quality and filters** – Empty or near-empty inputs; duplicate or near-duplicate content; label/flag distributions (e.g. toxicity, redaction).
8. **Content snapshots** – Sample conversations by turn count or length; spot-check formats and edge cases.
9. **Downstream use** – Summary stats that inform train/val splits, filtering, or data selection for fine-tuning.

WildChat adds: **moderation scores** (OpenAI + Detoxify), **per-utterance and per-conversation language**, **redaction/PII flags**, and **empty user input** caveats. The dataset card notes [README.md](../README.md) that toxic conversations are removed in this release, so `toxic` is mostly false; moderation columns are still useful for distribution and sanity checks.

---

## Proposed notebook structure for `explore.ipynb`

### Code organization: notebook vs Python module

- **Keep the notebook thin.** Do not implement large chunks of Python logic inside the notebook (e.g. multi-line functions, nested loops for aggregation, parsing of nested `conversation`/moderation columns).
- **Put substantial logic in a separate `.py` file** (e.g. `eda_utils.py` or `explore_utils.py` in the project root). That module should define functions for: sampling the dataset by conversation; computing per-conversation or per-message stats; building the step 3–9 tables (basic counts, message length, temporal, moderation, empty inputs, keyword counts, summary).
- **In the notebook:** Only (1) set config (SAMPLE_N, SAMPLE_PCT, LOAD_FROM_SAVED, EDA_OUTPUT_DIR), (2) load the dataset and call the sampling function, (3) call the utility functions and pass the sampled data / load from disk, (4) display tables and plots. Keep cells short and readable.

### 0. Configuration (user-editable at top)

- **Sampling**: Two mutually exclusive, user-controllable options (e.g. in one cell):
  - `SAMPLE_N = 5000` — use exactly this many conversations (default); set to `None` to use percentage instead.
  - `SAMPLE_PCT = None` — use this fraction of the dataset (e.g. `0.01` for 1%); used only when `SAMPLE_N is None`.
  - So: either set `SAMPLE_N = 5000` and leave `SAMPLE_PCT = None`, or set `SAMPLE_N = None` and `SAMPLE_PCT = 0.01` (or similar). Final sample size = `SAMPLE_N` if set, else `ceil(len(dataset) * SAMPLE_PCT)`.
- **Rerun / load from saved**: For each major step that produces a table, a flag in that step's cell(s), e.g. `LOAD_FROM_SAVED = False`. If `True`, load the table from the output folder; if `False`, run the analysis and then save the table. Makes the workflow reproducible and skippable for long steps.
- **Output folder**: e.g. `EDA_OUTPUT_DIR = "eda_output"`. All saved mid-stage dataset tables and optional plots go here. **Do not use `data/`** for these outputs—keep them in a separate folder (e.g. `eda_output/`) so raw/processed data in `data/` stays distinct.

### 1. Setup and load

- Install/import: `datasets`, `pandas`, `matplotlib`, `seaborn` (or `plotly`).
- Load WildChat:  
`datasets.load_dataset("allenai/WildChat", split="train")`  
or load from local path if you downloaded parquet (e.g. under `data/`).
- **Sampling (conversation-level, whole conversations only)**:
  - Sample **by row (by conversation)**, not by message. That way every selected row keeps its full `conversation` list—no conversation is split (no "some messages in, some out").
  - Compute target size: `n = SAMPLE_N if SAMPLE_N is not None else max(1, int(len(dataset) * SAMPLE_PCT))`; cap at `len(dataset)`.
  - Use a fixed `random_state` for reproducibility. With `datasets`: `shuffled = dataset.shuffle(seed=42)` then `sampled = shuffled.select(range(n))`, or `indices = random.Random(42).sample(range(len(dataset)), k=min(n, len(dataset)))` then `dataset.select(indices)`.
  - Result: one Dataset (or dataframe) with only full conversations; all messages of a conversation stay together.
- Convert the sampled dataset to pandas for downstream steps if desired: `df = sampled.to_pandas()`.

### 2. Schema and sample

- `dataset.features` and `dataset.column_names`.
- Show 1–2 full rows (e.g. one short and one multi-turn) to inspect `conversation`, `openai_moderation`, `detoxify_moderation` structure.
- Optional: flatten one conversation into a small table (turn index, role, content length, language) for clarity.

### Saving outputs and rerun/load pattern (for steps 3–9)

- **Output directory**: Use a folder **outside `data/`** (e.g. `EDA_OUTPUT_DIR = "eda_output"`). Create it if it doesn't exist (e.g. `Path(EDA_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)`). Mid-stage tables go here only—do not save them under `data/`.
- **Per-step behavior**: For each major step that produces a **pandas DataFrame**:
  - Assign a stable filename, e.g. `step3_basic_counts.parquet`, `step4_message_length.parquet`, under `EDA_OUTPUT_DIR`.
  - At the start of the step: if `LOAD_FROM_SAVED` is True and the file exists, load with `pd.read_parquet(path)` and skip computation; otherwise run the analysis to produce the table.
  - After computation: save with `df.to_parquet(path, index=False)`. Optionally save a small summary to a companion file.
- **Step-level flag**: Use one global `LOAD_FROM_SAVED` or per-step flags (e.g. `LOAD_STEP3`, `LOAD_STEP4`) so the user can rerun only some steps and load the rest.

### 3. Basic counts and distributions

- Total number of conversations.
- **Model**: value counts and bar chart (`gpt-3.5-turbo` vs `gpt-4`).
- **Turn**: distribution (min, max, mean, median); histogram or KDE; count of single-turn vs multi-turn.
- **Language**: top 10–20 languages by conversation count; bar chart; note "conversation-level" language (most frequent in utterances per conversation).
- **Save**: e.g. `eda_output/step3_basic_counts.parquet`. **Rerun/load**: if `LOAD_FROM_SAVED`, load from file; else compute and save.

### 4. Conversation and message length

- For each conversation: number of messages, total characters (and optionally tokens if you add a simple tokenizer).
- Per-message: extract length from `conversation[*].content`; separate user vs assistant; histograms and percentiles (e.g. 50, 90, 99).
- Optional: average user message length vs average assistant message length by conversation; scatter or box plot.
- **Save**: e.g. `eda_output/step4_message_length.parquet`. **Rerun/load**: same pattern.

### 5. Temporal distribution

- Parse `timestamp`; distribution of conversations by date (e.g. by month or week).
- Simple time series plot (conversation count over time).
- Optional: by `model` or by `language` over time if of interest.
- **Save**: e.g. `eda_output/step5_temporal.parquet`. **Rerun/load**: same pattern.

### 6. Moderation and quality flags

- **Toxic**: value counts (expect mostly false after dataset filtering).
- **Redacted**: value counts; fraction of conversations with PII redaction.
- **OpenAI moderation**: for each conversation, aggregate per-utterance `flagged` (e.g. any true); plot distribution of max or mean category scores across utterances (e.g. violence, sexual) if useful.
- **Detoxify**: same idea—per conversation, max or mean of `toxicity`, `severe_toxicity`, etc.; small histograms to confirm "clean" post-filter distribution.
- **Save**: e.g. `eda_output/step6_moderation.parquet`. **Rerun/load**: same pattern.

### 7. Empty and short user inputs

- Dataset card: empty user inputs exist. Identify conversations where at least one user message has empty or whitespace-only `content`.
- Count and percentage; optionally list a few `conversation_id`s for inspection.
- Short user message length (e.g. &lt; 5 chars) distribution to see how many are "near-empty".
- **Save**: e.g. `eda_output/step7_empty_inputs.parquet`. **Rerun/load**: same pattern.

### 8. Content sampling and sanity checks

- Sample 3–5 conversations: single-turn, 3-turn, and long (e.g. &gt; 10 turns); print role and content (truncate long content).
- Check for obvious formatting quirks (e.g. markdown, code blocks, multiple languages in one conversation).
- Optional: simple keyword or regex counts (e.g. "code", "translate", "write") to get a rough topic signal.
- **Save**: e.g. `eda_output/step8_keyword_counts.parquet` if keyword stats are computed. **Rerun/load**: same pattern for any computed table.

### 9. Summary table and notes

- One table: total conversations, total turns, date range, top-3 languages, model counts, % redacted, % with empty user message, median/mean message length (user vs assistant).
- **Save**: e.g. `eda_output/step9_summary.parquet`. **Rerun/load**: same pattern.
- Short markdown: main takeaways and any filters you'd apply for a downstream task (e.g. instruction tuning).

---

## Implementation notes

- **Code layout**: Implement non-trivial logic (sampling, aggregation, table building for steps 3–9) in a single Python module (e.g. `eda_utils.py`). The notebook imports from it and only runs config, calls, and display. This keeps the notebook clean and the logic testable/reusable.
- **Output location**: Save all mid-stage EDA tables under `EDA_OUTPUT_DIR` (e.g. `eda_output/`). Do **not** use `data/` for these outputs.
- **Memory**: With 529k rows and nested lists, either use `datasets` iterators / `select` to subsample (e.g. 5k default, or a user-controlled percentage) for expensive operations, or run on a machine with enough RAM to hold the full dataframe.
- **Sampling**: Always sample by **conversation (row)**, never by message, so that every selected row retains its full `conversation` list—no conversation is split across sample vs rest.
- **Nested columns**: Use list comprehensions or `dataset.map()` to compute per-conversation stats (turn count, message lengths, empty-user flag) and attach to a shallow table for plotting.
- **Reproducibility**: Set a random seed when sampling; document dataset version (e.g. "WildChat train, post–content update 2024-10-17").
- **References**: Dataset card and papers (WildChat ICLR 2024, WildVis EMNLP 2024) are linked in [README.md](../README.md); cite them if you use the dataset in a report or paper.

---

## Optional extensions (if time permits)

- **Code-switching**: Flag conversations where utterance-level `language` changes within the same conversation; count and show examples.
- **Response length vs model**: Compare assistant message length by `model` (gpt-4 vs gpt-3.5-turbo).
- **Correlation**: Turn count vs total length; or mean user length vs mean assistant length.
- **Export**: Save a small "clean" subset (e.g. English-only, 2–10 turns, no redaction) as a parquet or JSONL for training experiments.

This structure gives you a repeatable EDA that matches common practice for LLM conversation data and leverages WildChat's moderation, language, and turn metadata.
